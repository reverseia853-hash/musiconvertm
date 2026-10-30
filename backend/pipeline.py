"""
Pasos pesados del procesamiento:
  1. descargar el audio de un link de YouTube (yt-dlp)
  2. separar la voz del resto de la música (Demucs)
  3. transcribir con Whisper (faster-whisper) con tiempos por palabra

No depende de FastAPI, así que se puede probar por separado.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

ProgressFn = Callable[[float], None]
CancelledFn = Callable[[], bool]

SINGING_VAD_THRESHOLD = 0.15
NO_SPEECH_THRESHOLD = 0.6
TRANSCRIPTION_MODES = {"vad", "no_vad", "compare"}
SEPARATION_LEVELS = {
    "low": {"shifts": 1, "overlap": 0.1},
    "medium": {"shifts": 2, "overlap": 0.25},
    "high": {"shifts": 4, "overlap": 0.5},
}


class JobCancelled(Exception):
    """Raised when a user requests cancellation of a background job."""


YOUTUBE_RE = re.compile(
    r"^https?://(?:(?:www|m|music)\.)?(?:youtube\.com|youtu\.be)/", re.IGNORECASE
)


def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_RE.match(url.strip()))


def audio_duration(audio: Path) -> float:
    """Devuelve la duración del audio en segundos usando ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("No se encontró ffprobe. Instala ffmpeg y vuelve a intentarlo.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("No se pudo determinar la duración del audio a tiempo.") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "No se pudo leer la duración del audio: " + proc.stderr.strip()[-300:]
        )
    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("No se pudo determinar la duración del audio.") from exc
    if not 0 < duration < float("inf"):
        raise RuntimeError("El archivo de audio tiene una duración no válida.")
    return duration


# ---------------------------------------------------------------- YouTube
def _youtube_help(exc: Exception) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip()
    detail = text.splitlines()[-1][:250] if text else exc.__class__.__name__
    return (
        "No se pudo descargar el video de YouTube. Prueba actualizar yt-dlp "
        '(pip install -U "yt-dlp[default]") e instalar Deno '
        "(winget install DenoLand.Deno), y luego reinicia el servidor. "
        f"Detalle: {detail}"
    )


def download_youtube(
    url: str,
    job_dir: Path,
    on_progress: ProgressFn,
    max_minutes: int = 12,
    max_bytes: int | None = None,
    is_cancelled: CancelledFn | None = None,
) -> tuple[Path, str]:
    """Descarga el audio como mp3. Devuelve (ruta_del_audio, titulo)."""
    is_cancelled = is_cancelled or (lambda: False)
    if is_cancelled():
        raise JobCancelled()
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError('Falta yt-dlp. Ejecuta: pip install "yt-dlp[default]"') from exc

    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "no_color": True}

    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        if is_cancelled():
            raise JobCancelled() from exc
        raise RuntimeError(_youtube_help(exc)) from exc
    if is_cancelled():
        raise JobCancelled()

    if info.get("_type") == "playlist" or info.get("entries"):
        raise RuntimeError("Pega el link de un solo video, no de una lista de reproducción.")
    if info.get("is_live"):
        raise RuntimeError("No se pueden usar transmisiones en vivo.")
    duration = info.get("duration") or 0
    if duration > max_minutes * 60:
        raise RuntimeError(
            f"El video dura {int(duration // 60)} min; el máximo permitido es {max_minutes} min."
        )
    title = info.get("title") or "YouTube"

    def hook(d: dict) -> None:
        if is_cancelled():
            raise JobCancelled()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    opts = {
        **base,
        "format": "bestaudio/best",
        "max_filesize": max_bytes,
        "outtmpl": str(job_dir / "original.%(ext)s"),
        "progress_hooks": [hook],
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        if is_cancelled():
            raise JobCancelled() from exc
        raise RuntimeError(_youtube_help(exc)) from exc

    audio = job_dir / "original.mp3"
    if not audio.exists():
        found = sorted(job_dir.glob("original.*"))
        if not found:
            raise RuntimeError("La descarga terminó pero no se encontró el archivo de audio.")
        audio = found[0]
    return audio, title


# ----------------------------------------------------------------- Demucs
def separate_vocals(
    audio: Path,
    job_dir: Path,
    on_progress: ProgressFn,
    model: str = "htdemucs",
    is_cancelled: CancelledFn | None = None,
    separation_level: str = "low",
) -> Path:
    """Aísla la voz con Demucs (CPU). Devuelve la ruta de vocals.mp3."""
    is_cancelled = is_cancelled or (lambda: False)
    if is_cancelled():
        raise JobCancelled()
    if separation_level not in SEPARATION_LEVELS:
        raise ValueError(f"Nivel de separación no válido: {separation_level}")
    profile = SEPARATION_LEVELS[separation_level]
    out_dir = job_dir / "separated"
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", model,
        "--two-stems=vocals",
        "--mp3", "--mp3-bitrate", "192",   # evita depender de torchaudio.save
        "--shifts", str(profile["shifts"]),
        "--overlap", str(profile["overlap"]),
        "-d", "cpu",
        "-o", str(out_dir),
        str(audio),
    ]
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # por si torch es muy nuevo
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    assert proc.stdout is not None
    buf = b""
    best = 0

    def stop_process() -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    try:
        while True:
            if is_cancelled():
                raise JobCancelled()
            chunk = proc.stdout.read1(512)
            if not chunk:
                break
            buf = (buf + chunk)[-3000:]
            found = re.findall(rb"(\d{1,3})%\|", buf)  # barra de progreso (tqdm)
            if found:
                best = max(best, min(int(found[-1]), 100))
                on_progress(best / 100)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            stop_process()
        proc.stdout.close()
    if is_cancelled():
        raise JobCancelled()

    if code != 0:
        tail = buf.decode("utf-8", errors="replace").replace("\r", "\n").strip()[-600:]
        missing = re.search(r"No module named '([^']+)'", tail)
        if missing:
            name = missing.group(1)
            root = name.split(".")[0]
            if root == "demucs":
                raise RuntimeError("Demucs no está instalado. Ejecuta: pip install --no-deps demucs")
            raise RuntimeError(
                f"A Demucs le falta la librería '{root}'. "
                f"Ejecuta: pip install {root.replace('_', '-')} y reinicia el servidor."
            )
        if code in (3221227274, 3221225477, 3221225725, -1073741819, -1073740791):
            raise RuntimeError(
                f"Demucs se cerró de golpe (código {code}): casi siempre es falta de memoria RAM "
                "o un conflicto de PyTorch en Windows. Cierra otros programas y prueba una canción "
                "más corta; si persiste, reinstala: pip install --force-reinstall torch==2.5.1 torchaudio==2.5.1"
            )
        raise RuntimeError(f"Demucs falló (código {code}). {tail}")

    vocals = out_dir / model / audio.stem / "vocals.mp3"
    if not vocals.exists():
        found_files = sorted(out_dir.rglob("vocals.*"))
        if not found_files:
            raise RuntimeError("Demucs terminó pero no generó el archivo de voz.")
        vocals = found_files[0]
    return vocals


# ---------------------------------------------------------------- Whisper
def load_whisper(model_size: str):
    from faster_whisper import WhisperModel  # import tardío: arranca más rápido

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",  # lo más rápido en CPU
        cpu_threads=os.cpu_count() or 4,
    )


def cut_clip(src: Path, start: float, out: Path) -> None:
    """Recorta el audio desde `start` hasta el final (mono, 16 kHz) con ffmpeg."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(start, 0):.2f}", "-i", str(src),
           "-ac", "1", "-ar", "16000", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("No se encontró ffmpeg. Instálalo con: winget install ffmpeg") from exc
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg no pudo recortar el audio. {proc.stderr.strip()[-300:]}")


def transcribe(
    model,
    audio: Path,
    language: str,
    beam_size: int,
    on_progress: ProgressFn,
    vad: bool = True,
    offset: float = 0.0,
    is_cancelled: CancelledFn | None = None,
) -> list[dict]:
    """`vad=False` no descarta ningún tramo; `offset` suma segundos a todos los tiempos."""
    is_cancelled = is_cancelled or (lambda: False)
    segments, info = model.transcribe(
        str(audio),
        language=language or None,
        word_timestamps=True,
        vad_filter=vad,  # ignora silencios/instrumentales
        vad_parameters=(
            {"threshold": SINGING_VAD_THRESHOLD, "min_silence_duration_ms": 500}
            if vad else None
        ),
        condition_on_previous_text=False,  # evita bucles de texto repetido
        beam_size=beam_size,
        initial_prompt="Letra de una canción en español.",
        # Las canciones repiten estribillos y Whisper suele descartar esos tramos
        # por "sospechosos". Estos umbrales son más tolerantes que los de fábrica.
        compression_ratio_threshold=3.0,   # por defecto 2.4
        log_prob_threshold=-1.5,           # por defecto -1.0
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        temperature=[0.0, 0.2, 0.4, 0.6],  # reintenta con más variación si falla
    )
    duration = getattr(info, "duration", 0) or 1.0

    result: list[dict] = []
    for seg in segments:
        if is_cancelled():
            raise JobCancelled()
        words = [
            {
                "word": w.word.strip(),
                "start": round(w.start + offset, 2),
                "end": round(w.end + offset, 2),
            }
            for w in (seg.words or [])
            if w.word.strip()
        ]
        if words:
            result.append(
                {
                    "start": round(seg.start + offset, 2),
                    "end": round(seg.end + offset, 2),
                    "text": seg.text.strip(),
                    "words": words,
                }
            )
        on_progress(min(seg.end / duration, 1.0))
    if is_cancelled():
        raise JobCancelled()
    return result


def transcribe_mode(
    model,
    audio: Path,
    language: str,
    beam_size: int,
    on_progress: ProgressFn,
    mode: str,
    on_stage: Callable[[str], None] | None = None,
    is_cancelled: CancelledFn | None = None,
) -> tuple[list[dict], list[dict] | None]:
    """Transcribe with VAD, without VAD, or return both versions for review."""
    if mode not in TRANSCRIPTION_MODES:
        raise ValueError(f"Modo de transcripción no válido: {mode}")
    is_cancelled = is_cancelled or (lambda: False)

    def report_progress(progress: float) -> None:
        if is_cancelled():
            raise JobCancelled()
        on_progress(progress)

    def report_stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    if mode == "vad":
        report_stage("transcribing")
        return transcribe(model, audio, language, beam_size, report_progress, vad=True,
                          is_cancelled=is_cancelled), None

    if mode == "no_vad":
        report_stage("transcribing")
        return transcribe(model, audio, language, max(beam_size, 5), report_progress, vad=False,
                          is_cancelled=is_cancelled), None

    report_stage("transcribing")
    primary = transcribe(
        model, audio, language, beam_size,
        lambda progress: report_progress(progress * 0.5), vad=True, is_cancelled=is_cancelled,
    )
    if is_cancelled():
        raise JobCancelled()
    report_stage("transcribing_alternative")
    alternative = transcribe(
        model, audio, language, max(beam_size, 5),
        lambda progress: report_progress(0.5 + progress * 0.5), vad=False,
        is_cancelled=is_cancelled,
    )
    return primary, alternative
