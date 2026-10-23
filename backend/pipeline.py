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

YOUTUBE_RE = re.compile(
    r"^https?://(?:(?:www|m|music)\.)?(?:youtube\.com|youtu\.be)/", re.IGNORECASE
)


def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_RE.match(url.strip()))


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
) -> tuple[Path, str]:
    """Descarga el audio como mp3. Devuelve (ruta_del_audio, titulo)."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError('Falta yt-dlp. Ejecuta: pip install "yt-dlp[default]"') from exc

    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "no_color": True}

    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_youtube_help(exc)) from exc

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
    except Exception as exc:  # noqa: BLE001
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
    audio: Path, job_dir: Path, on_progress: ProgressFn, model: str = "htdemucs"
) -> Path:
    """Aísla la voz con Demucs (CPU). Devuelve la ruta de vocals.mp3."""
    out_dir = job_dir / "separated"
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", model,
        "--two-stems=vocals",
        "--mp3", "--mp3-bitrate", "192",   # evita depender de torchaudio.save
        "--overlap", "0.1",                # por defecto 0.25: menos trabajo repetido
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
    while True:
        chunk = proc.stdout.read1(512)
        if not chunk:
            break
        buf = (buf + chunk)[-3000:]
        found = re.findall(rb"(\d{1,3})%\|", buf)  # barra de progreso (tqdm)
        if found:
            best = max(best, min(int(found[-1]), 100))
            on_progress(best / 100)
    code = proc.wait()

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
) -> list[dict]:
    """`vad=False` no descarta ningún tramo; `offset` suma segundos a todos los tiempos."""
    segments, info = model.transcribe(
        str(audio),
        language=language or None,
        word_timestamps=True,
        vad_filter=vad,  # ignora silencios/instrumentales
        vad_parameters={"threshold": 0.3, "min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # evita bucles de texto repetido
        beam_size=beam_size,
        initial_prompt="Letra de una canción en español.",
        # Las canciones repiten estribillos y Whisper suele descartar esos tramos
        # por "sospechosos". Estos umbrales son más tolerantes que los de fábrica.
        compression_ratio_threshold=3.0,   # por defecto 2.4
        log_prob_threshold=-1.5,           # por defecto -1.0
        no_speech_threshold=0.8,           # por defecto 0.6
        temperature=[0.0, 0.2, 0.4, 0.6],  # reintenta con más variación si falla
    )
    duration = getattr(info, "duration", 0) or 1.0

    result: list[dict] = []
    for seg in segments:
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
    return result
