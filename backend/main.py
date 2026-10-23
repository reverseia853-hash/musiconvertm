"""
Musiconvert - Fase 2
Archivo de audio o link de YouTube
  -> (opcional) Demucs separa la voz del resto
  -> Whisper transcribe con tiempos por palabra
  -> el frontend lo muestra tipo karaoke
"""
import hashlib
import hmac
import math
import os
import secrets
import shutil
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import chords as chord_detect
import pipeline

# ---------- Configuración (se puede cambiar con variables de entorno) ----------
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny | base | small | medium
LANGUAGE = os.getenv("WHISPER_LANG", "es")        # "" para autodetectar
BEAM_SIZE = int(os.getenv("WHISPER_BEAM", "3"))
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
MAX_MINUTES = int(os.getenv("MAX_MINUTES", "12"))  # duración máxima de un video de YouTube
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "90")) * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
FREE_TIER_MODE = os.getenv("FREE_TIER_MODE", "false").strip().lower() in {"1", "true", "yes"}
DEFAULT_SEPARATE = os.getenv("DEFAULT_SEPARATE", "true").strip().lower() in {"1", "true", "yes"}
DEFAULT_CHORDS = os.getenv("DEFAULT_CHORDS", "true").strip().lower() in {"1", "true", "yes"}
AUTH_REQUIRED = os.getenv("APP_AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes"}
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
APP_SESSION_SECRET = os.getenv("APP_SESSION_SECRET", "")
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes"}
SESSION_MAX_AGE = 12 * 60 * 60
SESSION_COOKIE_NAME = "musiconvert_session"
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".mp4"}

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"

# Los trabajos viven en memoria; al arrancar se borran los archivos temporales viejos.
shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
UPLOADS_DIR.mkdir(exist_ok=True)


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            if key not in self._events and len(self._events) >= 10000:
                expired = [
                    client for client, entries in self._events.items()
                    if not entries or now - entries[-1] >= self.window_seconds
                ]
                for client in expired:
                    self._events.pop(client, None)
                if len(self._events) >= 10000:
                    return self.window_seconds
            events = self._events.setdefault(key, deque())
            while events and now - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= self.limit:
                return max(1, math.ceil(self.window_seconds - (now - events[0])))
            events.append(now)
        return 0

    def clear(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


login_limiter = SlidingWindowLimiter(limit=5, window_seconds=15 * 60)
job_limiter = SlidingWindowLimiter(limit=10, window_seconds=60 * 60)


def _client_key(scope: dict) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


def _session_signature(expires: str, nonce: str) -> str:
    message = f"{expires}.{nonce}".encode("ascii")
    return hmac.new(APP_SESSION_SECRET.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _valid_session(token: str | None) -> bool:
    if not token or not APP_SESSION_SECRET:
        return False
    try:
        expires, nonce, signature = token.split(".", 2)
        expires_at = int(expires)
    except (ValueError, TypeError):
        return False
    if expires_at <= int(time.time()) or expires_at > int(time.time()) + SESSION_MAX_AGE:
        return False
    return hmac.compare_digest(signature, _session_signature(expires, nonce))


class RequestGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        client_key = _client_key(scope)
        content_length = headers.get(b"content-length")

        if content_length is not None:
            try:
                request_size = int(content_length)
            except ValueError:
                await JSONResponse({"detail": "Encabezado Content-Length no válido."}, status_code=400)(scope, receive, send)
                return
            if request_size < 0:
                await JSONResponse({"detail": "Encabezado Content-Length no válido."}, status_code=400)(scope, receive, send)
                return
            max_size = 8192 if path == "/api/auth/login" else MAX_REQUEST_BYTES
            if request_size > max_size:
                await JSONResponse(
                    {"detail": f"La solicitud supera el límite permitido. El tamaño máximo de audio es {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
                    status_code=413,
                )(scope, receive, send)
                return

        public_login_page = path == "/" and method in {"GET", "HEAD"}
        if AUTH_REQUIRED and path not in {"/api/health", "/api/auth/session", "/api/auth/login"} and not public_login_page:
            if not APP_PASSWORD or not APP_SESSION_SECRET:
                await JSONResponse({"detail": "La protección de acceso no está configurada."}, status_code=503)(scope, receive, send)
                return
            cookies = {}
            cookie_header = headers.get(b"cookie", b"").decode("latin-1")
            for item in cookie_header.split(";"):
                name, separator, value = item.strip().partition("=")
                if separator:
                    cookies[name] = value
            if not _valid_session(cookies.get(SESSION_COOKIE_NAME)):
                await JSONResponse({"detail": "Inicia sesión para continuar."}, status_code=401)(scope, receive, send)
                return

        if AUTH_REQUIRED and method == "POST" and (
            path == "/api/jobs" or path.startswith("/api/jobs/") and path.endswith("/retranscribe")
        ):
            if content_length is None:
                await JSONResponse(
                    {"detail": "La solicitud debe incluir Content-Length para aplicar el límite de tamaño."},
                    status_code=411,
                )(scope, receive, send)
                return
            retry_after = job_limiter.hit(client_key)
            if retry_after:
                await JSONResponse(
                    {"detail": "Se alcanzó el límite de 10 trabajos por hora. Inténtalo más tarde."},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )(scope, receive, send)
                return

        if AUTH_REQUIRED and method == "POST" and path == "/api/auth/login":
            if content_length is None:
                await JSONResponse({"detail": "La solicitud de acceso no tiene un tamaño válido."}, status_code=411)(scope, receive, send)
                return
            retry_after = login_limiter.hit(client_key)
            if retry_after:
                await JSONResponse(
                    {"detail": "Demasiados intentos de acceso. Inténtalo más tarde."},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )(scope, receive, send)
                return

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(_app):
    if MAX_UPLOAD_BYTES <= 0:
        raise RuntimeError("MAX_UPLOAD_MB debe ser un entero mayor que cero.")
    if AUTH_REQUIRED:
        if len(APP_PASSWORD) < 16:
            raise RuntimeError("APP_PASSWORD debe configurarse con al menos 16 caracteres.")
        if len(APP_SESSION_SECRET) < 32:
            raise RuntimeError("APP_SESSION_SECRET debe configurarse con al menos 32 caracteres.")
    yield


app = FastAPI(title="Musiconvert", lifespan=lifespan)
app.add_middleware(RequestGuardMiddleware)


@app.get("/api/auth/session", include_in_schema=False)
def auth_session(request: Request) -> dict[str, bool | int]:
    defaults = {
        "required": AUTH_REQUIRED,
        "authenticated": not AUTH_REQUIRED,
        "freeTier": FREE_TIER_MODE,
        "defaultSeparate": DEFAULT_SEPARATE and not FREE_TIER_MODE,
        "defaultChords": DEFAULT_CHORDS and not FREE_TIER_MODE,
        "maxUploadMb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "maxMinutes": MAX_MINUTES,
    }
    if not AUTH_REQUIRED:
        return defaults
    return {
        **defaults,
        "authenticated": _valid_session(request.cookies.get(SESSION_COOKIE_NAME)),
    }


@app.post("/api/auth/login", include_in_schema=False)
def auth_login(request: Request, password: str = Form(...)):
    if not AUTH_REQUIRED:
        return {"authenticated": True}
    if not APP_PASSWORD or not APP_SESSION_SECRET:
        raise HTTPException(503, "La protección de acceso no está configurada.")
    if not hmac.compare_digest(password.encode("utf-8"), APP_PASSWORD.encode("utf-8")):
        raise HTTPException(401, "Contraseña incorrecta.")

    login_limiter.clear(_client_key(request.scope))
    expires = str(int(time.time()) + SESSION_MAX_AGE)
    nonce = secrets.token_urlsafe(18)
    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        SESSION_COOKIE_NAME,
        f"{expires}.{nonce}.{_session_signature(expires, nonce)}",
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/auth/logout", include_in_schema=False)
def auth_logout():
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    return response


@app.get("/api/health", include_in_schema=False)
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# ---------- Modelo Whisper (se carga una sola vez, la primera vez que se usa) ----------
ALLOWED_MODELS = {"base", "small", "medium"}
_models: dict = {}
_model_lock = threading.Lock()


def get_model(size: str):
    with _model_lock:
        if size not in _models:
            _models.clear()  # solo uno a la vez en memoria
            _models[size] = pipeline.load_whisper(size)
        return _models[size]


# ---------- Trabajos en segundo plano ----------
jobs: dict[str, dict] = {}
run_lock = threading.Semaphore(1)  # un trabajo pesado a la vez (CPU)


def run_job(
    job_id: str,
    job_dir: Path,
    audio_path: Path | None,
    url: str,
    separate: bool,
    model_size: str,
    want_chords: bool = True,
) -> None:
    job = jobs[job_id]

    def progress(p: float) -> None:
        job["progress"] = round(max(0.0, min(p, 1.0)), 3)

    t0 = time.time()
    last = {"name": None, "t": t0}

    def stage(name: str) -> None:
        now = time.time()
        if last["name"]:  # imprime en la terminal cuánto tardó la etapa anterior
            print(f"[{job_id[:6]}] {last['name']}: {now - last['t']:.0f} s", flush=True)
        last["name"], last["t"] = name, now
        job["status"] = name
        job["progress"] = 0.0

    try:
        with run_lock:  # mientras espera turno, el estado sigue siendo "queued"
            if url:
                stage("downloading")
                audio_path, title = pipeline.download_youtube(
                    url, job_dir, progress, MAX_MINUTES, MAX_UPLOAD_BYTES
                )
                if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
                    audio_path.unlink()
                    raise RuntimeError(
                        f"El audio descargado supera el límite de "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                    )
                job["title"] = title
                job["audio_ready"] = True  # el navegador ya puede pedir el audio

            source = audio_path
            chords_source = audio_path          # para los acordes: mejor la pista SIN voz si existe
            if separate:
                stage("separating")
                try:
                    vocals = pipeline.separate_vocals(audio_path, job_dir, progress, DEMUCS_MODEL)
                    source = job_dir / ("vocals" + vocals.suffix)  # se conserva para re-transcribir
                    shutil.copy(vocals, source)
                    inst = next(iter(sorted(vocals.parent.glob("no_vocals.*"))), None)
                    if inst is not None:                      # Demucs también genera la pista sin voz
                        chords_source = job_dir / ("instrumental" + inst.suffix)
                        shutil.copy(inst, chords_source)
                except Exception as exc:  # noqa: BLE001  (sigue con el audio original)
                    job["separate_error"] = str(exc) or exc.__class__.__name__
                    print(f"[{job_id[:6]}] Demucs falló, se sigue sin separar: {job['separate_error']}", flush=True)
            job["source"] = str(source)
            job["model_size"] = model_size  # para re-transcribir con el mismo modelo

            stage("loading_model")
            model = get_model(model_size)

            stage("transcribing")
            result = pipeline.transcribe(model, source, LANGUAGE, BEAM_SIZE, progress)

            if want_chords:  # los acordes se detectan sobre el audio original (con instrumentos)
                stage("chords")
                try:
                    found = chord_detect.detect(chords_source)
                    job["chords"], job["key"] = found["chords"], found["key"]
                    job["bpm"], job["beats"] = found["bpm"], found["beats"]
                    job["downbeat"] = found["downbeat"]
                except Exception as exc:  # noqa: BLE001  (si falla, la letra se entrega igual)
                    job["chords_error"] = str(exc) or exc.__class__.__name__

            job["result"] = result
            job["progress"] = 1.0
            job["status"] = "done"
            print(f"[{job_id[:6]}] {last['name']}: {time.time() - last['t']:.0f} s", flush=True)
            print(f"[{job_id[:6]}] TOTAL: {time.time() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(exc) or exc.__class__.__name__
    finally:
        shutil.rmtree(job_dir / "separated", ignore_errors=True)


def run_retranscribe(job_id: str, source: Path, start: float, model_size: str) -> None:
    """Vuelve a transcribir desde `start` hasta el final, sin filtro de silencios."""
    job = jobs[job_id]
    clip = source.parent / f"clip_{job_id}.wav"

    def progress(p: float) -> None:
        job["progress"] = round(max(0.0, min(p, 1.0)), 3)

    try:
        with run_lock:
            job["status"] = "loading_model"
            model = get_model(model_size)
            job["status"] = "transcribing"
            job["progress"] = 0.0
            pipeline.cut_clip(source, start, clip)
            job["result"] = pipeline.transcribe(
                model, clip, LANGUAGE, max(BEAM_SIZE, 5), progress, vad=False, offset=start
            )
            job["progress"] = 1.0
            job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(exc) or exc.__class__.__name__
    finally:
        clip.unlink(missing_ok=True)


# ---------- API ----------
@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = File(None),
    url: str = Form(""),
    separate: bool = Form(True),
    quality: str = Form(""),
    chords: bool = Form(True),
):
    model_size = MODEL_SIZE if FREE_TIER_MODE else quality if quality in ALLOWED_MODELS else MODEL_SIZE
    separate = separate and not FREE_TIER_MODE
    chords = chords and not FREE_TIER_MODE
    url = url.strip()
    has_file = bool(file and file.filename)
    if has_file == bool(url):
        raise HTTPException(400, "Sube un archivo o pega un link de YouTube (solo uno de los dos).")

    job_id = uuid.uuid4().hex
    job_dir = UPLOADS_DIR / job_id
    audio_path: Path | None = None

    if has_file:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Formato no soportado: {ext or 'desconocido'}")
        job_dir.mkdir(parents=True)
        audio_path = job_dir / f"original{ext}"
        upload_complete = False
        uploaded_bytes = 0
        try:
            with open(audio_path, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413,
                            f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                        )
                    out.write(chunk)
            upload_complete = True
        finally:
            await file.close()
            if not upload_complete:
                shutil.rmtree(job_dir, ignore_errors=True)
        if FREE_TIER_MODE:
            audio_valid = False
            try:
                duration = pipeline.audio_duration(audio_path)
                if duration > MAX_MINUTES * 60:
                    raise HTTPException(
                        400,
                        f"En el plan gratuito, el audio debe durar {MAX_MINUTES} minutos o menos.",
                    )
                audio_valid = True
            finally:
                if not audio_valid:
                    shutil.rmtree(job_dir, ignore_errors=True)
    else:
        if not pipeline.is_youtube_url(url):
            raise HTTPException(400, "Solo se aceptan links de YouTube (youtube.com o youtu.be).")
        job_dir.mkdir(parents=True)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "title": None,
        "audio_ready": False,
        "chords": None,
        "key": None,
        "bpm": 0,
        "beats": [],
        "downbeat": 0,
        "chords_error": None,
    }
    threading.Thread(
        target=run_job, args=(job_id, job_dir, audio_path, url, separate, model_size, chords),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/retranscribe")
def retranscribe(job_id: str, start: float = Form(...)):
    parent = jobs.get(job_id)
    if not parent or parent.get("status") != "done" or not parent.get("source"):
        raise HTTPException(404, "Esa canción ya no está disponible. Genera la letra de nuevo.")
    source = Path(parent["source"])
    if not source.exists():
        raise HTTPException(404, "El audio de esa canción ya no está en el servidor.")

    new_id = uuid.uuid4().hex
    jobs[new_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None,
                    "title": None, "audio_ready": False}
    model_size = parent.get("model_size") or MODEL_SIZE
    threading.Thread(
        target=run_retranscribe,
        args=(new_id, source, max(start, 0.0), model_size),
        daemon=True,
    ).start()
    return {"job_id": new_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Trabajo no encontrado")
    return job


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str):
    """Audio descargado de YouTube, para reproducirlo en el navegador."""
    job = jobs.get(job_id)
    if not job or not job.get("audio_ready"):
        raise HTTPException(404, "Audio no disponible")
    found = sorted((UPLOADS_DIR / job_id).glob("original.*"))
    if not found:
        raise HTTPException(404, "Audio no disponible")
    return FileResponse(found[0])


# El frontend se sirve en la raíz (debe ir al final)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
