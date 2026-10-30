@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv-4\Scripts\python.exe"

if not exist "%PYTHON%" set "PYTHON=%ROOT%backend\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo No se encontro el entorno Python de Musiconvert.
    echo Sigue la seccion "Windows local (CPU)" del README.md.
    pause
    exit /b 1
)

"%PYTHON%" -c "import fastapi, faster_whisper, numpy, demucs" >nul 2>&1
if errorlevel 1 (
    echo Faltan dependencias de Musiconvert en este entorno.
    echo Sigue la seccion "Windows local (CPU)" del README.md.
    pause
    exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo No se encontro FFmpeg en PATH. Instala FFmpeg y vuelve a intentarlo.
    pause
    exit /b 1
)

set "APP_AUTH_REQUIRED=false"
set "FREE_TIER_MODE=false"
set "WHISPER_MODEL=small"
set "DEFAULT_SEPARATE=true"
set "DEFAULT_CHORDS=true"
set "MAX_MINUTES=12"
set "MAX_UPLOAD_MB=90"
set "SESSION_COOKIE_SECURE=false"

echo Abre http://127.0.0.1:8000 cuando Uvicorn indique que el servidor esta activo.
pushd "%ROOT%backend"
"%PYTHON%" -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo Musiconvert se cerro con un error.
    pause
)
exit /b %EXIT_CODE%
