# Musiconvert — Karaoke con letras sincronizadas

Sube un archivo de audio → Whisper genera la letra con tiempos por palabra → se muestra tipo karaoke en el navegador.

## Requisitos

- Python 3.10 o superior
- `ffmpeg` instalado y en el PATH
  - Windows: `winget install ffmpeg` (o `choco install ffmpeg`)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Actualizar desde la Fase 1 (entorno ya creado)

```
venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps demucs==4.1.0
winget install DenoLand.Deno
```
Reinicia VS Code después de instalar Deno. Si `pip` se queja de conflictos, avísame con el mensaje completo.

## Instalación

```bash
cd karaoke-app/backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install --no-deps demucs==4.1.0
```

## Ejecutar

```bash
uvicorn main:app --reload
```

Abre http://127.0.0.1:8000 en el navegador, elige una canción y espera.

> La primera vez se descarga el modelo de Whisper (~500 MB para `small`). Después queda en caché.

## Modelo y velocidad (sin GPU)

Cambia el modelo con una variable de entorno antes de arrancar:

```bash
# Windows (PowerShell)
$env:WHISPER_MODEL="base"; uvicorn main:app

# macOS / Linux
WHISPER_MODEL=base uvicorn main:app
```

| Modelo   | Velocidad en CPU          | Precisión en español |
|----------|---------------------------|----------------------|
| `tiny`   | muy rápido                | baja                 |
| `base`   | rápido                    | media                |
| `small`  | **recomendado** (equilibrio) | buena             |
| `medium` | lento (varios minutos)    | muy buena            |

Otras variables: `WHISPER_LANG` (por defecto `es`; vacío = autodetectar) y `WHISPER_BEAM` (por defecto `3`).

## Estructura

```
karaoke-app/
├── Dockerfile             # Imagen Linux para desplegar
├── render.yaml            # Blueprint de Render
├── backend/
│   ├── main.py            # API FastAPI, trabajos y acceso
│   ├── pipeline.py        # Descarga, separación y transcripción
│   ├── chords.py          # Detección de acordes
│   ├── rhythm.py          # Detección y edición de tempo
│   └── requirements.txt
└── frontend/
    └── index.html         # Reproductor karaoke
```

## Cómo funciona

1. El navegador sube el audio a `POST /api/jobs` y recibe un `job_id`.
2. El servidor transcribe en un hilo aparte (`faster-whisper`, CPU, `int8`, `word_timestamps=True`, filtro VAD).
3. El navegador consulta `GET /api/jobs/{id}` cada 1.5 s para mostrar el progreso.
4. Con el resultado, la página resalta línea y palabra según `audio.currentTime`. Clic en una línea = saltar a ese punto.

## Problemas comunes

- **`ffmpeg not found`**: instálalo y reinicia la terminal.
- **Letra con errores o texto inventado**: normal en música con mucho instrumental. La Fase 2 (separar la voz con Demucs) lo mejora bastante.
- **Muy lento**: usa `WHISPER_MODEL=base`.

## Desplegar en Render

El proyecto incluye un `render.yaml` y un `Dockerfile`. Sube el código a un repositorio Git y conéctalo a Render; desde **New > Blueprint**, selecciona ese repositorio y revisa el plan antes de aplicarlo. La configuración actual usa el plan gratuito de Render para una demostración limitada, no para producción.

El plan gratuito dispone de 512 MB RAM y CPU compartida; Render suspende la instancia tras 15 minutos sin tráfico y su sistema de archivos es efímero. Por eso el perfil gratuito usa Whisper `tiny`, admite audios de hasta 3 minutos y 20 MB, y desactiva Demucs y los acordes. Incluso con estas reducciones no se garantiza que una transcripción termine: puede ser lenta, quedarse sin memoria o perderse si la instancia se reinicia. Render debe ejecutar una sola instancia y un solo worker; los trabajos y su estado se guardan en memoria. No uses esta configuración para producción, privacidad de datos sensibles ni conservar canciones/resultados.

Antes de desplegar, define `APP_PASSWORD` en Render con una contraseña aleatoria de al menos 16 caracteres, usando un gestor de contraseñas. Render genera `APP_SESSION_SECRET` automáticamente. La aplicación protege sus API con una cookie firmada, `HttpOnly`, `SameSite=Strict` y `Secure`; la sesión vence a las 12 horas. La página de acceso sigue siendo visible para permitir iniciar sesión, pero las operaciones y los resultados requieren una sesión válida. No compartas la contraseña en el repositorio ni en el chat.

Se limitan los intentos de acceso a 5 por 15 minutos y los trabajos a 10 por hora por dirección IP. Los audios subidos o descargados tienen un máximo de 90 MB; las solicitudes grandes se rechazan antes del procesamiento y los archivos se validan durante su escritura. Estos contadores viven en memoria: son una protección básica para una sola instancia, no un sistema distribuido contra ataques. La ruta `/api/health` comprueba que el servidor responde; no indica que los modelos estén cargados.

Para publicar, el proyecto debe estar en un repositorio Git conectado a Render. Al crear el Blueprint, introduce `APP_PASSWORD` en la configuración secreta solicitada y revisa el plan/costo antes de confirmar. No se incluye ninguna credencial ni se crea el servicio automáticamente. Si luego habilitas varias instancias o workers, mueve sesiones, límites y trabajos a servicios compartidos antes de escalar.

## Etapas de mejora del sitio

1. Mejoras de precisión y datos reales
2. Experiencia inicial
3. Móvil y accesibilidad
4. Progreso, errores y recuperación
5. Pulido y preparación para despliegue (completada)
6. Despliegue de demostración gratuito en Render (en curso)

### Lista segura antes de publicar el repositorio

1. Instala Git y crea un repositorio **privado** para el proyecto.
2. Comprueba que `.gitignore` excluye los entornos virtuales, `backend/uploads/` y archivos `.env`; nunca subas contraseñas, tokens ni audio de usuarios.
3. Sube el código desde tu equipo y conecta ese repositorio a Render usando **New > Blueprint**.
4. Antes de aplicar el Blueprint, confirma que el plan sea `free` y que el costo sea `$0`.
5. Introduce `APP_PASSWORD` directamente en la configuración secreta de Render (mínimo 16 caracteres). No la pongas en el repositorio, un archivo compartido ni el chat. Render genera `APP_SESSION_SECRET`.
6. Tras el primer despliegue, verifica `/api/health`, la pantalla de acceso, un inicio/cierre de sesión y un audio corto. El estado `/api/health` solo indica que la API responde; no confirma que los modelos ya estén cargados.

**El despliegue queda pendiente** hasta que el propietario cree y conecte el repositorio y confirme el costo del servicio. No hay Git, Docker ni credenciales de Render configurados en este entorno, por lo que no se ha creado ni publicado ningún servicio.

## Fase 2: YouTube y separación de voz

- Pega un link de YouTube (un solo video, máx. 12 min) y pulsa **Generar**.
- La casilla **Separar voz** aísla la voz con Demucs antes de transcribir: tarda varios minutos más en CPU, pero omite y confunde menos palabras. Desmárcala para ir rápido.
- Variables nuevas: `DEMUCS_MODEL` (por defecto `htdemucs`) y `MAX_MINUTES`.
- YouTube necesita Deno instalado y `yt-dlp` actualizado (`pip install -U "yt-dlp[default]"`). Para instalación manual, usa `pip install --no-deps demucs==4.1.0`.
