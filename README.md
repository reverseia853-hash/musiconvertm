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

### Windows local (CPU)

Para esta versión de PyTorch usa Python 3.12. Desde la carpeta raíz del proyecto:

```powershell
py -3.12 -m venv .venv-4
.\.venv-4\Scripts\python.exe -m pip install --upgrade pip
.\.venv-4\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.5.1+cpu torchaudio==2.5.1+cpu
.\.venv-4\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv-4\Scripts\python.exe -m pip install --no-deps demucs==4.1.0
```

Con FFmpeg instalado, abre `Iniciar-Musiconvert.bat`. El servidor solo escucha en `127.0.0.1`; los archivos procesados localmente no se envían a Render. El modelo `small` es el predeterminado; en **Opciones > Calidad** puedes elegir `medium` para priorizar la letra, a costa de más tiempo y memoria. La primera transcripción descargará el modelo elegido y lo conservará en la caché de Hugging Face.

## Ejecutar

```bash
cd karaoke-app/backend
uvicorn main:app --reload
```

Abre http://127.0.0.1:8000 en el navegador, elige una canción y espera.

> La primera vez se descarga el modelo de Whisper (~500 MB para `small`). Después queda en caché.

El BPM se estima con seguimiento de pulsos de librosa. Algunas canciones admiten una lectura a mitad o al doble del pulso; los botones `÷2` y `×2` muestran el valor alternativo y permiten compararlo con el ritmo oído. El BPM sirve como referencia temporal, pero los acordes pueden cambiar entre pulsos o durar varios.

En **Opciones > Modo de letra** puedes elegir por canción: **Actual (con VAD)** usa el filtro sensible al canto (`threshold=0.15`); **Sin VAD** analiza sin ese filtro, con un tamaño de búsqueda mínimo de 5; **Comparar ambas** ejecuta las dos pasadas sobre la misma fuente de audio y deja elegir cuál conservar. El modo de comparación tarda más porque realiza dos transcripciones y no combina automáticamente las letras: la versión con VAD queda activa mientras revisas ambas. Todos los modos mantienen `no_speech_threshold=0.6` para limitar texto inventado sobre la música. La separación de voz es otra opción independiente y se aplica antes del modo elegido cuando está activada. Sin VAD pueden aparecer palabras incorrectas sobre pausas o partes instrumentales; comparar más resultados no garantiza que una versión sea correcta.

La opción **Nivel de aislamiento** ajusta las pasadas de Demucs para intentar separar mejor una voz mezclada con instrumentos: **Poco** mantiene los valores rápidos actuales (1 shift, overlap 0.1), **Medio** usa 2 shifts y overlap 0.25, y **Mucho** usa 4 shifts y overlap 0.5. Los niveles superiores pueden tardar considerablemente más y no garantizan una transcripción más precisa. Si se desactiva **Separar voz**, el nivel de aislamiento no se aplica.

En el modo actual, si una canción de al menos un minuto produce menos de 8 palabras por minuto o deja sin letra un tramo final de más de 12 segundos o el 8% de la duración (lo que sea mayor), la página ofrece iniciar manualmente la segunda transcripción sin VAD. La primera versión se conserva para poder comparar y restaurar.

No se aplican filtros predefinidos por género (pop, cristiano, country): el género no es un filtro de voz, y orientar el prompt hacia un género sin conocer la letra puede sesgar palabras en vez de recuperarlas. Para priorizar la letra se puede elegir el modelo `medium` en **Opciones > Calidad**, a costa de mayor tiempo y memoria.

La exportación a imagen y PDF usa siempre una sola hoja A4 vertical. En los tramos instrumentales muestra únicamente una línea con los acordes, sin rótulos ni tiempos; no añade filas cuando no hay acordes. Las secciones con letra separadas por al menos 2.5 segundos se marcan como “Sección”. También busca bloques de dos o más líneas de letra que se repitan exactamente tras normalizar mayúsculas, tildes y puntuación: los bloques de al menos ocho palabras se marcan como “Coro”; los de seis o siete palabras solo si además coincide su secuencia de acordes. Cada aparición se conserva y se etiqueta, sin eliminar ni alterar letra o tiempos. La separación temporal por sí sola no se presenta como detección segura de estrofa, y la repetición es una señal práctica, no una clasificación musical infalible. La copia de la hoja de acordes aplica las mismas etiquetas. El tamaño del texto se adapta al contenido y al espacio disponible; las canciones muy largas pueden producir texto más pequeño.

Para los acordes, los cambios se evalúan sobre la cuadrícula de pulsos cuando el tempo es confiable. El detector no supone que solo pueden cambiar al inicio o a la mitad del compás y conserva los cambios de un solo pulso; sin una cuadrícula confiable, solo suaviza fluctuaciones muy breves. Reconoce acordes mayores, menores y séptimas dominante (`7`), mayor (`maj7`) y menor (`m7`). Solo muestra una séptima cuando su evidencia supera claramente la de la tríada correspondiente; si no, conserva la mayor o menor. Los acordes con poca evidencia se marcan como dudosos para revisarlos. La detección sigue siendo orientativa: puede equivocarse en canciones comerciales y requiere escuchar y corregir los resultados cuando sea necesario.

Los acordes detectados en intervalos sin letra también se muestran en filas instrumentales al inicio, entre frases y al final de la canción. La vista usa los mismos acordes y tiempos que el análisis; no realiza un procesamiento adicional.

En **Acordes > Cifrado** puedes alternar entre nombres de acordes internacionales (`C`, `Dm`, `G7`) y nombres latinos (`Do`, `Rem`, `Sol7`). La opción solo cambia la presentación: el análisis, los acordes almacenados, la transposición y las correcciones conservan el cifrado interno original. El cambio se refleja en la vista, el editor de acordes y las exportaciones. La herramienta para medir acierto sigue esperando acordes reales escritos con nombres internacionales (`C D E F G A B`).

Pruebas locales del análisis: desde la carpeta `backend`, ejecuta `..\.venv-4\Scripts\python.exe -m unittest discover -s tests -v`. Incluyen tempos sintéticos, acordes mayores/menores y séptimas en las 12 tonalidades, una comprobación para evitar convertir una tríada con una séptima débil, progresiones repetidas y cambios de acorde en pulsos consecutivos a 145 BPM; son comprobaciones de funcionamiento, no una medición de precisión en canciones comerciales.

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
- **Letra con errores o texto inventado**: puede ocurrir en música por la mezcla, los efectos o las voces de fondo. Demucs puede ayudar en algunas canciones, pero la separación no siempre mejora la transcripción; compara con y sin esa opción.
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
6. Despliegue de demostración gratuito en Render (completada)

### Estado del despliegue

La demo gratuita está publicada en Render y su ruta `/api/health` respondió correctamente durante la comprobación inicial. El perfil de Render es limitado y está pensado para demostración; para uso personal, se recomienda iniciar la versión local con `Iniciar-Musiconvert.bat`. Los cambios locales de este proyecto no se publican automáticamente en Render.

### Instalar Musiconvert en Android

Musiconvert puede instalarse como aplicación web progresiva (PWA), no como APK. Publica los cambios en el servicio HTTPS y abre su dirección en Chrome para Android; desde el menú del navegador elige **Instalar aplicación** o **Añadir a pantalla de inicio**. En navegadores compatibles también aparecerá el botón **Instalar app**. La PWA consulta la versión publicada al abrirse y recibe las actualizaciones al desplegar nuevos cambios. Necesita conexión a Internet y no guarda canciones, sesiones ni respuestas en caché; no funciona sin conexión.

La instalación gratuita usa los mismos límites de la demo de Render: el procesamiento puede ser lento o interrumpirse, y el perfil gratuito limita la duración y el tamaño de audio, además de desactivar separación de voz y detección de acordes. Para compartir la demo, todos usan la misma contraseña configurada en Render; no hay cuentas separadas. La aplicación no se publica ni se actualiza en el teléfono hasta que los cambios se desplieguen al servicio. No subas canciones privadas o sensibles a la demo pública.

## Fase 2: YouTube y separación de voz

- Pega un link de YouTube (un solo video, máx. 12 min) y pulsa **Generar**.
- La casilla **Separar voz** usa Demucs antes de transcribir: puede ayudar en algunas canciones, pero tarda más en CPU y el resultado depende de la mezcla. Compara con y sin separación si la letra no sale bien.
- Variables nuevas: `DEMUCS_MODEL` (por defecto `htdemucs`) y `MAX_MINUTES`.
- YouTube necesita Deno instalado y `yt-dlp` actualizado (`pip install -U "yt-dlp[default]"`). Para instalación manual, usa `pip install --no-deps demucs==4.1.0`.
