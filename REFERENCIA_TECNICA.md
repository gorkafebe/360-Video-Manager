# REFERENCIA TÉCNICA — 360-Video-Manager

> Documento de referencia técnica generado por análisis exhaustivo y literal del
> código fuente (no de README/docstrings sin verificar). Cada afirmación
> relevante cita fichero y, cuando es posible, línea o función exacta. Estado
> analizado: `git log` con HEAD en `ef66e8a` (rama `main`), working tree limpio.
> Entorno de verificación: venv en `/home/gorferna/360-Video-Manager` (Python
> 3.12.3, `opencv-python` 4.13.0.92 — ver discrepancia en §9.1).
>
> Convención de citas: `fichero.py:N` apunta a la línea N tal como existía en
> el momento de este análisis. Los bloques marcados con **⚠ DISCREPANCIA**
> señalan contradicciones entre documentación existente (README.md, AGENTS.md,
> docstrings, SCAN_REPORT.md, IMPROVEMENT_PLAN.md) y el comportamiento real del
> código.

---

## Índice

1. [Mapa de módulos y responsabilidades](#1-mapa-de-módulos-y-responsabilidades)
2. [Dependencias externas](#2-dependencias-externas)
3. [Flujo de datos end-to-end (workflows/unified_pipeline.py)](#3-flujo-de-datos-end-to-end)
4. [Pipeline de detección de proyección — visión general y orden de fases](#4-pipeline-de-detección-de-proyección)
5. [Fase A — Extracción de frames (detector/video_io.py)](#5-fase-a--extracción-de-frames)
6. [Fase B — Detección de línea horizontal/vertical (detector/line_detection.py)](#6-fase-b--detección-de-línea)
7. [Fase C — Detección estéreo (detector/stereo_detection.py)](#7-fase-c--detección-estéreo)
8. [Fase D — Análisis de movimiento / flujo óptico (detector/motion_analysis.py)](#8-fase-d--análisis-de-movimiento--flujo-óptico)
9. [Fase E — Scoring EAC vs Cúbica (detector/projection_logic.py)](#9-fase-e--scoring-eac-vs-cúbica)
10. [Fase F — Evidencia equirectangular positiva (detector/equirectangular_detection.py)](#10-fase-f--evidencia-equirectangular-positiva)
11. [Orquestación, política de fiabilidad y reintentos (detector/pipeline.py)](#11-orquestación-política-de-fiabilidad-y-reintentos)
12. [Dominio de salida completo y degradación a `unknown`](#12-dominio-de-salida-completo-y-degradación-a-unknown)
13. [Tabla maestra de constantes, umbrales y variables de entorno](#13-tabla-maestra-de-constantes-umbrales-y-variables-de-entorno)
14. [Conversión a equirectangular con ffmpeg (detector/projection_conversion.py)](#14-conversión-a-equirectangular-con-ffmpeg)
15. [Módulos de soporte: core/, config/, utils/, app/](#15-módulos-de-soporte)
16. [Tests](#16-tests)
17. [Known issues, caveats y suposiciones](#17-known-issues-caveats-y-suposiciones)
18. [Discrepancias documentación vs código](#18-discrepancias-documentación-vs-código)
19. [AFIRMACIONES QUE PODRÍAN SER IMPRECISAS EN UNA PRESENTACIÓN](#19-afirmaciones-que-podrían-ser-imprecisas-en-una-presentación)

---

## 1. Mapa de módulos y responsabilidades

| Módulo | Responsabilidad principal | Dependencias externas que invoca |
|---|---|---|
| `app/main.py`, `app/__main__.py` | Punto de entrada; delega a la GUI | — |
| `app/gui/gui_app.py` | GUI CustomTkinter; orquesta hilos de fondo; **no** contiene lógica de pipeline | `customtkinter`, `PIL`, `urllib.request` (descarga miniaturas) |
| `app/gui/progress_utils.py` | Parseo de progreso de yt-dlp y *throttling* de actualizaciones de UI | — |
| `config/settings.py` | Única fuente de configuración (env / `.env`); singleton `get_settings()` | `python-dotenv` (opcional, con fallback manual) |
| `config/logging_config.py` | Handlers de logging (consola coloreada + fichero) | — |
| `core/downloader.py` | Descarga de vídeo | `yt_dlp.YoutubeDL` |
| `core/youtube.py` | Búsqueda y miniaturas de YouTube | `googleapiclient.discovery.build` (YouTube Data API v3) |
| `core/uploader.py` | Cliente HTTP de subida a MediaCMS | `requests` |
| `core/preview_frames.py` | Extracción de frames de previsualización (independiente del detector) | `cv2.VideoCapture` |
| `core/models.py` | Dataclasses `JobResult`, `DetectorStats`, `UploadResult` | — |
| `core/job_manifest.py` | Persistencia JSON de cada job en `data/jobs/` | — |
| `detector/pipeline.py` | Orquestador de detección; gating de fiabilidad; reintentos | `cv2`, `numpy` |
| `detector/video_io.py` | Extracción de frames (OpenCV + fallback ffmpeg); `ffprobe`; transcodificación de compatibilidad | `cv2.VideoCapture`, `subprocess` → `ffmpeg`/`ffprobe` |
| `detector/line_detection.py` | Detección de costura horizontal/vertical | `cv2` (Hough, LSD, Canny, Sobel, FFT vía `numpy.fft`) |
| `detector/stereo_detection.py` | Detección de `stereo_equi` por similitud de histogramas/bordes | `cv2` (histogramas, Canny) |
| `detector/motion_analysis.py` | Backends de flujo óptico, consistencia FB, evidencia geométrica ORB | `cv2` (Farneback, DIS, `cv2.optflow.*`, ORB, `findHomography`) |
| `detector/projection_logic.py` | Scoring EAC vs cubemap por simetría de ángulos de cara | `numpy` |
| `detector/equirectangular_detection.py` | Evidencia positiva de wrap-around equirectangular | `cv2`, `numpy` |
| `detector/projection_conversion.py` | Conversión ffmpeg `v360` a equirectangular; fallbacks de encoder/audio | `subprocess` → `ffmpeg` |
| `detector/preprocessing.py` | Escala de grises + blur gaussiano | `cv2` |
| `detector/region_validation.py` | Filtro de validez de región por concentración/actividad | — |
| `detector/debug_utils.py` | I/O de imágenes/logs de depuración | `cv2` |
| `workflows/unified_pipeline.py` | Orquestación de las 8 etapas (descarga→...→manifiesto) | — |
| `utils/exceptions.py` | Jerarquía de excepciones | — |
| `utils/paths.py` | Helpers de rutas delegando en `get_settings()` | — |
| `scripts/diagnose_line_detection.py` | Herramienta CLI de diagnóstico standalone para `line_detection.py` | `cv2` |

No se han detectado imports circulares entre estos módulos (confirmado por lectura directa de los `import` de cada fichero).

---

## 2. Dependencias externas

Declaradas en `requirements.txt` y `pyproject.toml`:

| Paquete | Rango | Uso real verificado en código |
|---|---|---|
| `google-api-python-client` | `>=2.196.0,<3` | `core/youtube.py` — `googleapiclient.discovery.build("youtube","v3",...)` |
| `yt-dlp` | `>=2025.1.0,<2027.0.0` | `core/downloader.py` — `YoutubeDL` |
| `requests` | `>=2.32.0,<3` | `core/uploader.py` |
| `python-dotenv` | `>=1.0.1,<2` | `config/settings.py::_load_dotenv` (con parser manual de respaldo `_parse_dotenv` si falta) |
| `pillow` | `>=10.4.0,<12` | `app/gui/gui_app.py` — miniaturas (`PIL.Image`) |
| `opencv-contrib-python` | `>=4.13.0.92,<4.14` | Todo `detector/*` y `core/preview_frames.py`. **Ver discrepancia §18.1**: el venv de este repositorio en disco tiene instalado `opencv-python` (sin *contrib*), no `opencv-contrib-python`. |
| `numpy` | `>=1.26,<3` | Todo `detector/*` |
| `customtkinter` | `>=5.2,<6` | `app/gui/gui_app.py` |

Dependencias de sistema invocadas vía `subprocess` (no declaradas como paquetes Python):

- **ffmpeg** — usado en `detector/video_io.py` (extracción de frames de respaldo, transcodificación de compatibilidad) y `detector/projection_conversion.py` (conversión `v360`).
- **ffprobe** — usado en `detector/video_io.py::_run_ffprobe_json` para metadatos de stream (codec, fps, dimensiones, nº de frames).

---

## 3. Flujo de datos end-to-end

`workflows/unified_pipeline.py::process_video_job(options: JobOptions) -> JobResult` (líneas 345–561) ejecuta, en este orden literal:

| Etapa | Función interna | Condición de ejecución |
|---|---|---|
| 1. Resolución de URL/búsqueda | `_stage_search_and_resolve` (línea 126) | Solo si `local_video_path` no está definido |
| 2. Descarga | `_stage_download` (línea 165) → `core.downloader.download_video` | Idem |
| 3. Normalización de codec | `_stage_normalize_codec` (línea 182) → `detector.video_io.convert_video_codec` | **Solo si** `force_full_codec_normalization` es `True` (línea 421: `if force_full_codec_normalization: ... else: logger.info("Full-file normalization disabled...")`). Por defecto (`JobOptions.force_full_codec_normalization=False`, línea 119, y `Settings.force_full_codec_normalization=False`, `config/settings.py:200`) **esta etapa se omite** y se usa directamente el vídeo descargado. |
| 4. Frames de previsualización (UI) | `_stage_preview_frames` (línea 224) → `core.preview_frames.extract_preview_frames` | Siempre |
| 5. Detección de proyección | `_stage_detect_projection` (línea 239) → `detector.pipeline.run_detection_with_retries` | Siempre; con un reintento automático de normalización de codec si la extracción muestreada falla (`frame_extraction_timeout` / `frame_extraction_decode_failure`, líneas 455–486) |
| 6. Conversión a equirectangular | `_stage_convert_to_equirectangular` (línea 264) | Solo si `options.convert_if_needed=True` (default `True`, línea 108) |
| 7. Subida a MediaCMS | `_stage_upload` (línea 315) | Solo si `options.upload=True` (default `False`, línea 107) |
| 8. Persistencia de manifiesto JSON | `core.job_manifest.save_job_manifest` | Solo si `options.save_manifest=True` (default `True`, línea 117); se ejecuta en un bloque `finally` (línea 554) independientemente del éxito/fracaso de las etapas anteriores |

**⚠ Nota de precisión**: el docstring del módulo (`unified_pipeline.py:1-31`) numera "3. Codec normalisation" como un paso fijo de la secuencia. En la implementación real es condicional y está deshabilitado por defecto; el comentario en línea 424-426 lo confirma: *"Full-file normalization disabled; detection will use sampled extraction fallback when needed."* El README (`README.md:188-189`) sí refleja esto correctamente ("no full-file transcode by default").

`JobResult.final_video_path` (propiedad, `core/models.py:138-148`) resuelve la prioridad: `converted_video_path` > `normalized_video_path` > `original_video_path`. Esto determina qué fichero se sube en la etapa 7 (`gui_app.py:782-786` usa la misma prioridad para habilitar el botón "Subir").

---

## 4. Pipeline de detección de proyección

Punto de entrada de producción: `workflows/unified_pipeline.py::_stage_detect_projection` → `detector/pipeline.py::run_detection_with_retries` (líneas 1571–1726), que internamente llama a `run_detection_pipeline` (líneas 873–1543) una o más veces.

### 4.1 Orden de fases dentro de una sola pasada de `run_detection_pipeline`

1. **Extracción de frames principales** (`extract_main_frames`) y **secuencias secundarias** (`extract_secondary_frames`) — detallado en §5.
2. **Filtro de frame negro** por cada frame principal: `frame_esta_en_negro` (línea 175; umbral de intensidad 16, proporción mínima oscura 0.98). Los frames negros se descartan y no pasan a las fases siguientes.
3. **Evidencia equirectangular por frame** (no descarta nada, solo acumula evidencia): `compute_frame_equirectangular_evidence` (línea 1059) — ver §10.
4. **Detección de línea horizontal** por frame (`detect_horizontal_line`, línea 1061). Si no hay línea horizontal, se intenta **detección de línea vertical** como candidato a estéreo lado-a-lado (`detect_vertical_line`, línea 1112).
5. **Decisión temprana / bifurcación principal** según cuántos frames tienen línea (línea 1185 en adelante):
   - Si **ningún** frame tiene línea horizontal → rama de clasificación temprana (equirectangular o estéreo lado-a-lado), ver §4.2.
   - Si **algunos** frames tienen línea horizontal → comprobación de estéreo arriba/abajo (`detect_stereo`, línea 1278) y, si no es estéreo, **análisis de movimiento** (`_classify_non_equirectangular`, línea 1352) para decidir EAC vs cúbica.
6. **Comprobación de suficiencia de datos** (`evaluar_suficiencia_datos`, línea 1417): si el ratio de descarte es demasiado alto o hay muy pocos frames analizados, fuerza `unknown` salvo que la clasificación ya fuera `equirectangular` (línea 1425).
7. **Gate final de fiabilidad**: si la clasificación es `eac` o `cubic` pero `motion_reliable` es `False`, se degrada a `unknown` (línea 1430-1431).
8. **Regla de rescate B** (línea 1433-1445): si el resultado es `unknown` (o no fiable) y la evidencia equirectangular positiva es fuerte y el test de estéreo fue negativo, se reclasifica como `equirectangular`.

### 4.2 Bifurcación cuando no hay línea horizontal (línea 1187 en adelante)

```
frames_with_line == 0
 ├─ ¿hay candidatos de línea vertical (indices_con_linea_vertical)?
 │   ├─ Sí → detect_stereo(..., arrangement="left-right")
 │   │        ├─ es estéreo  → projection_type = "stereo_equi"
 │   │        └─ no es estéreo → projection_type = "equirectangular"
 │   └─ No → projection_type = "equirectangular"
 └─ (en todos los casos de esta rama, motion_reliable = True;
    no se ejecuta análisis de movimiento)
```

Esta rama nunca llega a `_classify_non_equirectangular`: la ausencia total de costura central se interpreta directamente como evidencia de equirectangular mono o estéreo lado-a-lado, nunca como EAC/cúbica.

### 4.3 Bifurcación cuando sí hay línea horizontal (línea 1252 en adelante)

```
frames_with_line > 0
 ├─ len(indices_con_linea) < stereo_min_seam_frames (default 2)
 │     → se omite detect_stereo (resultado simulado is_stereo=False)
 ├─ si no, detect_stereo(..., arrangement="up-down")
 │     ├─ es estéreo → projection_type = "stereo_equi"
 │     └─ no es estéreo:
 │          ├─ frames_with_line >= min_frames_with_line_required
 │          │   OR (ratio frames_with_line/frames_analyzed >= 0.50 AND frames_with_line >= 2)
 │          │        → ejecuta _classify_non_equirectangular → "eac" | "cubic" | "unknown"
 │          └─ si no se cumple ninguna condición → motion_reliable=False,
 │               reason="insufficient_structural_frames:..."
```

La condición de "ratio gate" (línea 1323-1330) permite proceder al análisis de movimiento incluso si el conteo absoluto de frames con línea es menor que `min_frames_with_line_required`, siempre que al menos el 50 % de los frames analizados tengan línea y haya un mínimo de 2.

---

## 5. Fase A — Extracción de frames

Fichero: `detector/video_io.py`.

### 5.1 Frames principales — `extract_main_frames(...)` (líneas 544–781)

- Modo por defecto `"equiespaciados"`: calcula posiciones equiespaciadas entre `start_frame` y `end_frame`, dejando un *padding* de `padding_segundos=3.0` s (o el 10 % del total de frames, lo que sea menor) en cada extremo (línea 592-595). Posiciones exactas: `_compute_even_positions` (línea 227-233).
- Si `num_frames` solicitado excede el rango utilizable, se reduce automáticamente (línea 596-597).
- **Cadena de fallback de lectura**, por cada posición de frame, en este orden exacto:
  1. `cv2.VideoCapture.read()` vía `_CachedCapture` (si OpenCV puede decodificar el contenedor).
  2. Si falla: extracción por lotes con un único proceso `ffmpeg` y filtro `select=` (`_extract_batch_frames_ffmpeg`, línea 254).
     - Si el número de timestamps solicitados supera `MAX_SINGLE_PASS_FRAMES = 10` (línea 365), se delega en `_extract_batch_frames_ffmpeg_parallel` (línea 381): un proceso `ffmpeg` por timestamp con *fast-seek* (`-ss` antes de `-i`), hasta `max_workers=4` hilos (línea 385). Si el modo paralelo lanza una excepción, se reintenta con el modo de paso único `select=` (línea 294-295).
  3. Si algún frame sigue faltando: `_extract_single_frame_ffmpeg_with_diagnostics` por frame individual (línea 168), con timeout de 25 s por defecto.
  4. Si ningún plan de extracción produce frames, se reintenta con un **segundo plan de posiciones reducido** (la mitad de frames, mínimo 2) antes de fallar (línea 614-619).
- Si todos los intentos fallan: `FrameExtractorError` con `code="frame_extraction_timeout"` o `"frame_extraction_decode_failure"` (línea 709-721), incluyendo diagnóstico detallado por intento.
- Codecs AV1 (`{"av1","libaom-av1","libdav1d"}`, línea 16) son detectados explícitamente vía `probe_video_stream` cuando OpenCV no puede decodificar, y se registra un `logger.warning` específico (línea 573-578) — sin cambiar el comportamiento de extracción en sí (sigue yendo por la cadena de fallback ffmpeg general).

### 5.2 Frames secundarios — `extract_secondary_frames(...)` (líneas 783–855)

Para cada posición principal `pos`, construye una ventana de **hasta 3 posiciones únicas**: `{max(0, pos-paso_frames), pos, min(total_frames-1, pos+paso_frames)}` (línea 804-809; `paso_frames` por defecto 5, pero la cadena de reintentos usa 5 → 9 → 13, ver §11.3). El número real de frames por secuencia puede ser **menor de 3** si el recorte a los límites del vídeo produce posiciones duplicadas (p. ej. al principio o final del vídeo). Cada secuencia secundaria es la entrada para el análisis de movimiento entre pares de frames (§8).

### 5.3 Normalización de codec — `convert_video_codec(...)` (líneas 453–541)

Solo se invoca si `_can_decode_with_opencv` falla. Transcodifica con `ffmpeg` a H.264/`yuv420p` sin audio (`-an`) a un fichero temporal **en el mismo directorio que el vídeo de entrada**, con prefijo `<nombre>_compat_` (línea 470-476, `tempfile.NamedTemporaryFile(..., dir=input_dir, delete=False)`). Ver fuga de ficheros real observada en §17.5.

### 5.4 `probe_video_stream(...)` (líneas 104–156)

Usa `ffprobe -show_streams -show_format` y cachea el resultado en memoria por `(ruta absoluta, mtime)` (`_PROBE_CACHE`, línea 37). El caché **no tiene política de expulsión** (crece sin límite durante la vida del proceso); `clear_probe_cache()` solo se invoca explícitamente en tests.

---

## 6. Fase B — Detección de línea

Fichero: `detector/line_detection.py` (1218 líneas). Implementa `detect_horizontal_line` (101–586) y, de forma simétrica, `detect_vertical_line` (676–1135).

### 6.1 Cadena de algoritmos de detección de segmentos (en este orden exacto, con fallback)

1. **`cv2.HoughLinesP`** (Hough probabilístico) sobre el ROI central tras Canny + cierre morfológico (línea 179-186). Parámetros: `threshold=max(40, 0.04·W)`, `minLineLength=max(30, 0.15·W)` (comentario explícito en código: *"raised from 5% to 15% of width"*, línea 130), `maxLineGap=max(10, 0.02·W)`.
2. Si Hough no produce un candidato válido (no supera el *coverage gate*): **`cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)`** — LSD (*Line Segment Detector*) (línea 346, 901).
3. Si LSD tampoco produce candidato: **`cv2.fitLine`** (ajuste robusto `cv2.DIST_L12`) sobre todos los píxeles de borde Canny del ROI, exigiendo ≥100 píxeles de borde (línea 385-386, *"raised from 20"*).

No se usa Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) ni `goodFeaturesToTrack` en ningún punto de la detección de línea ni del resto del repositorio (verificado por `grep` global — cero coincidencias fuera de la documentación).

### 6.2 Confirmación espectral — análisis DFT/FFT (no Hough, no LSD)

`_verify_line_with_fft` (línea 57-98, horizontal) / `_verify_vertical_line_with_fft` (línea 589-629, vertical):

- `np.fft.fft2` + `np.fft.fftshift` sobre el ROI, con ventana de Hann separable (`np.outer(np.hanning(h), np.hanning(w))`, línea 69) y supresión del bin DC (línea 76). Estos dos refuerzos se añadieron explícitamente en la "Sesión 4" (`IMPROVEMENT_PLAN.md`, Fix 4.1) para reducir *spectral leakage*.
- Para horizontal: compara energía en una banda de columnas alrededor de `u≈0` (eje de frecuencia vertical) frente a una banda de filas alrededor de `v≈0`; ancho de banda = 10 % de la dimensión correspondiente (línea 80, 87).
- `dominance = (energía_horizontal − energía_vertical) / (suma + 1e-8)`; `confirmed = dominance >= fft_min_dominance` (default `0.10`, configurable vía `VPD_LINE_FFT_MIN_DOMINANCE`).

### 6.3 Confirmación espacial — perfil de proyección Sobel (gate opcional)

`_verify_line_with_projection_profile` (línea 632-673): aplica `cv2.Sobel` (orden `(0,1)`, `ksize=3`) sobre el ROI, calcula el perfil de gradiente medio por fila (`np.mean(abs_sobel, axis=1)`), y exige que el pico sea ≥ `profile_min_prominence` veces la mediana (default `3.0`) y que ≥ `profile_min_coverage_ratio` (default `0.20`) de las columnas tengan su pico de gradiente a ±1 fila del pico global. **Desactivado por defecto** (`enable_profile_gate=False` / `VPD_LINE_ENABLE_PROFILE_GATE=false`); cuando está desactivado se calcula igualmente y se registra como clave *advisory* (`profile_confirmed`, `profile_coverage_ratio`, `profile_prominence`) en `debug_line_info["quality_gate"]` sin afectar la decisión (línea 501-502).

### 6.4 Gate de calidad final (`detection_confirmed`)

Para candidatos **no-fallback** (provienen de Hough, no de LSD/fitLine):
```
detection_confirmed = fft_confirmed AND quality_ok AND strict_centered AND continuity_ok
                      AND (slope_tight OR high_coverage)
```
Para candidatos **fallback** (LSD o fitLine) se exige el conjunto más estricto:
```
detection_confirmed = fft_confirmed AND fallback_quality_ok AND strict_centered
                      AND slope_tight AND continuity_ok
```
donde `strict_centered = distancia_al_centro <= tolerancia·0.65`, `slope_tight = pendiente <= max_slope·0.70`, `continuity_ok = continuity_ratio >= 0.70`, `high_coverage = longitud >= strong_coverage_px AND no es fallback` (líneas 470-500, 1020-1050).

**Nota histórica relevante (Sesión 3, `IMPROVEMENT_PLAN.md`)**: antes de un fix aplicado, existía una rama `OR (high_coverage AND ...)` que permitía confirmar una línea de alta cobertura **sin** FFT. Esa rama ya no existe en el código actual; `fft_confirmed` es obligatorio en todas las rutas (verificado leyendo línea 493-500 y 1043-1050 directamente).

`quality_score` (`_compute_candidate_quality`, línea 16-54) es una combinación ponderada: `0.30·coverage_norm + 0.20·strong_coverage_norm + 0.20·center_score + 0.15·slope_score + 0.10·continuity_score + 0.05·span_score`, recortada a `[0,1]`.

### 6.5 Resultado de la fase

`detect_horizontal_line` devuelve `has_horizontal_line: bool`, y si es `True`, un campo `"projection_type": "non_equirectangular"` dentro del dict de retorno. **Este campo nunca se propaga ni se usa en `pipeline.py`** (verificado por `grep`: la única lectura de claves de este resultado en `pipeline.py` es `has_horizontal_line`, `line_data`, `fft_confirmed`, `fft_confidence`, `debug_line_info`); es un valor inerte que no figura entre los 5 valores finales de proyección (§12).

---

## 7. Fase C — Detección estéreo

Fichero: `detector/stereo_detection.py` (256 líneas). Función pública: `detect_stereo(...)` (línea 111-255).

### 7.1 Extracción de regiones comparables alrededor de la costura

`_extract_seam_regions` (línea 58-108) recorta dos mitades (arriba/abajo o izquierda/derecha) **excluyendo una banda de guarda** alrededor del centro de costura detectado (`seam_guard_ratio`, default `0.02` → `VPD_STEREO_SEAM_GUARD_RATIO`). Si el centro de costura no se conoce para un frame, usa el centro geométrico del frame como respaldo (línea 70-71, 91-92). Exige que cada mitad cubra al menos `min_valid_half_ratio` (default `0.22`) de la dimensión total tras el recorte; si no, descarta el frame del cómputo (línea 81-83, 102-104).

### 7.2 Métricas de similitud por frame (tres señales independientes, combinadas)

Por cada frame con costura válida:
1. **Histograma de grises** (256 *bins*, `cv2.calcHist` + `cv2.normalize` con `NORM_MINMAX`) → `correlation` (`cv2.HISTCMP_CORREL`) y `bhattacharyya` (`cv2.HISTCMP_BHATTACHARYYA`) vía `compare_histograms` (línea 17-22).
2. **Similitud de bordes** (`_compute_edge_similarity`, línea 25-41): IoU (*intersection over union*) entre mapas de bordes Canny (`cv2.Canny(gray, 80, 160)`) de ambas mitades.
3. **Score combinado** por frame: `0.55·corr + 0.25·(1−bhatt) + 0.20·edge_sim` (línea 175-179).

Un frame se considera "match" si **las tres** condiciones se cumplen simultáneamente (línea 180-184):
```
corr >= similarity_threshold        (VPD_STEREO_HIST_THRESHOLD, default 0.92)
AND bhatt <= bhattacharyya_threshold  (0.30 — hardcoded, NO configurable vía env)
AND edge_similarity >= edge_similarity_threshold  (VPD_STEREO_EDGE_SIMILARITY_THRESHOLD, default 0.08)
```

**Importante**: `bhattacharyya_threshold` (default `0.30`) y `min_match_ratio` (default `0.60`) son parámetros de la función `detect_stereo` con valores por defecto codificados en `stereo_detection.py:115-116`. Ambas llamadas reales desde `pipeline.py` (líneas 1190-1201 y 1278-1288) **no pasan estos dos argumentos**, por lo que siempre se usan los valores hardcoded — no existe ninguna variable de entorno que los controle, a diferencia de `similarity_threshold` y `edge_similarity_threshold`, que sí son configurables.

### 7.3 Decisión final de "es estéreo"

```
is_stereo = (frames_evaluados >= min_frames_required)        # VPD_STEREO_MIN_SEAM_FRAMES, default 2
        AND (match_ratio >= min_match_ratio)                  # hardcoded 0.60
        AND (stability_ratio >= min_stability_ratio)          # VPD_STEREO_MIN_STABILITY_RATIO, default 0.55
```
(`stereo_detection.py:231-235`). `stability_ratio = 1 − (racha_más_larga_de_no-match / frames_evaluados)` (línea 225-230) — penaliza específicamente que los fallos de coincidencia se agrupen consecutivamente, no solo su frecuencia total.

### 7.4 Arreglo de la línea principal (`detector/pipeline.py`)

Se invoca dos veces con distinta orientación, nunca simultáneamente para el mismo conjunto de frames:
- `arrangement="up-down"` (por defecto) cuando hay frames con **línea horizontal** (línea 1278).
- `arrangement="left-right"` cuando **no** hay ninguna línea horizontal pero sí líneas **verticales** candidatas (línea 1190) — este es el único camino hacia `stereo_equi` por disposición lado-a-lado.

Si `len(indices_con_linea) < stereo_min_seam_frames`, la comprobación de estéreo arriba/abajo se omite por completo y se simula un resultado `is_stereo=False` (línea 1253-1276) sin invocar `detect_stereo`.

---

## 8. Fase D — Análisis de movimiento / flujo óptico

Fichero: `detector/motion_analysis.py` (423 líneas). Esta es la fase que decide EAC vs cúbica cuando hay costura central pero no es estéreo.

### 8.1 Tipo de flujo óptico: **denso**, nunca disperso

Todos los backends usados producen un campo de flujo denso (`H×W×2`, un vector por píxel). **No se usa Lucas-Kanade ni ningún método de flujo disperso** (sin `goodFeaturesToTrack` + `calcOpticalFlowPyrLK` en ningún punto del repositorio, verificado por búsqueda global). El nombre `sparse_to_dense` (uno de los backends soportados) es engañoso si se lee de forma aislada: es el algoritmo `cv2.optflow.createOptFlow_SparseToDense()` de OpenCV, que internamente interpola desde correspondencias dispersas pero **emite una salida densa** del mismo formato `H×W×2` que el resto — no es un tracker tipo Lucas-Kanade ni devuelve puntos discretos.

### 8.2 Backends reales y su disponibilidad

| Algoritmo | Llamada exacta | Requiere módulo `cv2.optflow` (contrib) |
|---|---|---|
| Farneback | `cv2.calcOpticalFlowFarneback(...)` (`motion_analysis.py:104-115`) | No (siempre disponible) |
| DIS (*Dense Inverse Search*) | `cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)` (línea 186) | No (parte del módulo `video` principal desde OpenCV 4.x) |
| TV-L1 (`DualTVL1`) | `cv2.optflow.createOptFlow_DualTVL1()` (línea 125) | **Sí** |
| DeepFlow | `cv2.optflow.createOptFlow_DeepFlow()` (línea 128) | **Sí** |
| PCAFlow | `cv2.optflow.createOptFlow_PCAFlow()` (línea 131) | **Sí** |
| SparseToDense (salida densa) | `cv2.optflow.createOptFlow_SparseToDense()` (línea 134) | **Sí** |
| Refinamiento variacional (post-proceso opcional) | `cv2.optflow.createVariationalRefinement()` (línea 156) | **Sí** |

Parámetros Farneback fijos en código (`motion_analysis.py:103-115`): `pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0`. No son configurables vía entorno.

### 8.3 Cadena de fallback de selección de algoritmo

`build_flow_fallback_chain(...)` (línea 67-100): construye una lista ordenada de candidatos a partir de un algoritmo preferido, las listas de "tier B" (`["tvl1","dis"]`) y "tier C" (`["deepflow","pcaflow","sparse_to_dense"]`), deduplica, y **filtra** dejando solo los disponibles según `get_opencv_capabilities()` (línea 21-39, basada en `hasattr`). `"farneback"` se garantiza siempre presente al final de la cadena si ningún otro candidato sobrevive al filtro (línea 98-99). En tiempo de ejecución, `compute_optical_flow` (línea 162-198) recorre la cadena en orden y usa el **primer** algoritmo que no lance excepción; si todos fallan, recurre explícitamente a Farneback (línea 195-196).

### 8.4 Resolución real del algoritmo configurado — `_resolve_motion_feature_flags` (`detector/pipeline.py:74-159`)

Este es el punto más sutil del sistema: el valor de `VPD_FLOW_ALGORITHM` (default `"deepflow"`, `config/settings.py:193`) **no se usa directamente** salvo en el perfil `"baseline"`. Para los perfiles por defecto (`"robust"`, `"high_accuracy"`):

```python
preferred_chain = []
if requested_algorithm in {"tvl1", "dis", "farneback"}:
    preferred_chain.append(requested_algorithm)     # "deepflow" NO entra aquí
preferred_chain.extend(tier_b_flow_algorithms)        # ["tvl1", "dis"]
preferred_chain.append("farneback")
preferred_chain.extend(tier_c_flow_algorithms)        # ["deepflow", "pcaflow", "sparse_to_dense"]
```
(`pipeline.py:120-126`). Como `"deepflow"` no pertenece a `{"tvl1","dis","farneback"}`, **nunca se antepone** a Tier B. El primer elemento de `preferred_chain` cuyo *fallback chain propio* sea no vacío gana — y como Tier B (`tvl1`, `dis`) se evalúa antes que Tier C (`deepflow`, ...), el algoritmo efectivamente seleccionado es:

- `"tvl1"` si `cv2.optflow.createOptFlow_DualTVL1` está disponible (típico con `opencv-contrib-python` completo).
- `"dis"` si no hay módulo `optflow` pero sí `cv2.DISOpticalFlow_create` (como en el venv de este repositorio, ver §18.1).
- `"farneback"` como último recurso.

Es decir: **el valor por defecto documentado de `VPD_FLOW_ALGORITHM` (`deepflow`) casi nunca es el algoritmo que realmente se ejecuta** en los perfiles `robust`/`high_accuracy` (que son el perfil por defecto), salvo que ni `tvl1` ni `dis` estén disponibles y `deepflow` sí. Este comportamiento está confirmado por test explícito: `tests/test_detection_retry.py::test_high_accuracy_prefers_tier_b_over_requested_tier_c` fuerza `flow_algorithm="deepflow"` con `has_tvl1=True` y comprueba `flags["flow_algorithm"] == "tvl1"`. El propio `SCAN_REPORT.md` (§4, punto 8) documenta un caso simétrico relacionado (ver §18.4).

En el perfil `"baseline"` (`pipeline.py:118-119`): `selected_flow_algorithm = "dis"` solo si se pidió explícitamente `"dis"` y está disponible; en cualquier otro caso (incluido pedir `"deepflow"` o `"tvl1"`) se usa `"farneback"` sin más fallback.

### 8.5 Refinamiento variacional y consistencia *forward-backward*

- `enable_flow_refinement` / `enable_forward_backward_check` / `enable_geometry_evidence`: sus valores de entorno (`VPD_FLOW_ENABLE_REFINEMENT`, `VPD_FLOW_ENABLE_FB_CHECK`, `VPD_ENABLE_GEOMETRY_EVIDENCE`, todos con default `false`) **se sobrescriben a `True`** cuando el perfil es `"robust"` o `"high_accuracy"` (`pipeline.py:107-114`) — que es el perfil por defecto. Es decir, en la configuración de fábrica estas tres funciones están **activas** aunque sus variables de entorno digan `false`.
- Consistencia forward-backward (`compute_forward_backward_consistency_mask`, `motion_analysis.py:201-228`): calcula el flujo en ambos sentidos y máscara los píxeles donde `||flow_fwd + flow_bwd|| > fb_threshold` (default `1.5` px, `VPD_FLOW_FB_THRESHOLD`). Esto **duplica el coste de cómputo de flujo óptico** por par de frames cuando está activo.
- Refinamiento variacional (`_apply_variational_refinement`, línea 141-159): no-op silencioso si `cv2.optflow.createVariationalRefinement` no existe en el build de OpenCV — no produce error ni log, simplemente devuelve el flujo sin modificar.

### 8.6 División en regiones y agregación direccional

`split_into_regions` (línea 231-246): rejilla fija de **2 filas × 3 columnas = 6 regiones** por frame — coincide exactamente con la suposición de *layout* de cubemap 3×2 documentada en `CUBEMAP_LAYOUT` (§9) y en los *known issues* (§17.1).

`compute_region_motion` (línea 249-316): para cada región, máscara los píxeles cuya magnitud de flujo supera `umbral_magnitud` (default `1.0`), recorta valores atípicos al percentil 97 (`outlier_percentile`, rango forzado `[85, 99.5]`), y calcula un **ángulo medio circular ponderado** (suma vectorial de `cos`/`sin` ponderada por magnitud) y una **concentración** = `|suma vectorial| / suma de pesos` (línea 308-309) — una región solo es válida si supera `min_concentration` (default `0.25`) y `min_active_ratio` (default `0.06`), ver `region_validation.py:6-18`.

### 8.7 Refinamiento del ángulo por región vía ORB + RANSAC (sobrescribe el ángulo de flujo óptico)

`compute_region_affine_angles` (`motion_analysis.py:319-365`): por cada región, detecta puntos ORB (`nfeatures=300`), empareja con `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`, ajusta una transformación afín parcial (`cv2.estimateAffinePartial2D`, RANSAC, `ransacReprojThreshold=3.0`, mínimo `6` *inliers*) y extrae el ángulo de rotación `atan2(M[1,0], M[0,0])`.

**Punto no obvio**: en `pipeline.py:283-287`, si este ángulo afín está disponible para una región válida, **sobrescribe** el ángulo derivado del flujo óptico circular-medio (§8.6) antes de pasar a los scorers EAC/cúbico. Es decir, la dirección que finalmente decide EAC-vs-cúbica proviene preferentemente de **ORB + RANSAC affine**, no directamente del flujo óptico denso, siempre que el emparejamiento de características tenga éxito; el flujo óptico solo aporta el ángulo cuando ORB no encuentra suficientes coincidencias.

### 8.8 Evidencia geométrica global (homografía)

`compute_global_geometry_evidence` (`motion_analysis.py:368-422`): ORB global (`nfeatures=500`), `cv2.findHomography` con `cv2.USAC_MAGSAC` si está disponible (si no, `cv2.RANSAC`), `ransacReprojThreshold=3.0`, mínimo `12` *inliers*. Calcula:
```
quality = clip((inlier_ratio − 0.25) / 0.55, 0, 1) · 0.70 + clip((4.0 − reproj_error) / 4.0, 0, 1) · 0.30
eac_score = quality ;  cubic_score = 1 − quality
```
Esta es una heurística: **asume** que una homografía global de mayor calidad (más *inliers*, menor error de reproyección) entre dos frames consecutivos es más compatible con geometría EAC que cúbica — no es una prueba geométrica rigurosa por proyección, sino una señal adicional que se funde (cuando `enable_geometry_evidence=True`, activo por defecto) con los scores de `projection_logic.py` mediante `_fuse_optional_score` (`pipeline.py:162-172`), con un peso `geometry_evidence_weight` recortado a `[0.0, 0.45]` (default `0.20`, `VPD_GEOMETRY_EVIDENCE_WEIGHT`).

### 8.9 Etiqueta de "Tier A" — metadato declarativo sin efecto en el flujo de control

`_DEFAULT_MOTION_FEATURE_TIERS["tier_a_features"]` (`pipeline.py:59-68`, replicado en `config/settings.py:202-211`) lista nombres de técnicas: `canny_morphology_houghlinesp`, `line_segment_detector`, `dft_orientation_checks`, `orb_bfmatcher`, `find_homography_ransac_usac_magsac`, `estimate_affine_partial_2d`, `forward_backward_consistency`, `variational_refinement`. **Esta lista nunca se lee** en ninguna condición del código (confirmado por `grep` de `tier_a_features` — solo aparece en su propia definición); es documentación incrustada en una estructura de datos, no una palanca de control. Las técnicas que nombra sí están todas presentes en el código, pero de forma incondicional (no gateadas por esta lista).

---

## 9. Fase E — Scoring EAC vs Cúbica

Fichero: `detector/projection_logic.py` (177 líneas). Esta es la fase que, dado el ángulo direccional de cada una de las 6 regiones (§8.6–8.7), decide si el patrón de movimiento observado es más compatible con EAC o con cubemap.

### 9.1 Las dos tablas de *layout* — correcciones angulares por cara

```python
EAC_LAYOUT = {
    (0,0): ("LEFT",   0.0),
    (0,1): ("FRONT",  0.0),
    (0,2): ("RIGHT",  0.0),
    (1,0): ("BOTTOM", -π/2),
    (1,1): ("BACK",   +π/2),
    (1,2): ("TOP",    +π/2),
}
CUBEMAP_LAYOUT = {
    (0,0): ("RIGHT",  0.0),
    (0,1): ("LEFT",   0.0),
    (0,2): ("TOP",    0.0),
    (1,0): ("BOTTOM", 0.0),
    (1,1): ("FRONT",  0.0),
    (1,2): ("BACK",   0.0),
}
```
(`projection_logic.py:9-24`). Ambas asumen la rejilla 2×3 de `split_into_regions` (§8.6) y por tanto heredan la **suposición de layout cubemap 3×2** documentada en `README.md` y en `IMPROVEMENT_PLAN.md`/`SCAN_REPORT.md` (ver §17.1).

### 9.2 Lógica de scoring EAC — simetría relativa, NO ángulos absolutos fijos

`_score_layout_eac` (línea 35-66): **no** compara contra un ángulo absoluto de 90°/45° fijo. En su lugar:
- Si están disponibles `LEFT`, `RIGHT` y `FRONT`: exige que `LEFT` y `RIGHT` sean **simétricos alrededor de `FRONT`** (`diff_left ≈ −diff_right`); error = `|diff_left + diff_right|`.
- Si falta `FRONT`: exige que `LEFT` y `RIGHT` sean **opuestos entre sí** (`|diff_LR| ≈ π`, es decir ≈180°).
- Si están disponibles `BACK` y `FRONT`: exige que difieran en **≈180°** (`error = |mag_diff − π|`).

En **ningún caso** el scorer EAC comprueba directamente una divergencia de ±90° entre caras — eso solo ocurre en el scorer cúbico (§9.3).

### 9.3 Lógica de scoring cubemap — aquí sí aparece la divergencia de ±90°

`_score_layout_cubemap` (línea 69-104): para `LEFT` y `RIGHT` respecto a `FRONT`:
```python
half_pi = π / 2.0
error_left  = abs(abs(diff_left)  - half_pi)   # exige ≈90° respecto a FRONT
error_right = abs(abs(diff_right) - half_pi)   # exige ≈90° respecto a FRONT
```
(línea 73-87). Esta es la única ubicación del código donde literalmente se compara una diferencia angular contra `π/2` (90°) — y es específica del scorer **cúbico**, no del EAC. `BACK` vs `FRONT` se sigue evaluando contra `π` (180°), igual que en EAC.

**Conclusión verificable**: la afirmación "el sistema decide EAC vs cúbica comprobando una divergencia angular de ±90° entre caras" es **literalmente cierta solo para la hipótesis cúbica** (`projection_logic.py:77,84`); para la hipótesis EAC el criterio es de **simetría relativa**, no de ángulo absoluto fijo. Generalizar "±90°" a ambas hipótesis por igual sería impreciso.

### 9.4 Tolerancia angular — el parámetro se llama "45" pero su valor por defecto es 20°

```python
def evaluate_eac(regions, min_concentration=0.25, min_active_ratio=0.06, tolerancia_45_deg: float = 20.0):
def evaluate_cubemap(regions, min_concentration=0.25, min_active_ratio=0.06, tolerancia_45_deg: float = 20.0):
```
(`projection_logic.py:135-164`). El parámetro `tolerancia_45_deg` se convierte a radianes (`np.radians(tolerancia_45_deg)`, línea 123) y se usa como divisor lineal del error angular: `score = max(0, 1 − error/tolerancia_rad)`. **El nombre del parámetro sugiere 45°, pero su valor por defecto literal en todo el código es `20.0` grados** (mismo valor en `pipeline.py:206`, que es quien lo invoca en producción). No hay ninguna ruta de código donde este parámetro tome el valor 45.

### 9.5 Decisión final — `decide_projection(...)` (línea 167-177)

```python
def decide_projection(score_eac, score_cubemap, min_margin=0.0) -> Optional[bool]:
    if score_eac is None or score_cubemap is None:
        return None
    if abs(score_eac - score_cubemap) < min_margin:
        return None
    return score_eac >= score_cubemap
```
Devuelve `True` (EAC), `False` (cúbica) o `None` (datos insuficientes / margen demasiado estrecho) **para un único par de frames**. Esta decisión por par se agrega después a través de muchos pares ponderados en `_classify_non_equirectangular` (§11.2) — `decide_projection` en sí mismo no produce la clasificación final del vídeo.

---

## 10. Fase F — Evidencia equirectangular positiva

Fichero: `detector/equirectangular_detection.py` (399 líneas). A diferencia de las fases B–E (que buscan evidencia de **no** ser equirectangular), esta fase busca evidencia **positiva** de que un frame **sí** es equirectangular, evaluando la continuidad de "wrap-around" en los bordes izquierdo/derecho.

### 10.1 Señales por frame — `compute_frame_equirectangular_evidence(...)` (línea 239-327)

1. **Score de relación de aspecto** (`compute_aspect_ratio_score`, línea 29-61): campana de coseno alrededor de `target_ratio=2.0` (relación 2:1 característica de equirectangular), con tolerancia `±0.20`; fuera de la tolerancia decae linealmente hasta un suelo de `0.05`.
2. **Score de "wrap"** (`compare_strips_with_vertical_shift`, línea 118-236): extrae tiras verticales de los bordes izquierdo/derecho (`strip_ratio=0.04` del ancho, mínimo `12` px), busca el mejor desplazamiento vertical en `±max_shift_ratio` (`0.03` de la altura de la tira) y combina cuatro métricas:
   ```
   wrap_score = 0.35·hist_corr + 0.25·(1−hist_bhatt) + 0.20·(1−nmad·4) + 0.20·edge_sim
   ```
   (línea 214-219).
3. **Combinación final por frame**: `score = 0.80·wrap_score + 0.20·aspect_ratio_score` (`wrap_weight=0.80`, `aspect_weight=0.20`, línea 246-247, 309).

### 10.2 Agregación a nivel de vídeo — `aggregate_equirectangular_evidence(...)` (línea 330-398)

```
final_score = 0.70·mediana(scores) + 0.30·media(scores)
is_strong_evidence = (final_score >= strong_frame_threshold=0.65) AND (strong_ratio >= min_strong_ratio=0.30)
```
Solo se consideran frames con `usable=True` (frame no vacío, ≥16×32 px, tiras extraíbles). Esta evidencia se usa en dos puntos de `pipeline.py`:
- Rama temprana sin línea horizontal (§4.2): determina la confianza reportada cuando se clasifica como `equirectangular` directamente.
- **Regla de rescate B** (`pipeline.py:1433-1445`): puede reclasificar un resultado `unknown`/no fiable como `equirectangular` si esta evidencia es fuerte y el test de estéreo dio negativo.

---

## 11. Orquestación, política de fiabilidad y reintentos

Fichero: `detector/pipeline.py` (1726 líneas). Las funciones clave son `_classify_non_equirectangular` (líneas 366-870), `run_detection_pipeline` (873-1543) y `run_detection_with_retries` (1571-1726).

### 11.1 Plan de pares de frames por secuencia secundaria

`_build_pair_plan` (línea 387-413, función interna de `_classify_non_equirectangular`): para una secuencia de longitud `seq_len`, genera pares `(i, j, gap)` con `gap ∈ {1, min(2,max_gap), min(3,max_gap), max_gap}` (deduplicados). Con secuencias de 3 frames (el caso normal, §5.2), esto produce típicamente los pares `gap=1` (`(0,1)`, `(1,2)`) y `gap=2` (`(0,2)`).

Existe una fase de **pre-evaluación rápida** (línea 469-510): se analizan primero los pares de `gap=1` ("short pairs"); si la confianza preliminar (`prelim_confidence`) es baja (`< 0.35`) o la señal de movimiento media es débil (`< 0.035`), se **priorizan los pares de mayor `gap`** en la evaluación principal (`prioritize_long=True`, línea 510-513) — la intuición es que con poco movimiento entre frames adyacentes, separar más los frames en el tiempo puede revelar más señal.

### 11.2 Ponderación de cada par — `_pair_weight(...)` (línea 415-432)

```
motion_component = min(1, motion_strength / 0.10)
gap_boost = min(1.35, 1.0 + 0.08·(gap−1))
weight = max(0.1, (0.30 + 0.35·motion_component + 0.20·min(1,layout_margin) + 0.15·clip(geometry_quality,0,1)) · gap_boost)
```
Los pares con mayor separación temporal (`gap` más alto) reciben hasta un **35 % más de peso** (`gap_boost` máximo `1.35` en `gap=5`). Los votos ponderados de todos los pares válidos de todas las secuencias se acumulan en `weighted_eac` / `weighted_cubic`; `ratio_eac = weighted_eac / (weighted_eac + weighted_cubic)`.

### 11.3 Detección de "movimiento bajo" y relajación adaptativa de umbrales

```python
low_motion_detected = (
    low_motion_ratio >= 0.60
    or mean_pair_active_ratio < 0.055
    or mean_pair_magnitude < 1.15
)
```
(`pipeline.py:678-685`; un par individual se marca "bajo" si `pair_mean_magnitude < 1.15` o `pair_mean_active_ratio < 0.05`, línea 434-437). Cuando se detecta, los umbrales de aceptación se relajan (línea 716-719):
```
effective_min_valid_pairs        = max(2, min_valid_pairs − 1)            # p.ej. 4 → 3
effective_min_motion_confidence  = max(0.10, min_motion_confidence · 0.60) # p.ej. 0.2 → 0.12
effective_dominant_ratio_threshold = max(0.65, dominant_ratio_threshold − 0.10) # p.ej. 0.8 → 0.70
```

### 11.4 Decisión de clasificación (EAC / cúbica / unknown) a nivel de vídeo

```
si ratio_eac  >= umbral_dominante_efectivo → "eac"
si ratio_cubic >= umbral_dominante_efectivo → "cubic"
si no:
    si |ratio_eac − ratio_cubic| <= ambiguity_gap (default 0.10):
        # intenta desambiguar con 3 señales secundarias, en este orden:
        1. voto de evidencia geométrica (geometry_vote_total>=0.20 y gap>=0.10)
        2. ratio de pares de gap largo (long_gap_total>=0.20 y gap>=0.12)
        3. persistencia de la decisión más larga (racha de aciertos consecutivos >=0.60)
        si ninguna desambigua → "unknown"
    si no → gana la hipótesis con mayor ratio
```
(`pipeline.py:746-764`).

### 11.5 Cálculo de `motion_confidence` — fórmula distinta según el perfil

```python
# high_accuracy (perfil por defecto):
motion_confidence = 0.32·base_confidence + 0.20·layout_confidence + 0.15·motion_signal_confidence
                   + 0.23·geometry_confidence + 0.10·seam_confidence
# robust:
motion_confidence = 0.40·base_confidence + 0.25·layout_confidence + 0.20·motion_signal_confidence
                   + 0.10·geometry_confidence + 0.05·seam_confidence
# baseline (cualquier otro valor de perfil):
motion_confidence = 0.60·base_confidence + 0.20·layout_confidence + 0.15·motion_signal_confidence
                   + 0.05·seam_confidence   # (sin componente de geometría)
```
(`pipeline.py:766-793`), donde `base_confidence = |ratio_eac − 0.5| · 2`.

### 11.6 Gate de fiabilidad y "forzado por evidencia consistente"

`reliable` es `True` solo si **ninguna** razón de no-fiabilidad aparte de `low_motion_detected:` está presente (línea 815) entre: `few_valid_pairs`, `low_motion_confidence`, `low_layout_margin`, `weak_dominance` (línea 796-813).

Si no es fiable pero la clasificación no es ya `unknown`, hay una **excepción de rescate** (línea 816-833):
```
contradictory_evidence = (score_margin <= max(0.03, ambiguity_gap·0.50))
                          AND |long_gap_ratio_eac − long_gap_ratio_cubic| < 0.10
                          AND |geometry_vote_eac − geometry_vote_cubic| < 0.08
                          AND max(persistence_eac, persistence_cubic) < 0.58
strong_consistency = max(persistence_eac, persistence_cubic) >= 0.65
                      OR |long_gap_ratio_eac − long_gap_ratio_cubic| >= 0.18
                      OR |geometry_vote_eac − geometry_vote_cubic| >= 0.14
                      OR seam_confidence >= 0.80

si contradictory_evidence OR NOT strong_consistency → clasificacion = "unknown"
si no → reliable = True ; reliability_reason += "forced_by_consistent_evidence"
```
Esto permite que un resultado que técnicamente no alcanzó los umbrales de fiabilidad "normales" se acepte igualmente como `eac`/`cubic` si la evidencia de señales independientes (persistencia temporal, votos de geometría, pares de gap largo) es fuertemente consistente entre sí — confirmado por test explícito (`tests/test_motion_classification_policy.py::test_forces_classification_with_consistent_evidence`).

### 11.7 `run_detection_with_retries` — plan de reintentos (líneas 1546-1726)

`_build_detection_retry_plan(num_frames, max_retries=2)` (línea 1546-1568) genera hasta 3 intentos con: número de frames decreciente (`num_frames`, `num_frames−2`, `num_frames−4`, mínimo 2), `paso_frames_secundarios` creciente (`5, 9, 13`), y `min_frames_with_line_required = max(2, int(num_frames_intento · 0.55))`. **Este valor calculado siempre sustituye** al parámetro por defecto `min_frames_with_line_required=7` de `run_detection_pipeline` (línea 881) en la ruta de producción — el literal `7` solo se observa si se llama a `run_detection_pipeline` directamente sin pasar por `run_detection_with_retries` (p. ej. en tests o en `workflows.unified_pipeline.analyze_video_projection`, el *wrapper* de conveniencia sin reintentos, línea 568-575 de `unified_pipeline.py`).

Criterios de parada anticipada del bucle de reintentos (línea 1683-1698):
- `motion_reliable == True` → para.
- `projection_type == "equirectangular"` (clasificación temprana) → para.
- Si la causa fue "movimiento bajo" y la ganancia de señal de movimiento entre intentos consecutivos es `< 0.005` → para (evita reintentos inútiles cuando el contenido es genuinamente estático).

Si ninguna condición de parada se cumple, el siguiente intento usa `paso_frames_secundarios` aumentado en al menos `+6` y reduce `num_frames` y `min_frames_with_line_required` en consecuencia (línea 1700-1713).

`min_valid_pairs=4` y `min_motion_confidence=0.2` están **hardcoded como variables locales** en `run_detection_with_retries` (línea 1590-1591) — no provienen de `config/settings.py` ni de ninguna variable de entorno.

---

## 12. Dominio de salida completo y degradación a `unknown`

### 12.1 Los 5 valores finales posibles de `projection_type`

Confirmado por `detector/__init__.py:10` (docstring del paquete) y por inspección exhaustiva de todas las asignaciones a la variable `projection_type` en `detector/pipeline.py`:

| Valor | ¿Cuándo se produce? |
|---|---|
| `"equirectangular"` | Sin línea horizontal Y sin estéreo lado-a-lado (§4.2); o regla de rescate B (§4.1, paso 8) |
| `"stereo_equi"` | `detect_stereo` positivo, en cualquiera de las dos orientaciones (arriba/abajo o izquierda/derecha) |
| `"eac"` | `_classify_non_equirectangular` decide EAC Y `motion_reliable=True` |
| `"cubic"` | `_classify_non_equirectangular` decide cúbica Y `motion_reliable=True` |
| `"unknown"` | Valor por defecto/degradado en cualquier punto donde la fiabilidad no se alcanza (ver §12.2) |

No existe un sexto valor en el código de producción. (`"non_equirectangular"` es un valor interno e inerte, ver §6.5; el módulo `app/gui/gui_app.py:788` usa el texto de respaldo `"desconocida"` solo para presentación en la UI, no como valor de dominio).

### 12.2 Puntos exactos donde el resultado se degrada a `unknown`

1. **Sin pares de movimiento válidos** (`_classify_non_equirectangular`, línea 687-711): si `pares_validos == 0`.
2. **Evidencia contradictoria o no-consistente tras fallar el gate normal** (línea 829-830, ver §11.6).
3. **Insuficiencia global de datos** (`evaluar_suficiencia_datos`, `pipeline.py:184-198`, invocada en línea 1417-1428): si `frames_analizados < min_frames_analyzed_required` (default `4`) o el ratio de descarte (`frames_descartados/frames_extraídos`) supera `max_discard_ratio` (default `0.65`) — **excepto** si la clasificación ya era `"equirectangular"`, que está exenta de esta comprobación (línea 1425: `if projection_type != "equirectangular" and insufficiency_reasons:`).
4. **Gate final post-clasificación** (línea 1430-1431): `if projection_type in ("eac","cubic") and not motion_reliable: projection_type = "unknown"` — esta es la última palabra; ninguna clasificación EAC/cúbica sobrevive sin `motion_reliable=True`.
5. **Frames estructurales insuficientes** (línea 1323-1414): si no se cumple ni el umbral absoluto ni el "ratio gate" del 50 %, nunca se llega a invocar `_classify_non_equirectangular`; el resultado por defecto de `projection_type` (inicializado en línea 1164) permanece `"unknown"`.
6. **Sin secuencias secundarias disponibles** (línea 1396-1399): aunque se cumplan los umbrales de líneas, si `secuencias_secundarias` está vacío no hay datos de movimiento que analizar.
7. **Excepción no controlada** en `run_detection_pipeline` (bloque `except Exception` global, línea 1508-1543): devuelve `unknown` con `confidence=0.0` y el mensaje de error en `result["error"]`.

### 12.3 Comportamiento de fallback `unknown → eac` — dónde se aplica exactamente

**Este fallback NO ocurre dentro del detector.** Está implementado únicamente en la capa de orquestación, en `workflows/unified_pipeline.py::_stage_convert_to_equirectangular` (línea 282-286):
```python
if projection_type == "unknown":
    logger.warning("Projection detection returned 'unknown'; falling back to EAC for conversion.")
    projection_type = "eac"
```
- `JobResult.projection_type` **no se modifica** por este fallback: se asigna antes (línea 487 de `unified_pipeline.py`, `result.projection_type = detection.get("projection_type", "unknown")`) y nunca se reescribe después. Solo la variable local usada para elegir el filtro `v360` cambia a `"eac"`.
- La función de bajo nivel `detector/projection_conversion.py::convert_detected_projection_to_equirectangular` trata `"unknown"` como un **skip** explícito (`reason="projection_unknown"`, línea 619-626) — **no** aplica ningún fallback a `eac`. Por tanto, si se invoca el detector/conversor directamente (p. ej. vía `detector.convert_to_equirectangular`, `detector/__init__.py:37-57`) sin pasar por `workflows/unified_pipeline.py`, una proyección `"unknown"` **se omite**, no se convierte a EAC.

---

## 13. Tabla maestra de constantes, umbrales y variables de entorno

### 13.1 Variables de entorno (`config/settings.py`, todas con prefijo `VPD_` salvo las indicadas)

| Variable | Valor por defecto | Controla |
|---|---|---|
| `YOUTUBE_API_KEY` | `None` | Clave YouTube Data API v3 |
| `CMS_API_URL` | `None` | Endpoint de subida MediaCMS |
| `CMS_USER` | `$USER` o `None` | Usuario HTTP Basic Auth de MediaCMS |
| `CMS_PASSWORD` | `None` | Password de MediaCMS |
| `CMS_TOKEN` | `None` | Token CSRF de MediaCMS |
| `VPD_PROJECT_ROOT` | autodetectado | Raíz del proyecto |
| `DOWNLOADS_DIR` | `data/downloads` | Directorio de descargas/conversión |
| `VPD_FRAMES_OUTPUT_DIR` | `data/frames` | Directorio de frames de análisis del detector |
| `VPD_DEBUG_OUTPUT_DIR` | = `VPD_FRAMES_OUTPUT_DIR` | Directorio de imágenes de depuración |
| `VPD_MIN_FRAMES_ANALYZED` | `4` | Mínimo de frames analizados para no marcar insuficiencia de datos |
| `VPD_MAX_DISCARD_RATIO` | `0.65` | Máxima proporción de frames descartados tolerada |
| `VPD_MIN_LAYOUT_SCORE_MARGIN` | `0.10` | Margen mínimo de score EAC/cúbico por par para considerarlo decidible |
| `VPD_LINE_CENTER_BAND_RATIO` | `0.08` | **Definido pero nunca leído en ningún cálculo** — confirmado por `grep`; legado inerte |
| `VPD_LINE_CENTER_SEARCH_BAND_RATIO` | `0.02` | Ancho de la banda ROI central para Hough/LSD |
| `VPD_LINE_CENTER_MAX_DISTANCE_RATIO` | `0.02` | Distancia máxima de la línea candidata al centro |
| `VPD_LINE_MAX_SLOPE` | `0.05` | Pendiente máxima aceptada para considerar "horizontal"/"vertical" |
| `VPD_LINE_MIN_COVERAGE_RATIO` | `0.20` | Cobertura mínima de la línea respecto al ancho/alto del frame |
| `VPD_LINE_MIN_QUALITY_SCORE` | `0.62` | Score de calidad mínimo (candidatos no-fallback) |
| `VPD_LINE_FALLBACK_MIN_QUALITY_SCORE` | `0.78` | Score de calidad mínimo (candidatos LSD/fitLine) |
| `VPD_LINE_FFT_MIN_DOMINANCE` | `0.10` | Dominancia espectral mínima para confirmar línea vía FFT |
| `VPD_LINE_STRONG_COVERAGE_RATIO` | `0.40` | Cobertura a partir de la cual se activa `high_coverage` |
| `VPD_LINE_MORPH_LENGTH_RATIO` | `0.02` | Longitud del kernel de cierre morfológico tras Canny |
| `VPD_LINE_ENABLE_PROFILE_GATE` | `false` | Activa el gate espacial Sobel como bloqueante (si no, es solo *advisory*) |
| `VPD_LINE_PROFILE_MIN_COVERAGE_RATIO` | `0.20` | Cobertura mínima de columnas/filas que coinciden en el pico de gradiente |
| `VPD_LINE_PROFILE_MIN_PROMINENCE` | `3.0` | Relación pico/mediana de gradiente mínima |
| `VPD_STEREO_HIST_THRESHOLD` | `0.92` | Correlación de histograma mínima para "match" estéreo |
| `VPD_STEREO_SEAM_GUARD_RATIO` | `0.02` | Banda de guarda excluida alrededor de la costura |
| `VPD_STEREO_MIN_VALID_HALF_RATIO` | `0.22` | Tamaño mínimo de cada mitad tras el recorte de guarda |
| `VPD_STEREO_EDGE_SIMILARITY_THRESHOLD` | `0.08` | Similitud de bordes (IoU) mínima para "match" |
| `VPD_STEREO_MIN_SEAM_FRAMES` | `2` | Frames con costura mínimos antes de evaluar estéreo |
| `VPD_STEREO_MIN_STABILITY_RATIO` | `0.55` | Estabilidad temporal mínima de coincidencias estéreo |
| `VPD_SAVE_STEREO_HALVES` | `true` | Guardar imágenes de depuración de las mitades estéreo |
| `VPD_FLOW_ALGORITHM` | `deepflow` | Algoritmo de flujo óptico preferido (**ver §8.4 — en la práctica casi nunca es el ejecutado bajo los perfiles por defecto**) |
| `VPD_FLOW_ENABLE_REFINEMENT` | `false` | Refinamiento variacional opcional (**forzado a `true` bajo perfil `robust`/`high_accuracy`**) |
| `VPD_FLOW_ENABLE_FB_CHECK` | `false` | Consistencia forward-backward (**idem, forzado a `true`**) |
| `VPD_FLOW_FB_THRESHOLD` | `1.5` | Umbral de error FB en píxeles |
| `VPD_ENABLE_GEOMETRY_EVIDENCE` | `false` | Fusión de evidencia geométrica ORB/homografía (**idem, forzado a `true`**) |
| `VPD_GEOMETRY_EVIDENCE_WEIGHT` | `0.20` | Peso de fusión (recortado en código a máx. `0.45`) |
| `VPD_MOTION_ROLLOUT_PROFILE` | `high_accuracy` | Perfil: `baseline` \| `robust` \| `high_accuracy` |
| `VPD_FORCE_FULL_CODEC_NORMALIZATION` | `false` | Reactiva la transcodificación completa previa a la detección |

### 13.2 Constantes embebidas en código (NO configurables vía entorno)

| Constante | Valor | Fichero:línea |
|---|---|---|
| `bhattacharyya_threshold` (estéreo) | `0.30` | `stereo_detection.py:115` |
| `min_match_ratio` (estéreo) | `0.60` | `stereo_detection.py:116` |
| `min_valid_pairs` (reintentos) | `4` | `pipeline.py:1590` |
| `min_motion_confidence` (reintentos) | `0.2` | `pipeline.py:1591` |
| `min_frames_with_line_required` (default de función, sustituido en producción) | `7` | `pipeline.py:881` |
| `gaussian_kernel_size` / `gaussian_sigma` | `5` / `1.2` | `pipeline.py:1586-1587` |
| `tolerancia_45_deg` (nombre engañoso, valor real) | `20.0°` | `pipeline.py:206`; `projection_logic.py:139,155` |
| `umbral_concentracion` / `min_concentration` | `0.25` | `pipeline.py:207`; `region_validation.py:8` |
| `min_active_ratio` | `0.06` | `pipeline.py:210,375`; `region_validation.py:9` |
| `dominant_ratio_threshold` | `0.8` | `pipeline.py:376` |
| `ambiguity_gap` | `0.10` | `pipeline.py:377` |
| `frame_esta_en_negro`: `umbral_intensidad` / `proporcion_minima_oscura` | `16` / `0.98` | `pipeline.py:175` |
| `paso_frames_secundarios` por intento de reintento | `5, 9, 13` | `pipeline.py:1557` |
| `_build_detection_retry_plan`: nº de intentos | `3` (1 inicial + `max_retries=2`) | `pipeline.py:1546-1568` |
| `MAX_SINGLE_PASS_FRAMES` | `10` | `video_io.py:365` |
| `max_workers` (extracción paralela ffmpeg) | `4` | `video_io.py:385` |
| Tamaño de ventana de frames secundarios | `3` (puede ser menor por recorte a límites) | `video_io.py:804-809` |
| Timeout `ffprobe` | `15` s | `video_io.py:75` |
| Timeout `ffmpeg` single-frame | `25` s (default) | `video_io.py:159,171` |
| Timeout `ffmpeg` batch | `120` s (default) | `video_io.py:257` |
| Timeout transcodificación de compatibilidad | `900` s | `video_io.py:504` |
| Timeout `convert_detected_projection_to_equirectangular` | `3600` s | `projection_conversion.py:451` |
| Orden de encoders hardware | `h264_nvenc, h264_qsv, h264_videotoolbox, h264_amf, libx264` | `projection_conversion.py:108` |
| Preset/CRF libx264 (conversión v360) | `veryfast` / `18` | `projection_conversion.py:413` |
| Preset/CRF libx264 (transcodificación compat) | `veryfast` / `23` | `video_io.py:493` |
| Bitrate AAC en conversión con audio | `192k` | `projection_conversion.py:425` |
| `_AV1_CODEC_NAMES` | `{"av1","libaom-av1","libdav1d"}` | `video_io.py:16` |
| Timeouts HTTP MediaCMS (`_REQUEST_TIMEOUT`) | conexión `10` s / lectura `300` s | `core/uploader.py:29-31` |
| Reintentos yt-dlp | `retries=20`, `fragment_retries=20`, `socket_timeout=60` | `core/downloader.py:139-141` |
| `DOWNLOAD_PROGRESS_UPDATE_MS` (UI) | `120` ms | `app/gui/progress_utils.py:9` |
| División en regiones de movimiento | `2 filas × 3 columnas = 6` | `motion_analysis.py:231-246` |
| Ángulo de corrección EAC: BOTTOM / BACK / TOP | `−π/2` / `+π/2` / `+π/2` | `projection_logic.py:13-15` |
| Divergencia angular ±90° (cubemap, LEFT/RIGHT vs FRONT) | `π/2` rad | `projection_logic.py:73,77,84` |
| Divergencia angular ±180° (BACK vs FRONT, ambas hipótesis) | `π` rad | `projection_logic.py:52,61,91,99` |

---

## 14. Conversión a equirectangular con ffmpeg

Fichero: `detector/projection_conversion.py` (827 líneas).

### 14.1 Identificadores `v360` exactos por tipo de proyección detectado

```python
_V360_INPUT_FORMAT = {
    "eac":   "eac",    # equi-angular cubemap — estándar YouTube 360
    "cubic": "c3x2",   # layout 3 columnas × 2 filas
}
```
(`projection_conversion.py:63-70`). El filtro final aplicado es siempre:
```
v360=<input>:equirect,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2
```
(`build_v360_filter_for_projection`, línea 308-336) — el `pad` adicional garantiza dimensiones pares, requeridas por `yuv420p`/`libx264`.

**Nota explícita de la propia documentación del módulo** (línea 26-29, verificada correcta): el identificador ffmpeg `e` corresponde a **entrada equirectangular**, no a EAC — usarlo por error produciría una salida silenciosamente corrupta. El código usa correctamente `"eac"`, nunca `"e"`.

### 14.2 Qué tipos se convierten y cuáles se omiten

| `projection_type` | Acción | Razón devuelta |
|---|---|---|
| `equirectangular` | Omitido | `already_equirectangular` |
| `stereo_equi` | Omitido | `already_equirectangular_stereo_layout` (la geometría base ya es equirectangular; **no** se aplana estéreo→mono, ver §17.2) |
| `eac` | Convertido | `converted_to_equirectangular` |
| `cubic` | Convertido | `converted_to_equirectangular` |
| `unknown` | Omitido (a este nivel; ver §12.3 para el fallback de la capa de workflow) | `projection_unknown` |
| Cualquier otro string | Omitido | `unsupported_projection_type:<valor>` |

### 14.3 Manejo de audio

1. Intento inicial: `-map 0:a? -c:a aac -b:a 192k` (audio opcional, se omite sin error si no existe stream de audio) (línea 419-426).
2. Si el `stderr` de ffmpeg contiene señales de fallo de audio (`"ambisonic"`, `"unsupported channel layout"`, o `"channel layout"` combinado con contexto de audio — `_is_audio_failure`, línea 275-305) y **no** contiene señales de fallo de vídeo (dimensiones impares, error de filtro, `libx264`, `v360` — estas tienen prioridad y descartan la hipótesis de audio), se reintenta automáticamente con `-an` (sin audio) (línea 757-785).
3. Fallback adicional de encoder: si el encoder hardware seleccionado (`detect_ffmpeg_h264_encoder`, orden de preferencia en §13.2) falla en tiempo de ejecución (CUDA no disponible, etc. — `_is_hardware_encoder_runtime_failure`, línea 183-194), se reintenta automáticamente con `libx264` (línea 720-750).
4. Tras un fallo final, si el fichero de salida quedó vacío (ffmpeg trunca el fichero al abrirlo antes de empezar a codificar), se elimina automáticamente (línea 806-818, con `os.stat()` único para evitar una condición de carrera TOCTOU entre comprobar existencia y tamaño).

### 14.4 Camino de *stream-copy* (sin transformación geométrica)

Cuando no se necesita filtro `v360` (tipos en el conjunto de *skip*), la función `build_ffmpeg_command_for_projection` no se invoca normalmente desde el flujo de conversión pública (los tipos *skip* nunca llegan a construir comando), pero la propia función soporta este camino para otros llamadores: usa `-hwaccel auto -c:v copy -c:a copy` (línea 427-439) — la aceleración hardware es segura aquí porque no hay filtro de software que requiera frames en memoria de CPU. Cuando sí hay filtro `v360`, `-hwaccel auto` se omite deliberadamente (comentario explícito en línea 354-361): los frames decodificados por hardware viven en memoria de dispositivo y el filtro `v360` (software) no puede leerlos, lo que produciría una salida vacía.

---

## 15. Módulos de soporte

### 15.1 `core/downloader.py`

`download_video(url, output_dir=None, progress_callback=None, format_spec="bestvideo+bestaudio/best")` (línea 88-165): envuelve `yt_dlp.YoutubeDL`. Resolución robusta de la ruta de salida final (`_resolve_downloaded_output_path`, línea 58-70) probando, en orden: `requested_downloads[].filepath/filename/_filename`, `info.filepath/filename/_filename`, `ydl.prepare_filename(info)`, variante forzada a `.mp4`. Opciones yt-dlp fijas: `merge_output_format="mp4"`, `retries=20`, `fragment_retries=20`, `socket_timeout=60` (línea 135-143). Excepciones traducidas a `utils.exceptions.DownloadError`.

### 15.2 `core/youtube.py`

`search_videos(query, api_key=None, youtube_client=None, max_results=10)` (línea 65-183): si `query` contiene `youtube.com`/`youtu.be` se trata como URL directa (vía `extract_video_id`, regex `(?:v=|/|embed/|youtu\.be/)([0-9A-Za-z_-]{11})`, línea 54); si no, hace `search().list(q=f"{query} 360", type="video")` seguido de `videos().list(part="snippet,contentDetails")`. **Filtra estrictamente** por `contentDetails.projection == "360"` (línea 151) — vídeos no marcados como 360° por la API de YouTube se excluyen aunque el texto de búsqueda los devuelva. Errores HTTP con `quotaExceeded`/`dailyLimitExceeded` se traducen a `YouTubeQuotaError`.

`get_video_thumbnail_urls(video_id)` (línea 186-202): construye URLs estáticas `https://img.youtube.com/vi/<id>/<calidad>.jpg` sin llamar a la API (no consume cuota).

### 15.3 `core/uploader.py`

`upload_video_asset(video_path, title, description="", api_url=None, playlist_id=None, new_playlist_name=None, tags=None)` (línea 192-286): `multipart/form-data` (`files={"media_file": (...)}`) vía `requests.post`. `_build_endpoint` (línea 61-79) deriva endpoints relacionados (`/playlists`) desde la URL base de `/media` reemplazando el último segmento de ruta. Los tags se normalizan (`_normalise_tags`, línea 99-112: recorta espacios, elimina duplicados y vacíos, fuerza a `str`) y se envían como CSV en el campo `tags`. **No existe soporte de categoría** en la firma actual (confirmado: ningún parámetro `category` en `upload_video_asset` ni en `JobOptions`) — fue retirado explícitamente (commit `5966d0b Remove patient category upload support`, ver `git log`).

### 15.4 `core/preview_frames.py`

`extract_preview_frames(video_path, num_frames=5, output_dir=None, prefix="preview", padding_ratio=0.05)` (línea 24-114): extracción **independiente** de la del detector — usa directamente `cv2.VideoCapture` sin fallback a `ffmpeg`. Si OpenCV no puede decodificar el vídeo, esta función simplemente devuelve una lista vacía (no lanza excepción, línea 62-64), por lo que un fallo aquí nunca interrumpe el resto del pipeline. JPEG con calidad `85` (línea 100).

### 15.5 `core/models.py` / `core/job_manifest.py`

`JobResult` (dataclass, `core/models.py:79-148`) es el contrato canónico devuelto por `process_video_job`. `save_job_manifest` (`core/job_manifest.py:27-56`) serializa a `data/jobs/job_<job_id>.json` con `schema_version: "1.0"`, usando `default=str` en `json.dump` para tolerar tipos no serializables. El `job_id` es un timestamp UTC con microsegundos (`_make_job_id`, línea 22-24: `%Y%m%d_%H%M%S_%f`).

### 15.6 `config/settings.py`

Singleton `get_settings()` (línea 280-285) construye `Settings()` una sola vez por proceso. `_load_dotenv` (línea 68-74) intenta `python-dotenv`; si falla con cualquier excepción (no solo `ImportError`), recurre silenciosamente a `_parse_dotenv` (parser manual línea-a-línea sin soporte de comillas anidadas ni interpolación). `as_detector_config()` expone un `dict` plano consumido por `detector/pipeline.py::load_config()` **una sola vez al importar el módulo** (`CONFIG = load_config()`, `pipeline.py:55`) — cambios posteriores de entorno en el mismo proceso no se reflejan en `CONFIG`, aunque sí en `_resolve_motion_feature_flags()`, que llama a `get_settings()` de nuevo en cada invocación.

### 15.7 `app/gui/gui_app.py`

GUI de un único fichero (893 líneas), sin lógica de pipeline propia: cada acción de usuario (`_on_search`, `_on_download_process`, `_on_upload`) lanza un `threading.Thread` daemon que invoca funciones de `workflows.unified_pipeline` / `core.*`, y devuelve resultados al hilo principal vía `master.after(0, ...)`. Estado de UI modelado como máquina de estados finita (`AppState`: `IDLE, SEARCHING, READY, PROCESSING, PROCESSED, UPLOADING`, línea 54-71) que habilita/deshabilita botones según una tabla fija (`_STATE_BUTTONS`).

---

## 16. Tests

Ejecución verificada en este análisis: `bin/python -m pytest tests/ -q` → **118 passed** (sin fallos, sin errores), reproduciendo exactamente el recuento documentado en `BASELINE_TEST_RESULTS.md` (118 passed, capturado antes de las fases de limpieza de código muerto `Phase 0`–`2c`). Esto confirma que dichas fases no rompieron ningún test existente.

| Fichero | Qué verifica (resumen, basado en lectura directa) |
|---|---|
| `test_detection_retry.py` | Plan de reintentos (`_build_detection_retry_plan`), resolución de flags de movimiento por perfil (`_resolve_motion_feature_flags`) incluida la prioridad Tier B sobre Tier C, gate de ratio del 50 % en `run_detection_pipeline`, normalización de codec condicional en `process_video_job` |
| `test_fallback_and_adaptive.py` | Fallback `unknown→eac` solo en la etapa de conversión; `equirectangular`/`stereo_equi` siguen omitiéndose; fórmula de `wraplength` adaptativo de la GUI |
| `test_line_and_stereo_strictness.py` | Rechazo de candidatos fragmentados de alta cobertura; rechazo por "no estrictamente centrado"; política de estéreo basada en región de costura vs. división central fija; mínimo de frames de costura |
| `test_line_detection.py` | Casos base de Hough; rechazo por fragmento corto; rechazo por pendiente diagonal; rechazo por descentrado; **regresión del bypass de FFT eliminado** (Fix 3.1); `fft_min_dominance` configurable; gate de perfil espacial desactivado por defecto / activado en horizonte vs costura real (horizontal y vertical) |
| `test_motion_classification_policy.py` | Forzado de clasificación por evidencia consistente (§11.6); mantenimiento de `unknown` ante evidencia contradictoria |
| `test_motion_flow_fallback.py` | Cadena de fallback de flujo óptico en perfil `robust` (con/sin disponibilidad) y `baseline` |
| `test_progress_and_downloader.py` | Parseo de progreso yt-dlp (bytes y `%` de texto), *throttling* de actualizaciones, resolución de ruta de salida del downloader |
| `test_projection_conversion.py` | Filtro `v360` con padding par; ausencia de `-hwaccel` en camino con filtro; presencia de `-hwaccel` en *stream-copy*; detección de fallo de encoder hardware vs. fallo de dimensión de vídeo; reintento automático con `libx264`; limpieza de fichero de salida vacío tras fallo |
| `test_uploader.py` | Construcción de endpoints (`_build_endpoint`) con distintas formas de URL base; normalización de tags; flujo de creación de playlist + subida |
| `test_video_io_av1.py` | Omisión de `-hwaccel` para codecs AV1 en transcodificación; retención de `-hwaccel` para H.264; ruteo a extracción paralela por encima de `MAX_SINGLE_PASS_FRAMES`; fallback de paralelo a `select=` |
| `test_video_io_sampling.py` | Fallback de muestreo a ffmpeg cuando OpenCV no decodifica; caché de `probe_video_stream` por mtime; sesión de captura compartida (una sola apertura de `cv2.VideoCapture`); diagnóstico enriquecido en fallo duro |
| `test_youtube.py` | Deduplicación de resultados de búsqueda preservando orden; filtro estricto de proyección 360°; extracción de ID de vídeo desde distintas formas de URL |

**Cobertura ausente confirmada** (sin fichero de test correspondiente, por inspección directa del directorio `tests/`): `core/job_manifest.py`, `core/preview_frames.py`, `detector/equirectangular_detection.py`, `detector/projection_logic.py`, `detector/preprocessing.py`, `detector/region_validation.py`, `detector/debug_utils.py`, y la mayoría de `workflows/unified_pipeline.py` (solo se cubren fragmentos vía mocks en `test_detection_retry.py`).

---

## 17. Known issues, caveats y suposiciones

### 17.1 Suposición de *layout* cubemap 3×2

`split_into_regions` (`motion_analysis.py:231-246`) divide siempre en una rejilla fija de 2 filas × 3 columnas, y `CUBEMAP_LAYOUT`/`EAC_LAYOUT` (`projection_logic.py:9-24`) asumen esa misma disposición de 6 caras. La conversión ffmpeg para `cubic` usa el identificador `c3x2` (`projection_conversion.py:67-69`). **Ningún otro layout de cubemap (tira 6×1, layout en cruz, etc.) es reconocido ni por la detección ni por la conversión.** Documentado también en `README.md` ("Known issues / caveats → Cubemap layout assumption").

### 17.2 `stereo_equi → mono` no implementado

La conversión actual **omite** deliberadamente cualquier transformación geométrica para `stereo_equi` (`projection_conversion.py:629-644`, razón `already_equirectangular_stereo_layout`) porque la geometría base ya es equirectangular — pero **no aplana el par estéreo a una salida mono** (ni recorte de un ojo, ni reproyección estéreo→mono). Esto es una decisión de alcance explícita, no un bug: documentada en el docstring del módulo (línea 40-45), en `README.md` y como "Needs Decision" en `IMPROVEMENT_PLAN.md`. **No hay ningún `NotImplementedError` en el código actual para este camino** — simplemente se toma el camino de *skip*, sin lanzar excepción (la referencia a un `NotImplementedError` en `IMPROVEMENT_PLAN.md` §"Needs Decision" no corresponde al estado actual del código; ver discrepancia §18.5).

### 17.3 Fuga de `VideoCapture` — corregida, verificada en el código actual

`extract_main_frames` (`video_io.py:544-781`) usa `try/finally` con liberación de `_CachedCapture` garantizada en el bloque `finally` (línea 778-780) **solo cuando la función es dueña de la captura** (`_owns_cap = cap_session is None`). Confirmado correcto en el HEAD actual — coincide con `IMPROVEMENT_PLAN.md` ("2.3 VideoCapture resource leak — ✅ Already applied").

### 17.4 Caché de `probe_video_stream` sin límite

`_PROBE_CACHE` (`video_io.py:37`) es un `dict` a nivel de módulo, indexado por `(ruta_absoluta, mtime)`, sin política de expulsión (TTL ni LRU). En un proceso de larga duración que procese muchos vídeos distintos, este caché crece de forma no acotada durante la vida del proceso. `clear_probe_cache()` (línea 49-51) existe pero solo se invoca explícitamente desde los tests, nunca desde código de producción.

### 17.5 Fuga real de ficheros temporales de transcodificación — observada en este análisis

`convert_video_codec` (`video_io.py:453-541`) crea el fichero temporal de compatibilidad con `tempfile.NamedTemporaryFile(..., dir=input_dir, delete=False)` (línea 470-476) **en el mismo directorio que el vídeo de entrada** — por diseño, ya que ese fichero es el resultado útil que se devuelve al llamador. En la raíz de este repositorio se observan actualmente **76 ficheros `fake_*_compat_*.mp4` de 0 bytes**, fechados en distintas ejecuciones (16–17 de junio), producidos por `tests/test_video_io_av1.py` al invocar `convert_video_codec("fake_av1.mkv")` etc. con `os.path.exists` mockeado a `True` pero **sin mockear `tempfile.NamedTemporaryFile`**: cada ejecución del test crea un fichero temporal real en la raíz del proyecto (porque `os.path.dirname(os.path.abspath("fake_av1.mkv"))` resuelve al directorio de trabajo actual). Estos ficheros están cubiertos por `.gitignore` (`*.mp4`) por lo que no contaminan `git status`, pero se acumulan en disco en cada ejecución de la suite de tests. Esto **no es un bug de producción** (en producción `video_path` es siempre una ruta real con directorio válido) sino un problema de higiene de tests no documentado previamente en `SCAN_REPORT.md`/`IMPROVEMENT_PLAN.md`.

### 17.6 `ColorizedFormatter` duplicado

`detector/debug_utils.py:18-36` redefine una clase idéntica en estructura a `config/logging_config.py:16-36`. Confirmado presente en el HEAD actual (no corregido pese a estar en el backlog de `SCAN_REPORT.md` §6.3).

### 17.7 `save_frame_debug` falla con ruta sin directorio

`detector/debug_utils.py:61-63`: `os.makedirs(os.path.dirname(filepath), exist_ok=True)` lanza `FileNotFoundError` si `filepath` no contiene componente de directorio (`os.path.dirname("foo.jpg") == ""`). No corregido en el HEAD actual.

### 17.8 `_load_dotenv` silencia excepciones no esperadas

`config/settings.py:71-74`: `except Exception: _parse_dotenv(...)` — captura cualquier excepción de `python-dotenv` (no solo `ImportError`) sin registrar ningún log, antes de recurrir al parser manual. Si `python-dotenv` está instalado pero falla por otra razón (p. ej. `.env` malformado de cierta manera), el fallo es completamente silencioso.

---

## 18. Discrepancias documentación vs código

> **⚠ DISCREPANCIA 18.1 — Paquete OpenCV instalado vs. declarado.** `requirements.txt`/`pyproject.toml` declaran `opencv-contrib-python>=4.13.0.92,<4.14`, y `README.md:108-110` advierte explícitamente *"this project uses opencv-contrib-python ... Do not install opencv-python in the same environment."* Sin embargo, el venv real de este repositorio (`/home/gorferna/360-Video-Manager`, `pip show opencv-python` → `opencv-python 4.13.0.92`, instalado en `lib/python3.12/site-packages/cv2`) tiene instalado **`opencv-python`** (sin *contrib*), confirmado por `pip show opencv-contrib-python` → *"Package(s) not found"*. Verificado en código: `cv2.optflow` no existe en ese build (`hasattr(cv2,'optflow') == False`), por lo que TV-L1, DeepFlow, PCAFlow, SparseToDense y el refinamiento variacional están **inactivos** en este entorno concreto, cayendo siempre a DIS (disponible en el módulo `video` principal) o Farneback. Esto no es un hallazgo sobre "el código" sino sobre el **estado actual de este entorno instalado**; en un entorno con `opencv-contrib-python` correctamente instalado, TV-L1 sí estaría disponible y sería el seleccionado bajo los perfiles por defecto (§8.4).

> **⚠ DISCREPANCIA 18.2 — `AGENTS.md` afirma un venv versionado que no existe en el repositorio git actual.** `AGENTS.md:35-36` dice: *"The repository tracks a committed virtual environment (bin/, lib/, share/, pyvenv.cfg). Agents must treat these as locked runtime artifacts."* Verificado: `git rev-parse --show-toplevel` apunta a `/home/gorferna/360-Video-Manager/360-Video-Manager` (el subdirectorio interno); el venv (`bin/`, `lib/`, `share/`, `pyvenv.cfg`) vive en `/home/gorferna/360-Video-Manager/` — **un nivel por encima del repositorio git**, y `git ls-files | grep -E '^(bin/|lib/|share/|pyvenv.cfg)'` no devuelve ninguna coincidencia. El propio `.gitignore` del repositorio (líneas `bin/`, `lib/`, `share/`, `include/`, `pyvenv.cfg`) confirma que estos directorios están explícitamente excluidos de versionado, no incluidos. La afirmación de `AGENTS.md` no se corresponde con el estado real del repositorio en este checkout.

> **⚠ DISCREPANCIA 18.3 — `SCAN_REPORT.md` e `IMPROVEMENT_PLAN.md` están desactualizados respecto al HEAD actual.** Ambos documentos (fechados antes de los commits `d749820`…`ef66e8a`, *"Phase 0"* a *"Phase 2c"* de limpieza de código muerto) listan como pendientes: `process_downloaded_video` (función legacy), y los alias `descargar_video`, `buscar_youtube`, `extraer_video_id`, `obtener_thumbnail_video`, `post_video`. Verificado por lectura directa de `detector/pipeline.py`, `core/downloader.py`, `core/youtube.py`, `core/uploader.py`: **ninguno de estos nombres existe ya en el código actual** — fueron eliminados por los commits referenciados (confirmado por `git log --oneline`). Sí permanecen sin eliminar los alias de excepción `DownloadErrorCustom` (`utils/exceptions.py:48`) e `InvalidYouTubeURLException` (`utils/exceptions.py:28`), que `SCAN_REPORT.md` también lista — estos **no** fueron tocados por las fases de limpieza y siguen presentes.

> **⚠ DISCREPANCIA 18.4 — Docstring de `convert_detected_projection_to_equirectangular` corregido respecto a lo que documenta `SCAN_REPORT.md`.** `SCAN_REPORT.md` §3c-2 señala que el docstring documentaba erróneamente `v360=e:equirect` y `v360=c6x1:equirect`. Verificado en el código actual (`projection_conversion.py:1-46, 576-618`): el docstring ya usa correctamente `eac` y `c3x2`, coincidiendo con `_V360_INPUT_FORMAT`. La corrección descrita como aplicada en `IMPROVEMENT_PLAN.md`/`SCAN_REPORT.md` §5 ("Fix 3") está efectivamente presente.

> **⚠ DISCREPANCIA 18.5 — `NotImplementedError` para `stereo_equi → mono` no existe en el código actual.** `IMPROVEMENT_PLAN.md:69` afirma: *"`convert_detected_projection_to_equirectangular` ... has a code path for `stereo_equi` input that raises `NotImplementedError`."* Verificado por lectura completa de `detector/projection_conversion.py`: no existe ninguna instancia de `NotImplementedError` en el fichero; el camino para `stereo_equi` es un *skip* silencioso (línea 629-644, ver §17.2). La descripción del plan de mejora no refleja el comportamiento real (puede describir un estado de un commit anterior al disponible en este checkout, o ser inexacta).

> **Nota (no discrepancia, confirmación positiva)**: las correcciones aplicadas y documentadas en `IMPROVEMENT_PLAN.md` (fuga de `VideoCapture`, frame incorrecto en visualización de depuración `secuencia[j]` vs `secuencia[i+1]`, `threading.Lock` para el snapshot de capacidades, extracción paralela por encima de `MAX_SINGLE_PASS_FRAMES`) están todas verificadas presentes en el código actual mediante lectura directa — el documento es preciso en estos puntos.

---

## 19. AFIRMACIONES QUE PODRÍAN SER IMPRECISAS EN UNA PRESENTACIÓN

Lista de afirmaciones que **suenan plausibles** dado el dominio (detección de proyecciones 360°) pero que **no están respaldadas literalmente por el código**, o que requieren matización importante:

1. **"El sistema usa Lucas-Kanade / flujo óptico disperso."** — Falso. No hay ninguna llamada a `cv2.calcOpticalFlowPyrLK` ni a `goodFeaturesToTrack` en todo el repositorio. Todo el flujo óptico es **denso** (Farneback, DIS, y opcionalmente TV-L1/DeepFlow/PCAFlow/SparseToDense vía `cv2.optflow`, ver §8.1–8.2).

2. **"El algoritmo de flujo óptico por defecto es DeepFlow."** — Técnicamente `VPD_FLOW_ALGORITHM` por defecto vale `"deepflow"` (`config/settings.py:193`), pero bajo el perfil por defecto (`high_accuracy`) ese valor **casi nunca es el que se ejecuta**: la lógica de `pipeline.py:120-138` prioriza Tier B (`tvl1`, `dis`) por delante de Tier C (`deepflow`), de modo que el algoritmo real depende de qué backends estén disponibles en el build de OpenCV (§8.4). Afirmar "el sistema usa DeepFlow por defecto" sin la matización del perfil sería impreciso.

3. **"EAC y cubemap se distinguen comprobando una divergencia angular de ±90° entre todas las caras."** — Parcialmente impreciso. La comprobación literal de `π/2` (90°) **solo existe en el scorer cúbico** (`_score_layout_cubemap`, LEFT/RIGHT vs FRONT). El scorer EAC usa un criterio de **simetría relativa** entre LEFT y RIGHT, no un ángulo absoluto fijo (§9.2–9.3).

4. **"La tolerancia angular usada en el scoring es de 45°."** — Falso por el nombre del parámetro (`tolerancia_45_deg`), pero el valor real por defecto en todo el código de producción es **20.0°** (§9.4). No existe ninguna ruta donde se use literalmente 45°.

5. **"El umbral mínimo de frames analizados es siempre 7."** — Impreciso. `7` es solo el valor por defecto de un parámetro de función (`run_detection_pipeline`) que en la ruta real de producción (`run_detection_with_retries`) **siempre se recalcula** como `max(2, int(num_frames_intento · 0.55))` (§11.7).

6. **"El sistema garantiza conversión automática de cualquier proyección desconocida a EAC."** — Cierto solo dentro de `workflows/unified_pipeline.py`. La función de bajo nivel `detector.projection_conversion.convert_detected_projection_to_equirectangular` (y el wrapper `detector.convert_to_equirectangular`) trata `unknown` como un *skip*, no como un fallback a `eac` (§12.3). Presentar el fallback como una propiedad universal del "conversor" sería impreciso.

7. **"La detección usa Machine Learning / redes neuronales."** — Falso. Confirmado por inspección de imports en todo `detector/`: no hay ninguna dependencia de frameworks de ML (no hay `torch`, `tensorflow`, `onnx`, etc., ni en `requirements.txt` ni en el código). El propio README lo afirma ("deterministic projection-detection engine (no ML models required)") y es coherente con el código leído.

8. **"El sistema soporta layouts de cubemap distintos al 3×2 (p. ej. tira 6×1 o cruz)."** — Falso. La rejilla de regiones (`motion_analysis.py:231-246`) y el identificador `c3x2` de conversión (`projection_conversion.py:67-69`) son fijos; no hay parametrización de layout alternativo en ningún punto del código (§17.1).

9. **"`stereo_equi` se convierte a una salida mono equirectangular."** — Falso en el estado actual. Se omite explícitamente sin transformación geométrica (§17.2).

10. **"El umbral de correlación de histograma para estéreo (0.92) es el único criterio de decisión."** — Incompleto. La decisión real exige simultáneamente correlación ≥ umbral, distancia de Bhattacharyya ≤ 0.30 (hardcoded, no configurable) y similitud de bordes ≥ umbral, además de un mínimo de frames evaluados y una estabilidad temporal mínima (§7.2–7.3). Citar solo el umbral de correlación de 0.92 simplifica en exceso una decisión de cuatro condiciones.

11. **"El ángulo de movimiento de cada región proviene del flujo óptico."** — Incompleto. Cuando hay suficientes coincidencias ORB válidas, el ángulo usado para el scoring EAC/cúbico se **sobrescribe** por una estimación afín ORB+RANSAC independiente del flujo óptico denso (§8.7); el flujo óptico solo decide el ángulo cuando ORB no produce suficientes *inliers*.

12. **"Todas las características de robustez (refinamiento variacional, consistencia forward-backward, evidencia geométrica) están desactivadas por defecto, como indican sus variables de entorno (`...=false`)."** — Falso en la práctica. Aunque sus variables de entorno individuales valen `false` por defecto, el perfil por defecto (`VPD_MOTION_ROLLOUT_PROFILE=high_accuracy`) las **fuerza a `true`** incondicionalmente (§8.5). Solo con `VPD_MOTION_ROLLOUT_PROFILE=baseline` explícito quedan realmente desactivadas.

