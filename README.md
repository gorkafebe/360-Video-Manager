# 360-Video-Manager

Download, detect, convert, and upload 360° videos — via GUI.

360-Video-Manager is a Python application that combines a deterministic
projection-detection engine (no ML models required) with a complete
download → process → upload workflow for 360° video content.

---

## Table of Contents

- [Architecture](#architecture)
- [Agent governance artifacts](#agent-governance-artifacts)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage — GUI](#usage--gui)
- [Pipeline stages](#pipeline-stages)
- [Projection types](#projection-types)
- [Output artifacts](#output-artifacts)
- [Environment variables](#environment-variables)
- [Testing](#testing)
- [Known issues / caveats](#known-issues--caveats)

---

## Architecture

```
app/
  main.py               Entry point: launches the GUI
  gui/
    gui_app.py          CustomTkinter GUI (no pipeline logic)
    progress_utils.py   Download progress parsing and rate-limiting helpers
config/
  settings.py           All configuration loaded from env / .env
  logging_config.py     Console and file log handlers
core/
  downloader.py         yt-dlp wrapper
  youtube.py            YouTube Data API search and thumbnail helpers
  uploader.py           MediaCMS HTTP upload client
  preview_frames.py     JPEG preview-frame extraction (UI thumbnails)
  models.py             JobResult, DetectorStats, UploadResult dataclasses
  job_manifest.py       JSON manifest persistence (data/jobs/)
detector/
  pipeline.py           Main detection orchestrator
  video_io.py           OpenCV frame extraction + ffmpeg codec-normalisation fallback
  line_detection.py     Horizontal / vertical seam detection
  stereo_detection.py   Histogram-based stereo-equirectangular detection
  motion_analysis.py    Optical-flow backends + region and geometry evidence scoring
  projection_logic.py   EAC vs cubemap hypothesis scoring
  equirectangular_detection.py  Wrap-around boundary-continuity evidence
  projection_conversion.py      ffmpeg v360 conversion
  preprocessing.py      Frame pre-processing helpers
  region_validation.py  Region filtering
  debug_utils.py        All debug-image and log I/O
workflows/
  unified_pipeline.py   Orchestration: download → normalise → detect → convert → upload
utils/
  exceptions.py         Exception hierarchy
  paths.py              Path helpers
tests/
  test_progress_and_downloader.py
  test_fallback_and_adaptive.py
```

## Prerequisites

- Python 3.9+
- **ffmpeg** on `PATH` — required for codec normalisation and equirectangular
  conversion. Install via your system package manager:
  ```bash
  sudo apt install ffmpeg          # Debian / Ubuntu
  sudo dnf install ffmpeg          # Fedora / RHEL
  brew install ffmpeg              # macOS (Homebrew)
  ```
- **tkinter** — not bundled by default on some Linux Python builds; install the system
  package before running the GUI:
  ```bash
  sudo apt install python3-tk      # Debian / Ubuntu
  sudo dnf install python3-tkinter # Fedora / RHEL
  ```

---

## Setup

```bash
git clone <repo>
cd 360-Video-Manager

# Create and activate a local virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
```

Or use the one-command helper:

```bash
./scripts/setup.sh
source .venv/bin/activate
```

> OpenCV dependency note: this project uses `opencv-contrib-python` (which
> already includes core OpenCV modules). Do not install `opencv-python` in the
> same environment.

All subsequent commands must be run from the project root with the virtualenv
active so that `app`, `config`, `core`, `detector`, `workflows`, and `utils`
are importable as top-level packages.

---

## Configuration

Copy `.env.example` to `.env` in the project root (or export the variables directly):

```bash
cp .env.example .env
```

Settings are loaded by `config/settings.py` at startup via `python-dotenv`
(with a built-in fallback parser if `python-dotenv` is unavailable).

```dotenv
# ── Required for YouTube search ──────────────────────────────────────────────
YOUTUBE_API_KEY=your_api_key_here

# ── Required for MediaCMS upload ─────────────────────────────────────────────
CMS_API_URL=https://your-cms.example.com/api/v1/media
CMS_USER=admin
CMS_PASSWORD=secret
CMS_TOKEN=csrf_token_here

# ── Optional directory overrides ─────────────────────────────────────────────
DOWNLOADS_DIR=data/downloads
VPD_FRAMES_OUTPUT_DIR=data/frames
VPD_DEBUG_OUTPUT_DIR=data/frames
```

All variables are optional for detection-only workflows; YouTube search and
MediaCMS upload will fail gracefully with a clear error when their credentials
are missing.

---

## Usage — GUI

```bash
python -m app.main
```

The GUI launches a CustomTkinter window. Workflow:

1. **Search** — type a YouTube URL or search query and press Enter or click
   Search. Results appear as scrollable cards.
2. **Select** — click a result card to populate the detail panel with the
   title, channel, and URL.
3. **Download & Process** — starts the unified pipeline in a background thread:
   download → codec normalise → detect projection → convert if needed.
   Progress and status are shown in the always-visible bottom bar.
4. **Upload to CMS** — becomes available after a successful Download & Process.
   Assign a title, optionally choose or create a **patient category** (selected
   by the psychologist), optionally add personalized tags, and choose an
   existing MediaCMS playlist or create a new one before uploading.
5. **Log panel** — toggled with the "Show log" button; displays structured
   pipeline log output in real time.

The Download & Process and Upload buttons are always visible regardless of
window size (they are anchored to a fixed bottom frame). The results, detail
panel, and upload options are in a scrollable content area above.

---

## Pipeline stages

```
source URL / local file
    │
    ├─ Stage 1: YouTube search / URL resolution   (skipped for local-file jobs)
    │
    ├─ Stage 2: yt-dlp download                   → data/downloads/
    │           (skipped for local-file jobs)
    │
    ├─ Stage 3: Codec compatibility telemetry     ffprobe metadata + decode strategy log
    │           (no full-file transcode by default; legacy full pass is feature-flagged)
    │
    ├─ Stage 4: Preview frame extraction          → data/downloads/previews/
    │           (core/preview_frames.py)           JPEG, evenly-spaced, skips head/tail
    │
    ├─ Stage 5: Projection detection              (detector/pipeline.py)
    │           Line detection → stereo check → optical-flow scoring
    │           Returns projection_type + confidence
    │
    ├─ Stage 6: Equirectangular conversion        (detector/projection_conversion.py)
    │           Only when convert_if_needed=True and confidence ≥ threshold
    │           unknown → falls back to eac (see fallback note below)
    │
    ├─ Stage 7: MediaCMS upload                   (optional; core/uploader.py)
    │
    └─ Stage 8: JSON job manifest                 → data/jobs/job_<id>.json
                (optional; core/job_manifest.py)
```

### Projection detection

The detector (`detector/pipeline.py`) runs without ML models:

1. Black-frame filtering.
2. Horizontal / vertical seam detection — a central structural line is the
   primary cue for non-equirectangular content.
3. Stereo histogram matching — identifies stereo-equirectangular layouts.
4. Dense optical-flow analysis — Farneback or DIS; per-region directional
   aggregation; EAC vs cubemap hypothesis scoring.
5. Equirectangular wrap-around boundary-continuity evidence.
6. Reliability gate — if confidence is too low or too few valid frames were
   analysed, the detection result is `unknown`.

### Conversion decision

| Detected projection | Conversion action |
|---|---|
| `equirectangular` | Skipped — already target format |
| `stereo_equi` | Currently skipped — geometry is already equirectangular (see note below for expected behavior) |
| `eac` | Converted via `ffmpeg v360=eac:equirect` |
| `cubic` | Converted via `ffmpeg v360=c3x2:equirect` |
| `unknown` | **Falls back to `eac`** and conversion is attempted |

For `stereo_equi`, the expected outcome is conversion to mono
equirectangular output for both stereo packing variants (top-bottom and
left-right). The current implementation still takes the skip path for
`stereo_equi`.

The `unknown → eac` fallback is applied in `workflows/unified_pipeline.py`
`_stage_convert_to_equirectangular` with a `WARNING` log entry, so the
fallback is always traceable. The `JobResult.projection_type` field retains
the original detected value (`unknown`); only the conversion branch uses `eac`.

**Audio handling during conversion**

360° content commonly carries ambisonic audio with non-standard channel
layouts. The conversion layer:

1. Attempts conversion with audio re-encoded to AAC (`-c:a aac -b:a 192k`).
2. On an ambisonic / unsupported-layout ffmpeg error, retries automatically
   with audio dropped (`-an`). The resulting file has no audio track.
3. If a hardware H.264 encoder is listed but not usable at runtime
   (for example NVENC without CUDA runtime), conversion retries automatically
   with `libx264`.
4. A conversion failure never fails the overall job — detection results are
   always returned.

**Output profile for server compatibility**

When conversion runs (`eac`/`cubic`, or `unknown -> eac` fallback), output is
forced to a mono equirectangular profile compatible with high-quality server
variants (such as `2160_playlist`):

- Target resolution: `4320x2160` (2:1 equirectangular)
- Video encoder quality (`libx264` path): `-preset medium -crf 16`

Both resolution and quality are configurable through environment variables
(`VPD_CONVERSION_TARGET_WIDTH`, `VPD_CONVERSION_TARGET_HEIGHT`,
`VPD_CONVERSION_PRESET`, `VPD_CONVERSION_CRF`).

---

## Projection types

| Value | Description |
|---|---|
| `equirectangular` | Standard 2:1 equidistant cylindrical format |
| `stereo_equi` | Stereo pair in equirectangular layout (side-by-side or top-bottom) |
| `eac` | Equi-Angular Cubemap — the YouTube 360° VR standard |
| `cubic` | Conventional cubemap in a 3×2 face layout |
| `unknown` | Insufficient data or confidence below threshold |

---

## Output artifacts

| Artifact | Default location | Notes |
|---|---|---|
| Downloaded video | `data/downloads/` | Raw yt-dlp output |
| Codec-normalised video | `data/downloads/` | Only when OpenCV could not decode the original; filename gets a `_compat_` infix |
| Converted video | `data/downloads/` | `<stem>_equirectangular.mp4`; only produced when conversion ran |
| Preview frames | `data/downloads/previews/` | JPEG thumbnails for the GUI |
| Detector analysis frames | `data/frames/<video_stem>-<timestamp>/` | Per-run subdirectories: `main_frames/`, `secondary_frames/`, `motion_vectors/`, `line_detection/`, `logs/` |
| Job manifest | `data/jobs/job_<id>.json` | JSON record with all result fields; `schema_version: "1.0"` |

All `data/` subdirectories are created automatically by
`Settings.ensure_runtime_dirs()` on startup.

> `data/` is excluded from version control by `.gitignore`.

---

## Environment variables

All variables are read from the environment or a `.env` file at the project
root. Defaults shown are the values used when a variable is not set.

| Variable | Default | Description |
|---|---|---|
| `YOUTUBE_API_KEY` | — | YouTube Data API v3 key |
| `CMS_API_URL` | — | MediaCMS API endpoint (e.g. `https://cms.example.com/api/v1/media`) |
| `CMS_USER` | `$USER` | MediaCMS username for HTTP Basic Auth |
| `CMS_PASSWORD` | — | MediaCMS password |
| `CMS_TOKEN` | — | MediaCMS CSRF token |
| `VPD_PROJECT_ROOT` | auto-detected | Override the project root directory |
| `DOWNLOADS_DIR` | `data/downloads` | Download and conversion output directory |
| `VPD_FRAMES_OUTPUT_DIR` | `data/frames` | Detector analysis frame output |
| `VPD_DEBUG_OUTPUT_DIR` | same as `VPD_FRAMES_OUTPUT_DIR` | Debug visualisation output |
| `VPD_MIN_FRAMES_ANALYZED` | `4` | Minimum analysed frames required for a valid result |
| `VPD_MAX_DISCARD_RATIO` | `0.65` | Maximum acceptable frame-discard ratio |
| `VPD_MIN_LAYOUT_SCORE_MARGIN` | `0.10` | Minimum score margin between layout hypotheses |
| `VPD_LINE_CENTER_BAND_RATIO` | `0.08` | Legacy seam-band knob kept for compatibility (strict search now uses `VPD_LINE_CENTER_SEARCH_BAND_RATIO`) |
| `VPD_LINE_CENTER_SEARCH_BAND_RATIO` | `0.02` | Strict seam search band ratio (thin central ROI used for Hough/LSD detection) |
| `VPD_LINE_CENTER_MAX_DISTANCE_RATIO` | `0.02` | Max relative distance of seam from centre |
| `VPD_LINE_MAX_SLOPE` | `0.05` | Maximum accepted seam slope |
| `VPD_LINE_MIN_COVERAGE_RATIO` | `0.20` | Minimum seam coverage fraction across frame width |
| `VPD_LINE_MIN_QUALITY_SCORE` | `0.62` | Minimum seam structural quality score for non-fallback candidates |
| `VPD_LINE_FALLBACK_MIN_QUALITY_SCORE` | `0.78` | Minimum seam quality score for fallback (LSD/fitLine) candidates |
| `VPD_STEREO_HIST_THRESHOLD` | `0.92` | Histogram correlation threshold for stereo detection |
| `VPD_STEREO_SEAM_GUARD_RATIO` | `0.02` | Guard band excluded around detected seam before top/bottom (or left/right) comparison |
| `VPD_STEREO_MIN_VALID_HALF_RATIO` | `0.22` | Minimum usable side/half size ratio required after seam-guard cropping |
| `VPD_STEREO_EDGE_SIMILARITY_THRESHOLD` | `0.08` | Minimum edge-structure similarity required for a stereo frame match |
| `VPD_STEREO_MIN_SEAM_FRAMES` | `2` | Minimum seam-detected frames required before stereo classification runs |
| `VPD_STEREO_MIN_STABILITY_RATIO` | `0.55` | Minimum temporal match stability ratio for stereo decision |
| `VPD_SAVE_STEREO_HALVES` | `true` | Save left/right half-frame debug images |
| `VPD_FLOW_ALGORITHM` | `deepflow` | Preferred optical-flow algorithm hint; rollout profile and runtime capabilities can override to a higher-priority available algorithm |
| `VPD_FLOW_ENABLE_REFINEMENT` | `false` | Enable optional variational refinement on top of base optical flow |
| `VPD_FLOW_ENABLE_FB_CHECK` | `false` | Enable forward-backward optical-flow consistency filtering |
| `VPD_FLOW_FB_THRESHOLD` | `1.5` | Forward-backward consistency threshold in pixels |
| `VPD_ENABLE_GEOMETRY_EVIDENCE` | `false` | Enable geometry-evidence fusion from robust homography fitting |
| `VPD_GEOMETRY_EVIDENCE_WEIGHT` | `0.20` | Blend weight for geometry evidence in EAC-vs-cubic scoring |
| `VPD_MOTION_ROLLOUT_PROFILE` | `high_accuracy` | Motion profile: `baseline`, `robust`, `high_accuracy` |
| `VPD_FORCE_FULL_CODEC_NORMALIZATION` | `false` | Re-enable legacy full-file pre-normalization transcode before detection |
| `VPD_CONVERSION_TARGET_HEIGHT` | `2160` | Target converted equirectangular output height in pixels |
| `VPD_CONVERSION_TARGET_WIDTH` | `4320` | Target converted equirectangular output width in pixels |
| `VPD_CONVERSION_CRF` | `16` | libx264 CRF used for geometric conversion quality |
| `VPD_CONVERSION_PRESET` | `medium` | libx264 preset used for geometric conversion |

### Motion rollout profile behavior

- **Tier A (primary evidence/features)**: Canny+morphology+HoughLinesP, LSD, DFT orientation checks, ORB+BFMatcher, homography with RANSAC/USAC_MAGSAC, `estimateAffinePartial2D`, forward-backward consistency masking, variational refinement.
- **Tier B (high-value optional flow)**: `tvl1`, `dis`.
- **Tier C (fallback flow)**: `deepflow`, `pcaflow`, `sparse_to_dense`.

Runtime flow selection is capability-aware and deterministic:
- `baseline`: `farneback` safe default (or `dis` only when explicitly requested and available).
- `robust` / `high_accuracy`: prefer Tier B when available; Tier C is fallback; final fallback is always `farneback`.

Safety invariants:
- Non-equirectangular classification with failed reliability is downgraded to `unknown`.
- Contradictory motion evidence keeps `unknown` unless consistency overrides are met.
- Retry logic keeps early-stop behavior for stalled low-motion signals.

---

## Testing

```bash
# Run the full test suite (from project root, virtualenv active)
python -m pytest tests/ -v
```

Current test files:

| File | Coverage |
|---|---|
| `tests/test_progress_and_downloader.py` | Download-progress parsing, rate-limiting delay, yt-dlp output-path resolution |
| `tests/test_fallback_and_adaptive.py` | `unknown → eac` conversion fallback; `equirectangular`/`stereo_equi` skip behaviour; low-confidence skip; adaptive wraplength formula |
| `tests/test_uploader.py` | Playlist endpoint URL construction; robustness against trailing slashes and non-`/media` API paths |
| `tests/test_youtube.py` | Ordered de-duplication of search results; 360° projection filter; URL video ID extraction |

---

## Known issues / caveats

### tkinter system dependency

On Linux environments where tkinter is missing, importing `app.gui.gui_app`
will fail and tests that import GUI modules will fail at collection time.
Install `python3-tk` / `python3-tkinter` before running the GUI or full tests.

### Cubemap layout assumption

Conversion of `cubic` projection assumes the 3×2 face layout (`ffmpeg v360=c3x2:equirect`).
Videos using other cubemap arrangements (6×1 strip, cross layout, etc.) will
produce incorrect output. Update `_V360_INPUT_FORMAT["cubic"]` in
`detector/projection_conversion.py` if needed.

### Static-scene detection reliability

The motion-analysis branch requires sufficient camera or scene motion. Near-
static 360° content may produce low-confidence results and a final classification
of `unknown`, which then triggers the EAC fallback during conversion.

---

## Agent governance artifacts

This repository now includes agent-governance artifacts grounded in the current
implementation:

- `AGENTS.md` — repository-specific operational and safety constraints for autonomous agents.
- `skills/unified-pipeline-contract/SKILL.md` — stage-order and result-contract enforcement.
- `skills/projection-detection-reliability/SKILL.md` — detector reliability-policy enforcement.
- `skills/conversion-fallback-integrity/SKILL.md` — conversion mapping and fallback safety.
- `prompts/analyze-repository.md` — explicit read-only repository audit prompt.
- `prompts/audit-detection-policy.md` — explicit detector policy audit prompt.
- `prompts/plan-controlled-refactor.md` — explicit constrained-refactor planning prompt.
