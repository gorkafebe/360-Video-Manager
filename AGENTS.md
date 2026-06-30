# AGENTS.md

## Scope

This document governs autonomous agent behavior in this repository only.

## Mandatory pre-action inspection

Before proposing or making any change, agents MUST inspect:

- `README.md`
- `config/settings.py`
- `workflows/unified_pipeline.py`
- `detector/pipeline.py`
- `detector/projection_conversion.py`
- `core/models.py`
- Relevant tests in `tests/` for the touched area

If inspection is incomplete, stop and refuse to proceed.

## Repository reality (do not assume beyond this)

- Language: Python (`app/`, `config/`, `core/`, `detector/`, `utils/`, `workflows/`, `tests/`)
- UI: `customtkinter` (`app/gui/gui_app.py`)
- Video/vision stack: OpenCV + NumPy (`detector/*`, `core/preview_frames.py`)
- Download: `yt-dlp` (`core/downloader.py`)
- YouTube API: `google-api-python-client` (`core/youtube.py`)
- Upload integration: MediaCMS over HTTP (`core/uploader.py`)
- Conversion engine: `ffmpeg` v360 (`detector/projection_conversion.py`)
- Runtime configuration: environment + `.env` (`config/settings.py`)
- Persistent artifacts: `data/` runtime dirs + JSON manifests (`core/job_manifest.py`)

## Architectural boundaries

- GUI layer (`app/gui/gui_app.py`) orchestrates user interaction only; it delegates processing to `workflows.unified_pipeline`.
- Unified orchestration lives in `workflows/unified_pipeline.py`; this is the canonical stage order.
- Detector internals and classification policy live in `detector/`.
- Upload API behavior and endpoint construction live in `core/uploader.py`.
- Shared contracts are in dataclasses (`core/models.py`), especially `JobResult`, `DetectorStats`, `UploadResult`.

Do not bypass these boundaries with cross-layer shortcuts.

## Safety-critical behavioral contracts

### Detection and reliability

- Non-equirectangular classifications are reliability-gated in `detector/pipeline.py`.
- `unknown` is a valid terminal classification and must not be silently reinterpreted except where explicitly implemented.
- Retry strategy and low-motion handling in `run_detection_with_retries` are deliberate; preserve semantics.

### Conversion

- Conversion mapping is explicit: `eac -> v360=eac:equirect`, `cubic -> v360=c3x2:equirect`.
- `equirectangular` and `stereo_equi` are skip paths.
- In unified workflow stage conversion, `unknown` currently falls back to `eac` for conversion attempts.
- Audio-failure retry behavior (fallback to `-an`) is intentional.

### Upload and metadata

- Upload supports category and tags; tags are normalized and comma-joined.
- Playlist/category endpoints are host-rooted via `_build_endpoint`; keep URL construction behavior consistent.

## Security and secrets handling

- Secrets are provided through env vars (`YOUTUBE_API_KEY`, `CMS_*`) and must never be hardcoded.
- Do not log raw credentials or token values.
- Do not commit generated media/runtime data under `data/`.

## Validation rules

- Baseline test command: `python -m pytest tests/ -v`   # run from within the activated .venv
- In Linux environments, test collection may fail without `tkinter` (required by GUI imports in tests).
- No GitHub Actions workflow files are present under `.github/workflows/`; do not assume CI definitions not in repo.

## Change control policy for agents

1. Read relevant modules + tests first.
2. Make smallest complete change.
3. Preserve existing public function contracts unless explicitly requested to change them.
4. If behavior is unclear, add a hard restriction and stop instead of guessing.
5. Update README when architecture/governance artifacts change.

