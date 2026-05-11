# 360-Video-Manager

> Download, detect, convert, and upload 360° videos — end-to-end, in one command.

360-Video-Manager is a production-ready Python application that combines a robust
projection-detection engine (no ML models required) with a complete download-process-upload
workflow for 360° video content.

---

## Architecture

```
app/              — Entry points
  main.py         — python -m app.main  (GUI) or  --cli  (CLI mode)
  cli.py          — argparse CLI, delegates to workflows layer
  gui/
    gui_app.py    — Tkinter GUI, no business logic
config/           — Centralised settings + logging configuration
core/             — Shared services (YouTube search, download, upload, models, manifests)
detector/         — Projection detection engine (OpenCV-based, no ML)
workflows/
  unified_pipeline.py  — Orchestration: download → normalise → detect → convert → upload
utils/            — Path helpers and exception hierarchy
tests/            — Test suite
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- ffmpeg on `PATH` (required for codec normalisation and equirectangular conversion)

### Install

```bash
git clone <repo>
cd 360-Video-Manager
pip install -r requirements.txt
```

### Configure

Copy or create a `.env` file in the project root:

```dotenv
# YouTube Data API v3
YOUTUBE_API_KEY=your_api_key_here

# MediaCMS instance
CMS_API_URL=https://your-cms.example.com/api/v1/media
CMS_USER=admin
CMS_PASSWORD=secret
CMS_TOKEN=csrf_token_here

# Optional: override default data directories
DOWNLOADS_DIR=data/downloads
```

---

## Usage

### GUI (default)

```bash
python -m app.main
```

### CLI

```bash
# Download, detect, convert and upload from a YouTube URL
python -m app.main --cli --url "https://youtu.be/XXXXXXXXXXX" --upload

# Use a search query instead of a direct URL
python -m app.main --cli --url "360 drone ocean sunset" --upload --title "Ocean 360"

# Process a local file (skip download)
python -m app.main --cli --local /path/to/video.mp4 --no-upload

# Full options
python -m app.main --cli --help
```

Key CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--url` | — | YouTube URL or search query |
| `--local` | — | Path to a local video (skips download) |
| `--upload` | false | Upload to MediaCMS |
| `--no-convert` | false | Skip equirectangular conversion |
| `--confidence-threshold` | 0.5 | Minimum detection confidence for conversion |
| `--detection-frames` | 10 | Frames passed to the detector |
| `--preview-frames` | 5 | Preview frames for the UI |
| `--playlist` | — | Existing MediaCMS playlist ID/name |
| `--new-playlist` | — | Create a new playlist and add the video to it |
| `--no-manifest` | false | Do not save a JSON job manifest |

---

## Workflow Stages

```
source URL / local file
    │
    ▼
1.  YouTube search / URL resolution  (optional)
2.  yt-dlp download                  → data/downloads/
3.  Codec normalisation (ffmpeg)     — single pass, shared by all subsequent steps
4.  Preview frame extraction         → data/downloads/previews/   (UI thumbnails)
5.  Projection detection             (OpenCV analysis engine)
6.  Equirectangular conversion       (ffmpeg v360 filter)   — only when warranted
7.  MediaCMS upload                  (optional)
8.  Job manifest saved               → data/jobs/job_<timestamp>.json
```

### Preview frames vs. detector frames

| | Preview frames | Detector analysis frames |
|---|---|---|
| Purpose | UI thumbnails / display | Projection classification |
| Module | `core/preview_frames.py` | `detector/video_io.py` |
| Output dir | `data/downloads/previews/` | `data/frames/` |
| Format | JPEG | PNG |

---

## Projection Types

| Value | Description |
|---|---|
| `equirectangular` | Standard 2:1 360° format |
| `stereo_equi` | Side-by-side or top-bottom stereo equirectangular |
| `eac` | Equi-Angular Cubemap (YouTube VR format) |
| `cubemap` / `cubic` | 3×2 cubic layout |
| `unknown` | Could not determine with sufficient confidence |

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `YOUTUBE_API_KEY` | — | YouTube Data API v3 key |
| `CMS_API_URL` | — | MediaCMS `/api/v1/media` endpoint |
| `CMS_USER` / `USER` | — | MediaCMS username (Basic Auth) |
| `CMS_PASSWORD` / `PASSWORD` | — | MediaCMS password |
| `CMS_TOKEN` / `TOKEN` | — | MediaCMS CSRF token |
| `DOWNLOADS_DIR` | `data/downloads` | Download output directory |
| `VPD_FRAMES_OUTPUT_DIR` | `data/frames` | Detector frame cache |
| `VPD_DEBUG_OUTPUT_DIR` | `data/debug` | Debug visualisation output |
| `VPD_MIN_FRAMES_ANALYZED` | `5` | Minimum frames for a valid result |
| `VPD_MAX_DISCARD_RATIO` | `0.8` | Maximum acceptable frame discard ratio |
| `VPD_MIN_LAYOUT_SCORE_MARGIN` | `0.1` | Confidence margin between top layouts |
| `VPD_LINE_CENTER_BAND_RATIO` | `0.3` | Line detection centre-band width |
| `VPD_LINE_CENTER_MAX_DISTANCE_RATIO` | `0.05` | Line centre-distance tolerance |
| `VPD_LINE_MAX_SLOPE` | `0.1` | Maximum accepted line slope |
| `VPD_LINE_MIN_COVERAGE_RATIO` | `0.7` | Minimum line coverage across frame |
| `VPD_STEREO_HIST_SIMILARITY_THRESHOLD` | `0.85` | Histogram similarity for stereo detection |
| `VPD_SAVE_STEREO_HALVES` | `false` | Save left/right half debug images |
| `VPD_FLOW_ALGORITHM` | `farneback` | Optical flow algorithm (`farneback` or `dis`) |

---

## Output Artifacts

| Artifact | Location | Description |
|---|---|---|
| Downloaded video | `data/downloads/` | Original download from yt-dlp |
| Normalised video | `data/downloads/` | Codec-normalised copy (if different) |
| Converted video | `data/downloads/` | Equirectangular-converted output |
| Preview frames | `data/downloads/previews/<stem>/` | JPEG thumbnails for the UI |
| Debug frames | `data/debug/<run>/` | Optical-flow and line visualisations |
| Job manifest | `data/jobs/job_<timestamp>.json` | Full structured job record |

---

## Development

```bash
# Run tests
python -m pytest tests/

# Run the smoke-test suite
python test_fixes.py

# Run a single detection (no upload)
python -m app.main --cli --local /path/to/video.mp4 --no-upload --verbose
```

---

## Original VideoProjectionDetector

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Modules](#modules)
- [Projection Types](#projection-types)
- [Conversion Behavior](#conversion-behavior)
- [API](#api)
- [Configuration](#configuration)
- [Development](#development)
- [Tests](#tests)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

## Background

360° video platforms and players encode content in several spatial layouts. Automatically identifying the layout is a prerequisite for correct playback, transcoding, and quality analysis pipelines.

VideoProjectionDetector works without external ML models. It uses deterministic computer-vision steps:

1. A central horizontal (or vertical) structural line that appears in non-equirectangular 360° frames is detected and used as a primary cue.
2. Stereo histogram matching is used to identify stereo-equirectangular pairs.
3. Dense optical-flow analysis and layout hypothesis scoring (EAC vs. cubemap) resolves the remaining cases.
4. A reliability-aware retry strategy re-runs the analysis with different frame spacing when initial confidence is insufficient.

## Install

**System requirements**

- Python ≥ 3.9
- OpenCV with contrib (`opencv-contrib-python`)
- ffmpeg on `PATH` — required for video compatibility fallback and for projection conversion

**Python environment**

```sh
python -m venv VideoProjectionDetector
source VideoProjectionDetector/bin/activate
pip install opencv-contrib-python numpy
```

**ffmpeg**

Install ffmpeg through your system package manager:

```sh
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

## Usage

### Python API

```python
from pipeline import process_downloaded_video

result = process_downloaded_video("/path/to/video.mp4")

print(result["projection_type"])   # e.g. "equirectangular"
print(result["confidence"])        # e.g. 0.87
print(result["motion_reliable"])   # True / False

# Conversion result (when applicable)
conv = result["conversion"]
print(conv["skipped"])             # True when no conversion is needed
print(conv["output_path"])         # Path to converted file, or None
```

### CLI

```sh
python detector.py /path/to/video.mp4
```

The CLI prints the detected projection type, confidence, and the paths of any debug frames saved to disk.

### Output directory

Each run creates a timestamped directory under `frames/` containing:

| Subfolder | Contents |
|---|---|
| `main_frames/` | Sampled primary frames |
| `secondary_frames/` | Temporal neighborhoods per main frame |
| `motion_vectors/` | Optical-flow overlays and per-pair decisions |
| `line_detection/` | Structural line detection outcomes per frame |

## Pipeline

`pipeline.py` orchestrates the following sequence:

1. **Video compatibility fallback** — attempt direct OpenCV decoding; if it fails, transcode to H.264/yuv420p with ffmpeg and re-open.
2. **Frame extraction** — extract equidistant *main frames* for structural detection and denser *secondary frames* (temporal neighborhoods) for motion analysis.
3. **Black-frame filtering** — discard frames that are fully or near-fully black.
4. **Horizontal and vertical line detection** — detect the central seam that characterises non-equirectangular frames. Vertical detection is used as a fallback for left-right stereo candidate identification.
5. **Equirectangular wrap-around evidence** — compute positive boundary-continuity evidence per frame as a supplement to the seam-based classifier.
6. **Stereo histogram matching** — compare top/bottom (or left/right) halves to detect stereo-equirectangular layouts.
7. **Motion analysis** — compute dense Farneback optical flow over secondary frame pairs; split each frame into a 2×3 region grid; derive magnitude-weighted directional vectors per region.
8. **Layout scoring** — score EAC and cubemap hypotheses using angular consistency rules; aggregate pairwise decisions.
9. **Reliability gate and retry** — if the result is unreliable (too few valid pairs, low confidence margin), retry with wider temporal spacing (up to two additional attempts).
10. **Projection conversion** — if the detected type is EAC or cubemap, optionally convert to monoscopic equirectangular via ffmpeg v360.

## Modules

| Module | Responsibility |
|---|---|
| `pipeline.py` | Main orchestrator: end-to-end flow, retry logic, final classification |
| `detector.py` | Backward-compatible entrypoint; legacy Spanish function names; CLI |
| `video_io.py` | Video decoding, codec fallback, main-frame and secondary-frame extraction |
| `preprocessing.py` | Grayscale conversion and Gaussian blur for line and motion branches |
| `line_detection.py` | Horizontal and vertical seam detection (Canny + HoughLinesP + FFT verification) |
| `stereo_detection.py` | Histogram correlation and Bhattacharyya-based stereo matching |
| `motion_analysis.py` | Farneback optical flow, 2×3 region splitting, directional aggregation |
| `region_validation.py` | Region filtering by concentration and active-pixel ratio |
| `projection_logic.py` | EAC / cubemap hypothesis scoring and pairwise decision rules |
| `equirectangular_detection.py` | Positive equirectangular evidence via wrap-around boundary continuity and 2:1 aspect-ratio prior |
| `projection_conversion.py` | ffmpeg-based conversion of detected EAC / cubemap to equirectangular |
| `debug_utils.py` | All file I/O for debug artifacts; logging helpers; run-directory management |

## Projection Types

| Type | Description |
|---|---|
| `equirectangular` | Standard monoscopic 2:1 equidistant cylindrical projection |
| `stereo_equi` | Stereo equirectangular (two stacked or side-by-side equirectangular images) |
| `eac` | Equi-angular cubemap (YouTube 360 standard) |
| `cubic` | Conventional cubemap; 3×2 layout assumed (adjust `_V360_INPUT_FORMAT` in `projection_conversion.py` for other variants) |
| `unknown` | Insufficient data or low-reliability result |

## Conversion Behavior

After detection, `process_downloaded_video()` automatically calls `convert_detected_projection_to_equirectangular()`. The output of the conversion is included in the returned result dict under the `"conversion"` key.

| Detected type | Conversion action | Reason |
|---|---|---|
| `equirectangular` | Skipped | Already target format |
| `stereo_equi` | Skipped | Geometry is already equirectangular; stereo-to-mono flattening is out of scope |
| `eac` | Converted | `ffmpeg v360=eac:equirect` |
| `cubic` | Converted | `ffmpeg v360=c3x2:equirect` |
| `unknown` | Skipped | Cannot safely convert without a known layout |

Converted files are written to the run directory (or next to the source video when no run directory exists) with the suffix `_equirectangular.mp4`.

**Audio handling during conversion**

360° videos commonly carry first-order ambisonic audio with non-standard channel layouts that standard AAC encoders cannot process. The conversion layer:

1. Attempts conversion with audio re-encoded to AAC (`-c:a aac -b:a 192k`).
2. If ffmpeg fails with an ambisonic/unsupported-channel-layout error, retries automatically with audio dropped (`-an`).
3. A failure on the retry still returns the full detection result; only the conversion sub-result reflects the failure.

This means a conversion failure never turns a successful detection into a failed overall process.

## API

### `process_downloaded_video(video_path, output_dir=None)`

Main public entry point. Runs the full detection and conversion pipeline.

**Returns** a dict with at minimum:

```python
{
    "success": bool,
    "projection_type": str,       # "equirectangular" | "stereo_equi" | "eac" | "cubic" | "unknown"
    "confidence": float,          # [0, 1]
    "frames_analyzed": int,
    "frames_with_line": int,
    "motion_reliable": bool,
    "motion_reliability_reason": str,
    "motion_pairs_total": int,
    "motion_pairs_valid": int,
    "motion_confidence": float,
    "stats": dict,
    "conversion": {               # always present
        "attempted": bool,
        "success": bool,
        "skipped": bool,
        "reason": str,
        "output_path": str | None,
        "ffmpeg_command": list | None,
        "stderr": str,
        "error": str | None,
    },
    "converted_to_equirectangular": bool,
    "converted_video_path": str | None,
}
```

### `run_detection_pipeline(video_path, ...)`

Lower-level function that runs a single detection pass with explicit frame data. Useful when you supply pre-extracted frames or need fine control over thresholds. See `pipeline.py` for the full parameter list.

### `convert_detected_projection_to_equirectangular(video_path, projection_type, output_dir=None)`

Stand-alone conversion function. Returns the conversion result dict described above. Safe to call independently of the detection pipeline.

## Configuration

All thresholds are read from environment variables at startup (or from a `.env` file in the project root). Key variables:

| Variable | Default | Description |
|---|---|---|
| `VPD_FRAMES_OUTPUT_DIR` | `./frames` | Root directory for debug output |
| `VPD_NUM_FRAMES` | `10` | Main frames per detection attempt |
| `VPD_LINE_CENTER_MAX_DISTANCE_RATIO` | `0.10` | Vertical tolerance for seam acceptance |
| `VPD_STEREO_HIST_THRESHOLD` | `0.85` | Minimum histogram correlation for stereo match |
| `VPD_MIN_MOTION_CONFIDENCE` | `0.20` | Minimum confidence for motion classification |
| `VPD_FLOW_ALGORITHM` | `farneback` | Optical-flow algorithm |

Run `python -c "from pipeline import load_config, CONFIG; print(CONFIG)"` to inspect all active values.

## Development

```sh
git clone https://github.com/gorkafebe/VideoProjectionDetector.git
cd VideoProjectionDetector
python -m venv VideoProjectionDetector
source VideoProjectionDetector/bin/activate
pip install opencv-contrib-python numpy
```

All processing modules are side-effect free — they return data structures and do not write files directly. All debug I/O is centralised in `debug_utils.py`.

## Tests

```sh
source VideoProjectionDetector/bin/activate
python -m unittest tests/test_projection_conversion.py -v
```

The test suite covers:

- Correct ffmpeg v360 filter generation for `eac` (`v360=eac:equirect`) and `cubic` (`v360=c3x2:equirect`)
- Audio-present vs audio-dropped command variants
- Audio-failure detection and automatic retry without audio
- Skip behavior for `equirectangular` and `stereo_equi`
- Skip behavior for `unknown` projection
- Output path construction

## Known Limitations

- **ffmpeg dependency** — video compatibility fallback and projection conversion both require ffmpeg on `PATH`. Detection-only mode works without it, but the compatibility fallback may fail on certain codecs.
- **Cubemap layout assumption** — conversion assumes a 3×2 cubemap layout (`c3x2`). Sources using a different cube arrangement (e.g. 6×1 strip, cross layout) will require updating `_V360_INPUT_FORMAT["cubic"]` in `projection_conversion.py`.
- **No stereo-to-mono flattening** — `stereo_equi` videos are skipped during conversion because the geometry is already equirectangular. Flattening two stacked equirectangular images into a single mono frame is out of scope in the current implementation.
- **Static scenes** — the motion-analysis branch requires sufficient camera or scene motion. Near-static content may not yield reliable directional constraints for EAC vs. cubemap scoring.
- **Ambisonic audio** — 360° content commonly carries first-order ambisonic audio. The conversion layer handles this by retrying without audio, but the resulting output file will contain no audio track if the retry is needed.
- **Single-file assumption** — the pipeline processes one video file per call. Batch processing must be implemented at the caller level.

## Contributing

Open an issue or submit a pull request. Please keep changes focused and include or update tests when modifying conversion or detection logic.

## License

See [LICENSE](LICENSE).


## New Modular Configuration

The project has been refactored into a modular pipeline. Behavior, thresholds, output naming, and retry strategy are preserved.

### Core Modules

- `video_io.py`
   - Video compatibility fallback (OpenCV -> ffmpeg transcode)
   - Main-frame extraction
   - Secondary-frame extraction

- `preprocessing.py`
   - Frame preparation for optical flow (grayscale + Gaussian blur)
   - Frame preparation for line detection (grayscale + light Gaussian blur)

- `line_detection.py`
   - Horizontal line detection logic
   - Line debug drawing primitives (no file I/O)

- `motion_analysis.py`
   - Farneback optical flow computation
   - 2x3 region splitting
   - Per-region motion aggregation

- `region_validation.py`
   - Region filtering based on concentration and active-ratio constraints

- `projection_logic.py`
   - EAC/Cubemap hypothesis scoring
   - Angular consistency evaluation
   - Pair-level decision rule

- `stereo_detection.py`
   - Histogram build/compare
   - Stereo-equirectangular decision logic

- `debug_utils.py`
   - Logging helpers
   - Debug image rendering/saving
   - Run output directory creation and stats formatting

- `pipeline.py`
   - Main orchestrator connecting all modules
   - End-to-end decision flow and retry loop

- `detector.py`
   - Backward-compatible entrypoint and wrapper
   - Preserves CLI and legacy public function names while delegating to `pipeline.py`

## Overview

The detector distinguishes between:

- Non-equirectangular content (detected through a central horizontal structural cue)
- EAC vs Cubemap layout (detected through optical-flow geometry)

The system does not rely on external ML models. It uses deterministic computer vision steps, layout hypothesis scoring, and reliability-aware retries.

## High-Level Flow (Orchestrator)

`pipeline.py` executes the following sequence:

1. Extract main frames.
2. Detect horizontal line on each non-black main frame.
3. If no line is detected in analyzed frames -> classify as `equirectangular`.
4. If line exists -> run stereo histogram check.
5. If stereo matches threshold -> classify as `stereo_equi`.
6. Otherwise run motion analysis on secondary frame sequences.
7. Score EAC vs Cubemap hypotheses and aggregate pair outcomes.
8. Apply reliability gates and deterministic retry strategy if needed.

## Processing Pipeline

### 1) Video Compatibility Handling

1. Attempt direct decoding with OpenCV (`cv2.VideoCapture`).
2. If decoding fails, transcode with `ffmpeg` to a compatible H.264/`yuv420p` copy.
3. Re-open and validate the converted file with OpenCV before processing.

### 2) Frame Extraction

Two frame layers are used:

- Main frames: equidistant samples across the video (with temporal padding to avoid black boundaries).
- Secondary frames: temporal neighborhoods around each main-frame position (`P-Δ`, `P`, `P+Δ`) used only for motion analysis.

This separation keeps structural detection and motion scoring independent.

### 3) Horizontal Line Detection (Structural Cue)

For each valid main frame:

1. Convert to grayscale with a light Gaussian blur (kernel 3x3, sigma=0).
2. Detect edges (Canny).
3. Detect line segments (HoughLinesP).
4. Accept lines that are:
   - Nearly horizontal (low slope)
   - Centered around the vertical midpoint band

If enough main frames contain this cue, the video is treated as non-equirectangular and forwarded to motion-based layout classification.

### 4) Motion Analysis (Dense Optical Flow)

For each consecutive pair in secondary sequences:

1. Apply Gaussian blur (configurable kernel/sigma) only in the motion branch.
2. Compute dense Farneback optical flow.
3. Convert flow to magnitude/angle.
4. Split frame into a fixed 2x3 grid.
5. For each region, compute representative motion direction via magnitude-weighted circular mean.
6. Reject regions with weak motion or low directional concentration (high variance/bimodal behavior).

### 5) Projection Classification (EAC vs Cubemap)

For each frame pair, evaluate two layout hypotheses and select the better-scoring one.
Across all valid pairs, aggregate decisions; if EAC ratio is above threshold, final class is `eac`, otherwise `cubic`.

### 6) Reliability Check and Adaptive Retry

After pairwise motion aggregation, reliability is evaluated using:

- Minimum number of valid motion pairs
- Motion confidence derived from class ratio separation

If reliability is low, the pipeline retries with wider temporal spacing and reduced frame count (deterministic retry plan), re-running extraction and motion analysis from scratch.

## 2x3 Grid and Face Mapping

Frame is partitioned as:

- Top row: `(0,0) (0,1) (0,2)`
- Bottom row: `(1,0) (1,1) (1,2)`

## EAC Mapping

- Top row: `LEFT, FRONT, RIGHT`
- Bottom row: `BOTTOM, BACK, TOP`

## Cubemap Mapping

- Top row: `LEFT, RIGHT, TOP`
- Bottom row: `BOTTOM, FRONT, BACK`

## Motion-Based Layout Scoring

Each hypothesis is scored from orientation-corrected region angles using physically grounded relative relationships:

- `BACK` vs `FRONT` is evaluated as opposite direction (`|diff| ~ 180°`, magnitude-based).
- EAC emphasizes left/right symmetry around `FRONT` when all three are present.
- Cubemap emphasizes side-face perpendicularity (`|LEFT-FRONT|` and `|RIGHT-FRONT|` near `90°`) when `FRONT` is present.
- If `FRONT` is unavailable, fallback pairwise consistency checks are used instead of hard-failing the pair.

Angular comparisons are normalized on the circular domain, with configurable tolerance bands.

The layout with higher consistency score wins for the pair.

## Orientation Correction (Why It Matters)

Raw region angles are not directly comparable across all faces in EAC.

- In Cubemap, regions share a common orientation, so no per-region correction is required.
- In EAC, bottom-row faces have different local "up" directions and must be rotated to a common reference frame before angle comparison.

Current correction model for EAC bottom row:

- `BOTTOM`: `-90°`
- `BACK`: `+90°`
- `TOP`: `+90°`

Without this normalization, true geometric relationships can appear inconsistent (including apparent near-`180°` mismatches).

## Debugging and Artifacts

Each run creates a unique output directory with subfolders:

- `main_frames/`: sampled primary frames
- `secondary_frames/`: temporal frames grouped by parent main frame
- `motion_vectors/`: optical-flow overlays, region vectors, pair decisions
- `line_detection/`: main-frame structural outcomes (used/discarded with reason)

All debug side effects are centralized in `debug_utils.py`; processing modules return data structures and do not write files directly.

File naming encodes traceability details such as frame type, global frame index, parent relationship, and status (`used` / `discarded_*`).

Terminal logs are structured to trace lifecycle events at:

- Main frame level
- Secondary frame level
- Motion pair level
- Region level (valid/invalid + reason)

This enables direct mapping from console output to saved artifacts and final decisions.

## Limitations

- Requires sufficient motion: static or near-static scenes may not provide reliable directional constraints.
- Sensitive to low texture/noisy regions: optical flow may degrade in blurred, flat, or compression-heavy content.

## Environment Configuration

Configuration is loaded from `.env` (if present) and environment variables. Current keys:

- `VPD_PROJECT_ROOT`
- `VPD_FRAMES_OUTPUT_DIR`
- `VPD_DEBUG_OUTPUT_DIR`
- `VPD_MIN_FRAMES_ANALYZED`
- `VPD_MAX_DISCARD_RATIO`
- `VPD_MIN_LAYOUT_SCORE_MARGIN`
- `VPD_FFT_MIN_MARGIN` (Minimum EAC/cubic margin for FFT tiebreaker activation, default: 0.08)
- `VPD_LINE_CENTER_BAND_RATIO`
- `VPD_LINE_CENTER_MAX_DISTANCE_RATIO`
- `VPD_STEREO_HIST_THRESHOLD`
- `VPD_SAVE_STEREO_HALVES`
- `VPD_FLOW_ALGORITHM` – Flow algorithm for motion analysis (farneback or dis, default: farneback)

Defaults and loading behavior are defined in `pipeline.py`.

   