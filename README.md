# 360-Video-Manager

Download, detect, convert, and upload 360° videos — via GUI.

---

## What this does

360-Video-Manager is a Python desktop application with a CustomTkinter GUI
that takes a 360° video from a YouTube search through a complete pipeline:
download → projection detection → format conversion → upload to a MediaCMS
instance. (An already-downloaded local file can be processed programmatically
via `JobOptions(local_video_path=...)` in `workflows/unified_pipeline.py`; the
GUI itself works from YouTube.) Projection detection (equirectangular vs.
stereo vs. EAC vs. cubemap) is fully deterministic — it relies on classical
computer-vision techniques (Hough/LSD line detection, histogram comparison,
optical flow, FFT analysis) and requires no ML models or GPU.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Architecture](#architecture)
- [Pipeline stages](#pipeline-stages)
- [Projection types](#projection-types)
- [Output artifacts](#output-artifacts)
- [Environment variables](#environment-variables)
- [Known issues / caveats](#known-issues--caveats)
- [Building a standalone executable](#building-a-standalone-executable)

---

## Prerequisites

- **Python 3.9+**
- **ffmpeg** on `PATH` — required for codec normalisation and equirectangular
  conversion. Install via your system package manager:
  ```bash
  sudo apt install ffmpeg          # Debian / Ubuntu
  sudo dnf install ffmpeg          # Fedora / RHEL
  brew install ffmpeg              # macOS (Homebrew)
  ```
- **tkinter** — not bundled by default on some Linux Python builds; install
  the system package before running the GUI:
  ```bash
  sudo apt install python3-tk      # Debian / Ubuntu
  sudo dnf install python3-tkinter # Fedora / RHEL
  ```
- **A JavaScript runtime** (`deno` recommended, or `node`) — modern YouTube
  extraction requires one to compute the download tokens; without it yt-dlp
  falls back to a degraded client whose URLs fail partway through with
  `HTTP 403 Forbidden`. The downloader auto-discovers `deno`/`node`/`bun` on
  `PATH` and in common user-local locations (`~/.deno/bin`, `~/.local/bin`,
  `~/.local/node/bin`), so installing either is enough:
  ```bash
  curl -fsSL https://deno.land/install.sh | sh   # installs to ~/.deno/bin
  # or: sudo apt install nodejs
  ```

---

## Setup

### Option A — one command

```bash
./scripts/setup.sh
source .venv/bin/activate
```

`scripts/setup.sh` checks for `python3` and `ffmpeg` on `PATH`, creates
`.venv` if it doesn't exist, installs the runtime dependencies from
`requirements.txt`, and verifies `tkinter` is importable.

### Option B — manual

```bash
git clone https://github.com/gorkafebe/360-Video-Manager.git
cd 360-Video-Manager
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
```

> **OpenCV dependency note**: this project uses `opencv-contrib-python`
> (which already includes core OpenCV modules). Do not install
> `opencv-python` in the same environment — having both installed causes
> import conflicts.

All subsequent commands must be run from the project root with the
virtualenv active so that `app`, `config`, `core`, `detector`, `workflows`,
and `utils` are importable as top-level packages.

---

## Configuration

Copy `.env.example` to `.env` in the project root (or export the variables
directly):

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
MediaCMS upload will fail gracefully with a clear error when their
credentials are missing. The full reference of every recognised variable is
in [Environment variables](#environment-variables) below.

---

## Running the app

```bash
python -m app.main
```

The GUI launches a CustomTkinter window. Its interface is in Spanish; the
button labels are quoted below as they appear. Workflow:

1. **Search** — type a YouTube URL or search query and press Enter or click
   **Buscar**. Results appear as scrollable cards; **Mostrar más resultados**
   loads the next page.
2. **Select** — click a result card to populate the detail panel
   (**Vídeo seleccionado**) with the title, channel, and URL.
3. **⬇ Descargar** (download & process) — starts the unified pipeline in a
   background thread: download → codec telemetry → detect projection →
   convert if needed. Progress and status are shown in the always-visible
   bottom bar.
4. **⬆ Subir** (upload) — becomes available once download & processing has
   finished.
   Under **Opciones de subida**, set a title, optionally add comma-separated
   tags, and choose an existing MediaCMS playlist or create a new one
   (**＋ Nuevo…**) before uploading.
5. **Log panel** — toggled with **▶ Mostrar registro** / **▼ Ocultar
   registro**; displays structured pipeline log output in real time.

The download and upload buttons are always visible regardless of window size
(they are anchored to a fixed bottom frame). The results, detail panel, and
upload options are in a scrollable content area above.

---

## Architecture

```
app/
  main.py               Entry point: launches the GUI
  gui/
    gui_app.py           CustomTkinter GUI (no pipeline logic)
    progress_utils.py    Download progress parsing and rate-limiting helpers
config/
  settings.py           All configuration loaded from env / .env
  logging_config.py     Console and file log handlers
core/
  downloader.py         yt-dlp wrapper
  youtube.py             YouTube Data API search and thumbnail helpers
  uploader.py            MediaCMS HTTP upload client
  preview_frames.py      JPEG preview-frame extraction (UI thumbnails)
  models.py               JobResult, DetectorStats, UploadResult dataclasses
  job_manifest.py         JSON manifest persistence (data/jobs/)
detector/
  pipeline.py            Main detection orchestrator
  video_io.py             OpenCV frame extraction + ffmpeg codec-normalisation fallback
  line_detection.py       Horizontal / vertical seam detection
  stereo_detection.py     Histogram-based stereo-equirectangular detection
  motion_analysis.py      Optical-flow backends + region and geometry evidence scoring
  projection_logic.py     EAC vs cubemap angular scoring (motion stage; informative only)
  projection_geometry.py  Per-face pixel-to-ray geometry for EAC/c3x2 (from ffmpeg vf_v360.c)
  face_continuity_detection.py  EAC-vs-cubic decision from face-boundary content continuity
  equirectangular_detection.py  Positive equirectangular evidence: 360° join, polar stretch, mono check
  projection_conversion.py      ffmpeg v360 conversion
  preprocessing.py        Frame pre-processing helpers
  region_validation.py    Region filtering
  debug_utils.py          All debug-image and log I/O
workflows/
  unified_pipeline.py    Orchestration: download → normalise → detect → convert → upload
utils/
  exceptions.py          Exception hierarchy
  paths.py                Path helpers
scripts/
  setup.sh                One-command environment bootstrap
packaging/
  build_app.spec          PyInstaller spec for the standalone desktop executable
  README_build.md         How to build the executable (see that file for details)
  icon*.{png,ico,icns}, icon_generator.py   Application icons and their generator
requirements.txt          Runtime dependencies (mirrors pyproject.toml)
```

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
    │           (no full-file transcode by default; VPD_FORCE_FULL_CODEC_NORMALIZATION forces one)
    │
    ├─ Stage 4: Preview frame extraction          → data/downloads/previews/
    │           (core/preview_frames.py)           JPEG, evenly-spaced, skips head/tail
    │
    ├─ Stage 5: Projection detection              (detector/pipeline.py)
    │           Seam detection → stereo check → motion stage →
    │           face-boundary continuity (EAC vs cubic) → reliability gate
    │           Returns projection_type + confidence
    │
    ├─ Stage 6: Equirectangular conversion        (detector/projection_conversion.py)
    │           Only when convert_if_needed=True and confidence ≥ threshold
    │           unknown → conversion skipped, source left untouched
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
4. Dense optical-flow analysis — capability-aware backend selection among
   Farneback, DIS, TV-L1, DeepFlow, PCAFlow and SparseToDense (all dense;
   no Lucas-Kanade / sparse tracking is used anywhere in the detector);
   per-region directional aggregation. This runs the motion stage (used for the
   non-equirectangular reliability gate), but the EAC-vs-cubic *decision* is made
   by the face-boundary continuity check in step 5, not by the flow angles.
5. **Face-boundary continuity (EAC vs cubic).** EAC and c3x2 are geometrically
   near-identical — same six faces, same 3×2 packing — and differ only in the
   per-face resampling and in *which* internal borders are genuine shared cube
   edges. Derived from ffmpeg's own `vf_v360.c` (see
   `detector/projection_geometry.py`): an EAC frame's four within-row borders are
   real edges across which content flows continuously, whereas a c3x2 frame has
   no continuous internal border at all. `detector/face_continuity_detection.py`
   measures the *contrast* between the would-be-edge borders and the
   would-be-seam borders (contrast is used, not absolute continuity, because
   ordinary equirectangular content is continuous everywhere), plus an absolute
   edge floor for EAC and a symmetric absolute ceiling for cubic. Continuity is
   measured *structurally* — border strips are high-passed and correlated in short
   windows pooled by median — not as a raw pixel correlation, which would respond
   to the horizon brightness shared by two side-facing faces rather than to actual
   content continuity. The
   texture gate scales to each frame's own noise floor so near-black footage stays
   measurable, and `detector/preprocessing.py` applies CLAHE to dark frames before
   seam detection (gated on median luminance so ordinary frames are byte-identical).
   The check is
   single-frame and needs no motion — the case the optical-flow route handles
   worst. The optical-flow angular vote is not used for this decision: measured on
   its own it is right only 40% of the time it commits (below chance for a two-way
   decision), because it compares 3D face-normal angles against 2D packed-atlas flow
   directions that only coincide for roll-dominant motion. When the continuity signal is weak on both border
   groups the check abstains, yielding `unknown`.
6. Positive equirectangular evidence (`detector/equirectangular_detection.py`).
   Three geometric checks per frame, all required: **A** the 360° join — the
   frame's last and first columns must be as continuous as any two adjacent
   columns inside it; **B** polar stretching — near the top and bottom rows,
   horizontal detail all but vanishes; **C** mono, not stereo — the middle rows
   are an unstretched equator (they are poles in top-bottom stereo) and the frame
   does not match its own other half (stereo eyes are near-copies). Evidence is
   strong when enough frames pass all three (`VPD_EQUI_*`). The frame's aspect
   ratio is recorded for debugging only and takes no part in the decision.
7. Reliability gate — if confidence is too low or too few valid frames were
   analysed, the detection result is `unknown`.

**Seam confirmation.** A seam candidate must be confirmed spectrally before it
counts. The confirmer compares orientation energy in two bands of the 2-D power
spectrum as **mean energy per frequency bin**; because the search ROI is highly
anisotropic (roughly 20 × 1920 px at 1920×960), comparing band *sums* would make
the statistic resolution-dependent. `VPD_LINE_FFT_MIN_DOMINANCE` is therefore
measured against a ~0.000 baseline at every resolution.

**When no seam is found at all.** Finding no seam is *absence* of evidence for a
packed layout, not *presence* of evidence for equirectangular geometry, and a
missed seam is the detector's most common error. So a zero-seam frame set is
classified `equirectangular` only when step 5 independently corroborates it
(`is_strong_evidence`). Otherwise the result is `unknown` and not marked
reliable, which lets the retry plan resample with different frame counts and
spacing. Such a result carries confidence `0.0`, so it stays below the
conversion-confidence threshold and conversion is skipped rather than guessing
`eac` — the branch cannot tell a missed EAC seam from equirectangular content
whose wrap evidence fell just short, and warping the latter would corrupt it.

### Conversion decision

| Detected projection | Conversion action |
|---|---|
| `equirectangular` | Skipped — already target format |
| `stereo_equi` | Currently skipped — geometry is already equirectangular (see note below for expected behavior) |
| `eac` | Converted via `ffmpeg v360=eac:equirect` |
| `cubic` | Converted via `ffmpeg v360=c3x2:equirect` |
| `unknown` | **Skipped** — the source is left untouched (see note below) |

For `stereo_equi`, the expected outcome is conversion to mono
equirectangular output for both stereo packing variants (top-bottom and
left-right). The current implementation takes the skip path for
`stereo_equi`.

**Why `unknown` is never converted**

`unknown` is a legitimate terminal detection outcome, not an error: the
detector deliberately refuses to commit when the evidence is ambiguous.
Guessing a projection for `unknown` (for example, assuming `eac`, the most
common YouTube format) would push the video through the wrong geometric remap
whenever the guess is wrong, while the job still reports success — a true
equirectangular 1280×640 test pattern converted as EAC comes back as 1280×570
of warped, duplicated cube faces. The confidence gate alone cannot prevent this,
because `confidence` is assigned in `run_detection_pipeline` *before* the
classification is demoted to `unknown`, so an unknown result can carry a high
confidence value and pass the threshold.

`_stage_convert_to_equirectangular` therefore skips conversion for `unknown` and records
`JobResult.conversion_skipped_reason = "projection_unknown"`. The original
video is preserved unchanged and remains the asset offered for upload; the GUI
states explicitly that no conversion was applied and why.

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
| Codec-normalised video | next to the source video | Temporary, audio-stripped `<stem>_compat_*.mp4`, made only when OpenCV could not decode the original. Used for analysis only, never delivered or uploaded, and deleted when the job ends |
| Converted video | `data/downloads/` | `<stem>_equirectangular.mp4`; only produced when conversion ran |
| Preview frames | `data/downloads/previews/` | JPEG thumbnails for the GUI |
| Detector analysis frames | `data/frames/<video_stem>-<timestamp>/` | Per-run subdirectories: `main_frames/`, `secondary_frames/`, `motion_vectors/`, `line_detection/`, `logs/` |
| Job manifest | `data/jobs/job_<id>.json` | JSON record with all result fields; `schema_version: "1.0"` |

All `data/` subdirectories are created automatically by
`Settings.ensure_runtime_dirs()` when the GUI starts (and again at the start of
each job).

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
| `VPD_LINE_CENTER_BAND_RATIO` | `0.08` | Not used by the detector; the seam search band is set by `VPD_LINE_CENTER_SEARCH_BAND_RATIO` |
| `VPD_LINE_CENTER_SEARCH_BAND_RATIO` | `0.02` | Strict seam search band ratio (thin central ROI used for Hough/LSD detection) |
| `VPD_LINE_CENTER_MAX_DISTANCE_RATIO` | `0.02` | Max relative distance of seam from centre |
| `VPD_LINE_MAX_SLOPE` | `0.05` | Maximum accepted seam slope |
| `VPD_LINE_MIN_COVERAGE_RATIO` | `0.20` | Minimum seam coverage fraction across frame width |
| `VPD_LINE_MIN_QUALITY_SCORE` | `0.62` | Minimum seam structural quality score for non-fallback candidates |
| `VPD_LINE_FALLBACK_MIN_QUALITY_SCORE` | `0.78` | Minimum seam quality score for fallback (LSD/fitLine) candidates |
| `VPD_LINE_FFT_MIN_DOMINANCE` | `0.015` | Minimum FFT orientation-energy dominance. The statistic is area-normalised (mean energy per frequency bin), so its seamless baseline is ~0.000 at every resolution. Raise to reduce false positives; above ~0.025 it rejects most real seams. |
| `VPD_LINE_STRONG_COVERAGE_RATIO` | `0.40` | Fraction of frame dimension at which the slope-tightness requirement is relaxed |
| `VPD_LINE_MORPH_LENGTH_RATIO` | `0.02` | Morphological close kernel length ratio; reduce to prevent fragment bridging |
| `VPD_LINE_ENABLE_PROFILE_GATE` | `false` | Set `true` to make the spatial projection-profile check a hard gate (advisory-only by default) |
| `VPD_LINE_PROFILE_MIN_COVERAGE_RATIO` | `0.20` | Fraction of frame columns/rows that must agree on the profile peak location |
| `VPD_LINE_PROFILE_MIN_PROMINENCE` | `3.0` | Peak / median gradient ratio for the profile gate; raise for stricter rejection |
| `VPD_STEREO_HIST_THRESHOLD` | `0.92` | Histogram correlation threshold for stereo detection |
| `VPD_STEREO_SEAM_GUARD_RATIO` | `0.02` | Guard band excluded around detected seam before top/bottom (or left/right) comparison |
| `VPD_STEREO_MIN_VALID_HALF_RATIO` | `0.22` | Minimum usable side/half size ratio required after seam-guard cropping |
| `VPD_STEREO_EDGE_SIMILARITY_THRESHOLD` | `0.08` | Minimum edge-structure similarity required for a stereo frame match |
| `VPD_STEREO_MIN_SEAM_FRAMES` | `2` | Minimum seam-detected frames required before stereo classification runs |
| `VPD_STEREO_MIN_STABILITY_RATIO` | `0.55` | Minimum temporal match stability ratio for stereo decision |
| `VPD_SAVE_STEREO_HALVES` | `true` | Save left/right half-frame debug images |
| `VPD_FLOW_ALGORITHM` | `dis` | Dense optical-flow algorithm for the motion stage. Honoured literally whenever the OpenCV build provides it; only an unavailable algorithm falls back (see "Motion rollout profile behavior"). `deepflow` / `tvl1` remain selectable but cost ~34× / ~82× more per frame pair than `dis` |
| `VPD_FLOW_ENABLE_REFINEMENT` | `false` | Enable optional variational refinement on top of base optical flow. **Forced to `true`** under `VPD_MOTION_ROLLOUT_PROFILE=robust` or `high_accuracy` (the default) regardless of this variable |
| `VPD_FLOW_ENABLE_FB_CHECK` | `false` | Enable forward-backward optical-flow consistency filtering. **Forced to `true`** under `robust`/`high_accuracy` (the default) regardless of this variable |
| `VPD_FLOW_FB_THRESHOLD` | `1.5` | Forward-backward consistency threshold in pixels |
| `VPD_ENABLE_FACE_CONTINUITY` | `true` | Decide EAC vs cubic by single-frame face-boundary continuity (`detector/face_continuity_detection.py`). Set `false` to decide by the optical-flow angular vote instead |
| `VPD_FACE_CONTINUITY_EAC_CONTRAST` | `0.30` | Median edge-minus-seam continuity at or above which frames may be called EAC |
| `VPD_FACE_CONTINUITY_EAC_EDGE_FLOOR` | `0.35` | Absolute edge continuity an EAC verdict must clear on its own (real EAC: 0.68–0.85) |
| `VPD_FACE_CONTINUITY_CUBIC_CONTRAST` | `0.20` | Median contrast below which frames may be called cubic |
| `VPD_FACE_CONTINUITY_CUBIC_EDGE_CEILING` | `0.25` | Absolute edge continuity a cubic verdict must stay under (real cubic: −0.02–0.10) |
| `VPD_FACE_CONTINUITY_CUBIC_EDGE_DECISIVE` | `0.20` | Edge continuity below which cubic is called on edge evidence alone; values between the bands abstain |
| `VPD_FACE_CONTINUITY_MIN_FRAMES` | `2` | Minimum frames with measurable border texture before any verdict is issued |
| `VPD_EQUI_JUNCTION_MIN_CORR` | `0.84` | Equirectangular signal A: minimum correlation between the frame's last and first column (the 360° join) |
| `VPD_EQUI_JUNCTION_MAX_RATIO` | `11.0` | Signal A: maximum join difference relative to the median adjacent-column difference inside the frame |
| `VPD_EQUI_JUNCTION_MIN_TEXTURE` | `1.0` | Signal A: minimum std (grey levels) of each edge column; below it the frame is unusable rather than scored |
| `VPD_EQUI_POLAR_MAX_ANISOTROPY` | `0.41` | Signal B: maximum horizontal/vertical gradient ratio in the top and bottom bands (equirectangular poles are stretched sideways) |
| `VPD_EQUI_BAND_MIN_TEXTURE` | `0.01` | Signals B/C: minimum mean gradient of a band; a constant (e.g. black) band fails instead of passing trivially |
| `VPD_EQUI_EQUATOR_MIN_ANISOTROPY` | `0.27` | Signal C: minimum gradient ratio in the middle band; in top-bottom stereo the middle rows are poles |
| `VPD_EQUI_MAX_HALF_SIMILARITY` | `0.66` | Signal C: maximum similarity between the frame's two halves on either axis; stereo eyes are near-copies |
| `VPD_EQUI_MIN_PASS_RATIO` | `0.5` | Minimum fraction of usable frames passing A, B and C for positive equirectangular evidence |
| `VPD_EQUI_MIN_FRAMES` | `2` | Minimum usable frames before any equirectangular verdict |
| `VPD_ENABLE_GEOMETRY_EVIDENCE` | `false` | Opt-in per-pair ORB+homography telemetry pass. It does not influence the EAC-vs-cubic decision (its scores are identical for both hypotheses) |
| `VPD_GEOMETRY_EVIDENCE_WEIGHT` | `0.20` | Weight of the geometry telemetry channel; it does not affect the EAC-vs-cubic decision |
| `VPD_MOTION_ROLLOUT_PROFILE` | `high_accuracy` | Motion profile: `baseline`, `robust`, `high_accuracy` |
| `VPD_MOTION_MAX_WORK_RESOLUTION` | `960` | Long-edge cap (px) for frames entering dense optical flow / ORB / homography. Prevents multi-hour motion stages on 4K–8K sources; angles are scale-invariant. `0` disables (flow at native resolution) |
| `VPD_FORCE_FULL_CODEC_NORMALIZATION` | `false` | Force a full-file codec-normalization transcode before detection |

### Motion rollout profile behavior

- **Tier A (primary evidence/features)**: Canny+morphology+HoughLinesP, LSD, DFT orientation checks, ORB+BFMatcher, homography with RANSAC/USAC_MAGSAC, `estimateAffinePartial2D`, forward-backward consistency masking, variational refinement.
- **Tier B (high-value optional flow)**: `dis`, `tvl1` — **`dis` is preferred**. TV-L1 is
  an iterative variational method whose cost is prohibitive on 360 sources (a single TV-L1
  flow at 3840×2048 exceeds four minutes); DIS gives comparable dense flow, is robust to the
  brightness discontinuities at cube-face boundaries, and runs ~orders of magnitude faster.
- **Tier C (fallback flow)**: `deepflow`, `pcaflow`, `sparse_to_dense`.

**Working resolution.** Dense flow / ORB / homography run at a bounded working resolution —
frames are downscaled so the longer edge is at most `VPD_MOTION_MAX_WORK_RESOLUTION` (default
960px, `0` disables) before flow. Cost grows ~quadratically with pixels, and 360 sources are
commonly 4K–8K, so this is the difference between seconds and hours per run. Per-region motion
*angles* (what layout scoring consumes) are scale-invariant, and a fixed working resolution
also makes the pixel-magnitude thresholds consistent across source sizes.

Runtime flow selection is capability-aware and deterministic:
- **The configured `VPD_FLOW_ALGORITHM` runs whenever this OpenCV build provides it**, under
  every profile. `farneback` is always available. Choosing `deepflow` or `tvl1` logs a warning
  with its measured per-pair cost relative to `dis` (at the 960 px work resolution: DIS 72.9 ms,
  Farneback 202.4 ms, DeepFlow 2444.1 ms, TV-L1 5963.6 ms).
- Only when the requested algorithm is **unavailable** is a fallback chosen, from a chain of
  the request, then (`robust` / `high_accuracy`) Tier B `dis`, `tvl1` or (`baseline`) `dis`,
  then Tier C, then `farneback`. The first available entry runs, and the substitution is
  logged as a warning, never silent.
- `robust` / `high_accuracy` additionally **force-enable** `VPD_FLOW_ENABLE_REFINEMENT` and
  `VPD_FLOW_ENABLE_FB_CHECK` to `true`, overriding their individual environment values. These
  take their literal configured value under `baseline`. `VPD_ENABLE_GEOMETRY_EVIDENCE` is not
  force-enabled by any profile (its scores are identical for EAC and cubic).
- `tvl1`, `deepflow`, `pcaflow` and `sparse_to_dense` require `cv2.optflow` (the OpenCV
  *contrib* module — see the OpenCV note under [Setup](#setup)); `dis` and `farneback` are in
  core OpenCV. Without `cv2.optflow`, a request for one of the four contrib algorithms falls
  back to `dis` (if present in the OpenCV build) or `farneback`.

Safety invariants:
- Non-equirectangular classification with failed reliability is downgraded to `unknown`.
- Contradictory motion evidence keeps `unknown` unless consistency overrides are met.
- Retry logic keeps early-stop behavior for stalled low-motion signals.

---

## Known issues / caveats

### Cubemap layout assumption

Conversion of `cubic` projection assumes the 3×2 face layout (`ffmpeg v360=c3x2:equirect`).
Videos using other cubemap arrangements (6×1 strip, cross layout, etc.) will
produce incorrect output. Update `_V360_INPUT_FORMAT["cubic"]` in
`detector/projection_conversion.py` if needed.

### Static-scene detection reliability

The motion-analysis branch requires sufficient camera or scene motion. Near-
static 360° content may produce low-confidence results and a final classification
of `unknown`. Conversion is then skipped and the original video is preserved
unchanged rather than being remapped on a guess.

### Seam-detection recall

Measured against synthetic ground truth, the seam detector finds
**54 % of true seams at a 0 % false-positive rate**. The
remaining misses are attributable to the fallback-candidate quality floor, the
Hough stage finding no candidate at all, and the continuity gate — not to the
spectral confirmation step. Because a missed seam yields `unknown` rather
than `equirectangular` (see [Projection detection](#projection-detection)), the
practical effect is a retry followed by a skipped conversion, not a silently
wrong classification. Improving recall further would require changes to the
morphological kernel, the seam-candidate bucketing, and the search-band width.

### tkinter system dependency

On Linux environments where tkinter is missing, importing `app.gui.gui_app`
will fail. Install `python3-tk` / `python3-tkinter` before running the GUI.

---

## Building a standalone executable

For deploying to machines without Python installed, see
[`packaging/README_build.md`](packaging/README_build.md) for instructions on
building a double-clickable executable with PyInstaller.
