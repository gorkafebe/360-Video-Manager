# Improvement Plan — 360 Video Manager

_Last updated: 2026-06-16_

---

## Session 2 — AV1 Bug Fix & Performance

All items below were scoped in the approved plan for this session. Fixes are applied to `detector/video_io.py` and `detector/pipeline.py`.

### Status table

| Fix | Description | File | Status |
|-----|-------------|------|--------|
| 2.3 | VideoCapture resource leak — `finally` release | `video_io.py` | ✅ Already applied (Session 1) |
| AV1 | Omit `-hwaccel auto` for AV1 input in `convert_video_codec` | `video_io.py` | ✅ Applied |
| AV1-log | Warning when AV1 + OpenCV can't decode in `extract_main_frames` | `video_io.py` | ✅ Applied |
| 2.1 | Strip flow arrays from pair cache when visualizations are off | `pipeline.py` | ✅ Applied |
| 2.2 | Release flow array immediately after visualization write | `pipeline.py` | ✅ Applied |
| 2.5 | Thread-safe `_MOTION_CAPABILITY_SNAPSHOT_EMITTED` with `threading.Lock` | `pipeline.py` | ✅ Applied |
| 2.4 | Parallel ffmpeg extraction for large batches (>10 frames) | `video_io.py` | ✅ Applied |

### AV1 — Technical notes

**Root cause of crash**: `convert_video_codec` unconditionally passed `-hwaccel auto` to ffmpeg. On platforms where AV1 hardware decode is only partially supported at the API level (Windows DXVA2, NVDEC without AV1 support, Intel QSV), ffmpeg exits non-zero — raising `FrameExtractorError`.

**Fix**: After `_can_decode_with_opencv` returns `False`, `probe_video_stream` is called to determine the codec. If the codec is in `{"av1", "libaom-av1", "libdav1d"}`, `-hwaccel auto` is omitted from the ffmpeg command. Software decoding via `libaom-av1` or `libdav1d` (if present in the ffmpeg build) then succeeds.

**Platform notes**:
- macOS VideoToolbox: no AV1 decode support as of macOS 14.
- Windows DXVA2/D3D11VA: AV1 decode requires AV1 codec cards (Intel Arc, NVIDIA 30xx+, AMD 6xxx+). Older hardware fails silently or with non-zero exit.
- Linux VAAPI/VDPAU: AV1 support depends on driver version; software fallback (`libdav1d`) is generally available.
- The fix preserves `-hwaccel auto` for all other codecs (H.264, HEVC, VP9) — no behavior change for those.

**Preferred ffmpeg build**: Include `--enable-libdav1d` for fastest AV1 software decode. Fall back to `--enable-libaom` if libdav1d is unavailable.

### Fix 2.1 — Flow array memory

`_analyze_motion_pair_detailed` now accepts `store_flow: bool = False`. Both call sites in `_classify_non_equirectangular` pass `store_flow=bool(motion_visualizations_dir)`. When visualization is disabled (the common production path), flow arrays (up to ~58 MB at 4K per pair) are never stored in `short_pair_cache` or returned in pair results.

### Fix 2.2 — Flow array release timing

After `save_frame_debug(vis_path, vis, "MOTION")` writes the visualization frame, `pair_result["flow"] = None` is set immediately. This releases the large numpy array before the pair result is added to the accumulation list, rather than waiting for the entire classification function to return.

### Fix 2.4 — Parallel ffmpeg extraction

`_extract_batch_frames_ffmpeg` routes to `_extract_batch_frames_ffmpeg_parallel` when `len(timestamps_seconds) > MAX_SINGLE_PASS_FRAMES` (currently 10). The parallel path spawns one ffmpeg process per timestamp using fast-seek (`-ss` before `-i`), up to `max_workers=4` threads. On failure it falls back to the existing single-pass `select=` filter approach. The threshold can be tuned by changing `MAX_SINGLE_PASS_FRAMES` in `video_io.py`.

---

## Session 1 — Code Scan Backlog

Items from `SCAN_REPORT.md` not yet addressed.

| Item | Severity | Effort | Sprint |
|------|----------|--------|--------|
| `probe_video_stream` cache not invalidated between pipeline runs in long-lived process | Low | S | Next |
| `_classify_non_equirectangular` function length (~500 lines) — candidate for decomposition | Low | L | Later |
| No retry on `subprocess.TimeoutExpired` in `_extract_batch_frames_ffmpeg` | Low | S | Next |
| `convert_detected_projection_to_equirectangular` — `stereo_equi → mono` path intentionally unimplemented (see README) | — | — | Decision needed |
| Visualization code in `_classify_non_equirectangular` interleaved with classification logic | Low | M | Later |

---

## Needs Decision

### `stereo_equi → mono` conversion

`convert_detected_projection_to_equirectangular` (in `projection_conversion.py`) has a code path for `stereo_equi` input that raises `NotImplementedError`. The README notes this conversion is out of scope. Before implementing:

- Clarify whether the use case (top/bottom or side-by-side equirectangular stereo → mono equirectangular) is actually needed for any planned features.
- If yes, decide whether to implement simple cropping (take one eye) or full stereo-to-mono reprojection.

---

## Tests added this session

- `tests/test_video_io_av1.py` — 9 tests covering:
  - AV1 codec omits `-hwaccel` from transcode command
  - `libaom-av1` codec name also treated as AV1
  - H.264 input retains `-hwaccel auto`
  - Transcode succeeds without raising on AV1 input
  - Transcode raises `FrameExtractorError` on ffmpeg failure
  - Large batch (>10 frames) routes to parallel ffmpeg path
  - Small batch stays on single-pass select= path
  - Parallel failure falls back to select= filter
  - `return_diagnostics=True` on large batch sets `mode="parallel"`

---

## Session 3 — Line Detection False-Positive Fix (2026-06-16)

Fixes applied to `detector/line_detection.py`, `config/settings.py`, `detector/pipeline.py`.

### Status table

| Fix | Description | File | Status |
|-----|-------------|------|--------|
| 3.1 | Remove FFT bypass in non-fallback quality gate | `line_detection.py:468-471`, `line_detection.py:937-941` | ✅ Applied |
| 3.2 | Parameterise `fft_min_dominance` — forwarded from config and both detect functions | `line_detection.py`, `config/settings.py`, `pipeline.py` | ✅ Applied |
| 3.3 | Parameterise `strong_coverage_ratio` (was hardcoded 0.40) | `line_detection.py`, `config/settings.py`, `pipeline.py` | ✅ Applied |
| 3.4 | Parameterise `morph_length_ratio` (was hardcoded 0.02) | `line_detection.py`, `config/settings.py`, `pipeline.py` | ✅ Applied |
| 3.5 | Add `import logging` + `logger.debug` at quality gate in both detect functions | `line_detection.py` | ✅ Applied |

### Root causes addressed

**H1 [CRITICAL — Fixed]**: `detect_horizontal_line` and `detect_vertical_line` had an `or (high_coverage and ...)` branch in the quality gate that triggered detection for any Hough line spanning ≥40% of the frame width **without requiring FFT confirmation**. Any strong content edge (horizon, subtitle bar, building ledge) spanning 40%+ of width at centre position passed this gate unconditionally.

**Fix**: Gate rewritten to `fft_confirmed AND quality_ok AND strict_centered AND continuity_ok AND (slope_tight OR high_coverage)`. FFT is now required on all non-fallback paths. `high_coverage` still relaxes `slope_tight` (useful for perfectly horizontal real seams).

**H2 [HIGH — Configurable]**: FFT `min_dominance=0.10` (now `VPD_LINE_FFT_MIN_DOMINANCE`) was too permissive for narrow centre-strip ROIs where horizontal content edges (horizons, overlaid graphics) produce horizontal-dominant FFT signatures. The default is preserved at 0.10 for backward compatibility; raise to 0.20–0.30 if false positives remain.

**H3 [MEDIUM — Configurable]**: Morphological CLOSE kernel length (now `VPD_LINE_MORPH_LENGTH_RATIO`, default 0.02) at 38px for 1920px-wide frames bridges short content edge fragments into Hough-detectable segments. Reduce to 0.005 to suppress fragment bridging.

### New environment variables

| Env var | Default | Effect |
|---------|---------|--------|
| `VPD_LINE_FFT_MIN_DOMINANCE` | `0.10` | Minimum FFT horizontal-energy dominance; raise to reduce false positives |
| `VPD_LINE_STRONG_COVERAGE_RATIO` | `0.40` | Fraction of frame dimension at which `slope_tight` is relaxed |
| `VPD_LINE_MORPH_LENGTH_RATIO` | `0.02` | Morphological close kernel length ratio; reduce to prevent fragment bridging |

### Remaining limitation

A single strong horizontal content edge at frame centre (e.g. a stationary equirectangular camera aimed at the horizon) still passes all per-frame gates because the FFT of the narrow centre ROI genuinely confirms horizontal energy. The 55% frame-agreement guard in `pipeline.py` provides protection in this case. Adding seam-position stability across frames as an additional gate is recorded as a "Needs Decision" item.

### Tests added this session

- `tests/test_line_detection.py` — 10 tests covering:
  - No Hough lines → no detection (horizontal and vertical)
  - Full-width/height centre line detected (true positive preserved)
  - Short fragment rejected (below `hough_min_length`)
  - Diagonal edge rejected (slope > `max_slope`)
  - Off-centre line rejected (outside `center_tolerance`)
  - **Fix 3.1**: high-coverage Hough line rejected when FFT=False (horizontal and vertical)
  - **Fix 3.2**: `fft_min_dominance` forwarded to `_verify_line_with_fft`

### Diagnostic tool

`scripts/diagnose_line_detection.py` — sample frames from a video, run detection with configurable thresholds, print a summary table (frame / h-line / v-line / FFT-H / FFT-V / quality) and save debug images.

```bash
python scripts/diagnose_line_detection.py video.mp4 --frames 10
python scripts/diagnose_line_detection.py video.mp4 --fft-min-dominance 0.20
```
