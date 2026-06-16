# Project Scan Report

## 1. Project Map

### Purpose
360-Video-Manager is a Python GUI application for downloading, detecting, converting, and uploading 360° videos. It uses a deterministic projection-detection engine (no ML models) combined with a complete download → process → upload workflow.

**Target Python**: 3.9+  
**Entry points**: `python -m app.main` or console script `360-video-manager` → `app.main:main`  
**Public detector API**: `detector.analyze_video_projection()`, `detector.convert_to_equirectangular()`

### Package / module responsibilities

| Package/Module | Responsibility |
|---|---|
| `app/main.py` | CLI/console-script entry; delegates immediately to GUI |
| `app/__main__.py` | Enables `python -m app` invocation |
| `app/gui/gui_app.py` | CustomTkinter GUI — orchestrates pipeline via background threads; no pipeline logic |
| `app/gui/progress_utils.py` | Download-progress parsing, rate-limiting helpers for the GUI |
| `config/settings.py` | Loads all configuration from env/.env; exposes `get_settings()` singleton |
| `config/logging_config.py` | Console (colorised) and file log handlers |
| `core/downloader.py` | yt-dlp wrapper for video download |
| `core/youtube.py` | YouTube Data API v3 search and thumbnail helpers |
| `core/uploader.py` | MediaCMS HTTP upload client (upload, playlist CRUD) |
| `core/preview_frames.py` | JPEG preview-frame extraction for GUI thumbnails |
| `core/models.py` | `JobResult`, `DetectorStats`, `UploadResult` dataclasses |
| `core/job_manifest.py` | JSON manifest persistence for job auditing |
| `detector/pipeline.py` | Main detection orchestrator; frame sampling, black-frame filtering, line/stereo/motion detection coordination, reliability gating |
| `detector/video_io.py` | OpenCV/ffmpeg frame extraction with fallback chain; ffprobe metadata; codec-normalisation transcode |
| `detector/line_detection.py` | Horizontal / vertical seam detection (Hough/LSD/FFT quality) |
| `detector/stereo_detection.py` | Histogram-based stereo-equirectangular detection |
| `detector/motion_analysis.py` | Optical-flow backends; forward-backward consistency; per-region directional aggregation; ORB geometry evidence |
| `detector/projection_logic.py` | EAC vs cubemap hypothesis scoring by face-angle symmetry |
| `detector/equirectangular_detection.py` | Wrap-around boundary-continuity evidence (positive equi detection) |
| `detector/projection_conversion.py` | ffmpeg v360 conversion; hardware-encoder / audio-failure fallbacks |
| `detector/preprocessing.py` | Frame grayscale + blur preprocessing |
| `detector/region_validation.py` | Region motion concentration/activity threshold checks |
| `detector/debug_utils.py` | Debug-image saving, per-run directory creation, log formatting |
| `workflows/unified_pipeline.py` | Orchestration layer: all 8 pipeline stages, timing, error isolation |
| `utils/exceptions.py` | Custom exception hierarchy |
| `utils/paths.py` | Path helpers delegating to `get_settings()` |

### Internal dependency graph (key edges)

```
app/ → workflows/unified_pipeline → core/{downloader, youtube, uploader, preview_frames, job_manifest, models}
                                   → detector/{pipeline, video_io, projection_conversion}
                                   → config/settings
detector/pipeline → detector/{video_io, line_detection, stereo_detection, motion_analysis,
                              projection_logic, equirectangular_detection, projection_conversion,
                              debug_utils, region_validation, preprocessing}
                 → config/settings (via load_config at import time)
```

No circular imports detected.

### Third-party dependencies

| Package | Pinned range | Notes |
|---|---|---|
| `google-api-python-client` | `>=2.196.0,<3` | Reasonably pinned |
| `yt-dlp` | `>=2025.1.0,<2027.0.0` | Wide upper bound — intentional for a fast-moving tool |
| `requests` | `>=2.32.0,<3` | Good |
| `python-dotenv` | `>=1.0.1,<2` | Good |
| `pillow` | `>=10.4.0,<12` | Good |
| `opencv-contrib-python` | `>=4.13.0.92,<4.14` | Tightly pinned; explicit note in README about not mixing with `opencv-python` |
| `numpy` | `>=1.26,<3` | Good |
| `customtkinter` | `>=5.2,<6` | Good |

No unpinned or suspicious dependencies.

---

## 2. Performance & Complexity Findings

| Severity | File | Lines | Issue summary |
|---|---|---|---|
| MEDIUM | `detector/pipeline.py` | 869–1527 | `run_detection_pipeline` is ~660 lines with 6+ distinct responsibilities |
| MEDIUM | `detector/pipeline.py` | 365–866 | `_classify_non_equirectangular` is ~500 lines; functions-within-functions make it hard to test |
| MEDIUM | `detector/pipeline.py` | 319 | Flow numpy arrays (up to ~58 MB each at 4K) included in pair-result dicts and held in `short_pair_cache` throughout each sequence iteration |
| MEDIUM | `detector/video_io.py` | 248–336 | `_extract_batch_frames_ffmpeg` builds a `select=` filter expression by joining one clause per timestamp; at 30+ secondary frames this grows to hundreds of characters but stays within OS limits |
| LOW | `detector/pipeline.py` | 57 | Module-level `CONFIG = load_config()` is frozen at import time; runtime env-var changes after first import are invisible |
| LOW | `detector/video_io.py` | 31 | `_PROBE_CACHE` grows unboundedly across a session; in practice bounded by number of files but no eviction |
| LOW | `detector/motion_analysis.py` | 342 | `matches = sorted(matches, ...)` inside `compute_region_affine_angles` — minor O(n log n) sort per region per pair |

### Detailed findings

**2c-1 — Flow array memory in `short_pair_cache`** (`detector/pipeline.py:319`)

`_analyze_motion_pair_detailed` returns the raw flow array (shape `H×W×2`, float32) in the result dict under key `"flow"`. Results are stored in `short_pair_cache` for all short pairs within a sequence. At 4K resolution (3840×1920), each flow array is ≈58 MB. With a 3-frame secondary sequence there are 2 cached entries (≈116 MB). The cache is local per sequence iteration so it is freed between sequences, but peak within-iteration memory can be large for high-resolution content. The flow is only needed for the debug-visualization branch (`if motion_visualizations_dir:`), not for detection. When debug output is disabled, the flow is computed and stored but never used from the cache.

**2d-1 — `run_detection_pipeline` complexity** (`detector/pipeline.py:869–1527`)

The function mixes frame extraction, black-frame filtering, line detection, stereo detection, equirectangular evidence collection, motion classification dispatch, reliability gating, and stats aggregation. It is 660 lines with no sub-functions. Each branching path (early equi, LR stereo, horizontal seam + motion, insufficient frames) is deeply interleaved with logging. Cyclomatic complexity is very high, making it difficult to test individual branches in isolation.

**2d-2 — `_classify_non_equirectangular` complexity** (`detector/pipeline.py:365–866`)

Three nested inner functions (`_build_pair_plan`, `_pair_weight`, `_is_pair_low_motion`) and two levels of nested loops, plus inline confidence formula switching on profile name. At ~500 lines it is the second-largest function.

---

## 3. Error Handling & Edge Case Findings

| Severity | File | Lines | Issue summary |
|---|---|---|---|
| HIGH | `detector/video_io.py` | 494–703 | `_CachedCapture` is not released when exceptions occur inside `extract_main_frames` — VideoCapture resource leak |
| MEDIUM | `detector/pipeline.py` | 553 | `secuencia[i + 1]` used for motion debug visualization instead of `secuencia[j]`; shows wrong frame when pair gap > 1 |
| MEDIUM | `detector/projection_conversion.py` | 591–595 | Docstring says `v360=e:equirect` and `v360=c6x1:equirect`; actual values are `eac` and `c3x2` — misleading documentation |
| MEDIUM | `config/settings.py` | 73 | `_load_dotenv` catches all `Exception`; non-`ImportError` failures in `load_dotenv` fall back to the manual parser without any log entry |
| MEDIUM | `core/uploader.py` | 157 | `create_playlist` calls `resp.json()` directly without checking Content-Type; a non-JSON error response silently returns `None` via the outer `except` |
| MEDIUM | `detector/debug_utils.py` | 17–37 | `ColorizedFormatter` class duplicated vs `config/logging_config.py:16–36`; the two versions use different format strings and could diverge |
| LOW | `detector/debug_utils.py` | 62 | `os.makedirs(os.path.dirname(filepath), ...)` raises `FileNotFoundError` when `filepath` has no directory component (dirname → `""`) |
| LOW | `detector/pipeline.py` | 58 | `_MOTION_CAPABILITY_SNAPSHOT_EMITTED` module global is mutated without a lock; duplicate log lines possible under concurrent detection calls |
| LOW | `core/youtube.py` | 214 | `YouTubeError = YouTubeAPIError` re-export shadows `utils.exceptions.YouTubeError` (the base class) in callers that do `from core.youtube import *` |

### Detailed findings

**3a-1 — VideoCapture resource leak in `extract_main_frames`** (`detector/video_io.py:494–703`) — **HIGH**

`_CachedCapture` wraps `cv2.VideoCapture`. When `cap_session` is `None` on entry (`_owns_cap = True`), the function creates a new capture object at line 497. The only guaranteed release is at line 633–634, reached only on the successful extraction path. Three exceptional paths bypass this release:

1. Early raise at line 525 (`fps <= 0` for `primera_ultima` mode) — capture created, not released.
2. Any exception during the extraction loop (lines 545–631) that propagates to the `except` blocks at lines 696–703 — these blocks re-raise without releasing.
3. Exception from `probe_video_stream` (line 500), `_extract_batch_frames_ffmpeg` (lines 576–601), or `cv2.imwrite` (line 688).

OpenCV's C++ destructor will eventually reclaim the capture when the Python object is garbage-collected, but the timing is non-deterministic (CPython reference counting usually reclaims immediately; PyPy/GC-managed runtimes do not guarantee it). Under rapid successive calls (e.g., retry loops) this can exhaust available file descriptors or OpenCV capture handles.

**3b-1 — Wrong frame in motion debug visualization** (`detector/pipeline.py:553`) — **MEDIUM**

Inside `_classify_non_equirectangular`, the motion visualization renders `secuencia[i + 1]` as the base image for the pair `(i, j, gap)`. When `gap == 1`, `j == i + 1` so this is coincidentally correct. When `gap > 1`, the visualization draws motion vectors (computed between frames `i` and `j`) on top of frame `i + 1`, which is neither the source nor the target frame of the flow. Debug images at gap > 1 are therefore visually misleading — they show flow from a non-adjacent pair on the wrong frame. Detection is unaffected.

**3c-1 — Stale `CONFIG` module-level constant** (`detector/pipeline.py:57`) — (documented under Gaps)

`CONFIG = load_config()` is executed once at module import. If `get_settings()` is called before environment variables are loaded (e.g., in tests that patch env vars after import), the pipeline uses stale thresholds. `run_detection_with_retries` compensates by calling `_resolve_motion_feature_flags()` which reads settings fresh, but inline references to `CONFIG["min_frames_analyzed_required"]` etc. at lines 897–909 and 1576–1578 use the frozen snapshot.

**3c-2 — Docstring incorrect v360 identifiers** (`detector/projection_conversion.py:591–595`) — **MEDIUM**

The docstring for `convert_detected_projection_to_equirectangular` reads:
```
- **eac**: convert via ``ffmpeg v360=e:equirect``
- **cubic**: convert via ``ffmpeg v360=c6x1:equirect``
```
The actual implementation uses `v360=eac:equirect` and `v360=c3x2:equirect`. Using `e` as the v360 input format would mean equirectangular input (not EAC), producing silently corrupt output. Using `c6x1` is an incorrect cubemap layout. The code is correct; only the docstring misleads.

---

## 4. Gaps & Observations

1. **Non-obvious: `stereo_result` accessed inside a guard** (`detector/pipeline.py:1419`): The expression `(frames_with_line > 0 and not stereo_result["is_stereo"]) if frames_with_line > 0 else False` looks like a potential `NameError`, but `stereo_result` is always assigned in the `else` branch of `if frames_with_line == 0:` (lines 1237–1399). The guard `if frames_with_line > 0` is correct. The code is safe but a reader will pause here.

2. **Hidden coupling — `CONFIG` frozen snapshot**: `run_detection_pipeline` reads from the module-level `CONFIG` dict (set at import time) for some thresholds, but `run_detection_with_retries` reads from `_resolve_motion_feature_flags()` which queries `get_settings()` fresh. These two paths use slightly different config snapshots if settings change between import and invocation.

3. **Dead code — `process_downloaded_video`** (`detector/pipeline.py:1713–1800`): This 88-line function is a legacy entry point from before `workflows/unified_pipeline.py` existed. It is not called by any module in the current codebase (confirmed by grep). It still performs codec conversion before detection (the old forced-normalisation path) and unconditionally triggers conversion, which is now controlled by `convert_if_needed` in `JobOptions`.

4. **Dead code — backward-compatible aliases**: `descargar_video`, `buscar_youtube`, `extraer_video_id`, `obtener_thumbnail_video`, `post_video`, `DownloadErrorCustom`, `InvalidYouTubeURLException` — all defined and unused. They add noise and the aliases for exception classes shadow the authoritative hierarchy in `utils/exceptions.py`.

5. **Missing tests** — the following modules have no corresponding test file:
   - `core/job_manifest.py` — `save_job_manifest`, `load_job_manifest`
   - `core/preview_frames.py` — `extract_preview_frames`
   - `detector/equirectangular_detection.py` — `compute_frame_equirectangular_evidence`, `aggregate_equirectangular_evidence`
   - `detector/projection_logic.py` — `evaluate_eac`, `evaluate_cubemap`, `decide_projection`
   - `detector/preprocessing.py` — `prepare_frame_for_flow`, `prepare_frame_for_line_detection`
   - `detector/region_validation.py` — `is_region_valid`
   - `detector/debug_utils.py` — `save_frame_debug`, `create_run_debug_dir`
   - `workflows/unified_pipeline.py` — `process_video_job`, all stage functions

6. **Documentation gap — `convert_detected_projection_to_equirectangular` docstring** (`detector/projection_conversion.py:591–595`): Documents wrong ffmpeg identifiers (see 3c-2 above).

7. **Documentation gap — `run_detection_pipeline` has no module-level docstring** (`detector/pipeline.py`): The module has no module-level docstring, and the main entry functions (`run_detection_pipeline`, `_classify_non_equirectangular`) have no docstrings.

8. **Non-obvious: `_resolve_motion_feature_flags` ignores `requested_algorithm` when it is `farneback` in non-baseline profiles** (`detector/pipeline.py:121–138`): The config default is `deepflow`, but the function starts with `requested_algorithm = normalize_flow_algorithm_name(str(cfg.get("flow_algorithm", "farneback")))`. If the env var `VPD_FLOW_ALGORITHM=farneback` is set, the preferred chain for non-baseline profiles starts with farneback and immediately returns `["farneback"]`, bypassing tier-B/C algorithms entirely. This is a silent performance downgrade — the user sets the baseline default but gets baseline behavior regardless of profile.

### Needs Clarification

- The `stereo_equi` → mono equirectangular conversion is listed as "currently skipped" in both the README and docstring. Is this intentional indefinitely, or is there a planned implementation?
- `process_downloaded_video` — confirm this is dead code and can be removed, or if any external callers exist (e.g., integration tests, scripts directory).

---

## 5. Applied Fixes

### Fix 1 — VideoCapture resource leak in `extract_main_frames` (HIGH)

**Problem**: When `extract_main_frames` owns the `_CachedCapture` (caller passed no `cap_session`), the object is created inside the `try` block but only released on the happy path at line 633. Exceptions during fps validation, frame extraction, or file-save bypass that release, leaking the OpenCV VideoCapture handle.

**Before** (`detector/video_io.py:493–703`):

```python
_owns_cap = cap_session is None
try:
    video_path_procesado = video_path
    if cap_session is None:
        cap_session = _CachedCapture(video_path)
    ...
    if total_frames <= 0:
        if _owns_cap:
            cap_session.release()          # only this early-exit releases
        raise FrameExtractorError(...)
    ...
    if _owns_cap:
        cap_session.release()              # happy-path release
    if not frames:
        raise FrameExtractorError(...)
    ...
    return result
except FrameExtractorError:
    raise
except Exception as exc:
    raise FrameExtractorError(...) from exc   # no release here
```

**After**: Move cap creation before the `try` block and replace all manual releases with a single `finally` clause.

### Fix 2 — Wrong frame in motion-pair debug visualization (MEDIUM)

**Problem**: `secuencia[i + 1]` is used as the base image for motion-vector overlay. When `gap > 1`, frame `i + 1` is neither frame `i` nor frame `j`, so motion vectors computed between `i` and `j` are drawn on an unrelated frame.

**Before** (`detector/pipeline.py:553`):
```python
vis = secuencia[i + 1].copy()
```

**After**:
```python
vis = secuencia[j].copy()
```

### Fix 3 — Docstring wrong v360 format identifiers (MEDIUM)

**Problem**: The docstring for `convert_detected_projection_to_equirectangular` documents `v360=e:equirect` (equirectangular input, not EAC) and `v360=c6x1:equirect` (wrong layout). The code is correct but the docstring could mislead a developer making changes.

**Before** (`detector/projection_conversion.py:591–595`):
```
- **eac**: convert via ``ffmpeg v360=e:equirect``
- **cubic**: convert via ``ffmpeg v360=c6x1:equirect``
```

**After**:
```
- **eac**: convert via ``ffmpeg v360=eac:equirect``
- **cubic**: convert via ``ffmpeg v360=c3x2:equirect``
```

---

## 6. Backlog

1. **(MEDIUM) `_load_dotenv` swallows non-`ImportError` silently** (`config/settings.py:71–74`): Replace `except Exception:` with `except ImportError:` for the expected case, then log a warning for any other exception before falling back. Non-ImportError failures (e.g., unexpected `TypeError` in dotenv) currently produce zero diagnostic output.

2. **(MEDIUM) `create_playlist` calls `resp.json()` without error handling** (`core/uploader.py:157`): If the CMS returns an HTML error page, `resp.json()` raises `ValueError`, which is caught by the outer `except Exception` and logged as "Error creating playlist". Add explicit `_safe_json_response` (already used in `get_playlists`) and return `None` with a warning.

3. **(MEDIUM) `ColorizedFormatter` duplicated** (`detector/debug_utils.py:17–37` vs `config/logging_config.py:16–36`): `debug_utils.py` should import `ColorizedFormatter` from `config.logging_config` instead of redefining it. Also `configure_run_file_logging` in `debug_utils.py` is a near-duplicate of `configure_file_logging` in `config/logging_config.py`.

4. **(LOW) Flow arrays retained in `short_pair_cache` when debug output is disabled** (`detector/pipeline.py:319`): The `"flow"` key in the pair-result dict is only consumed in the `if motion_visualizations_dir:` branch. When debug output is off, the arrays are computed and stored unnecessarily. Fix: set `"flow": None` (or omit the key) in `_analyze_motion_pair_detailed` when called without a visualization target, or strip the key before returning when visualization is disabled.

5. **(LOW) `process_downloaded_video` is dead code** (`detector/pipeline.py:1713–1800`): This legacy function forces full codec normalisation before detection, duplicates conversion logic, and is not called anywhere. Remove it.

6. **(LOW) Backward-compatible aliases add noise** (`core/downloader.py:169`, `core/youtube.py:209–214`, `core/uploader.py:290`, `utils/exceptions.py:28,48`): `descargar_video`, `buscar_youtube`, etc. are defined but unused. Remove if no external callers exist.

7. **(LOW) `save_frame_debug` raises on bare filenames** (`detector/debug_utils.py:62`): `os.makedirs(os.path.dirname(filepath), exist_ok=True)` fails with `FileNotFoundError` when `os.path.dirname(filepath) == ""`. Add `if dirname:` guard: `dirname = os.path.dirname(filepath); if dirname: os.makedirs(dirname, exist_ok=True)`.

8. **(LOW) `_MOTION_CAPABILITY_SNAPSHOT_EMITTED` not thread-safe** (`detector/pipeline.py:58`): Under concurrent detection calls (e.g., if the GUI ever launches parallel jobs), the capability snapshot could be logged twice. Add a module-level `threading.Lock` guard, or convert to `threading.Event`.

9. **(LOW) `CONFIG` frozen at import time** (`detector/pipeline.py:57`): The module-level snapshot means in-process settings changes don't propagate. In tests that patch env vars, this can cause config leakage. Mitigate by calling `load_config()` inside `run_detection_pipeline` if the overhead is acceptable, or by using a sentinel that triggers reload when settings are reset.

10. **(LOW) Missing tests for key modules**: Add unit tests for `core/job_manifest.py` (manifest serialisation round-trip), `core/preview_frames.py` (frame extraction), `detector/equirectangular_detection.py` (evidence aggregation), `detector/projection_logic.py` (scoring symmetry), and `workflows/unified_pipeline.py` (stage mocking).

---

## 7. Summary

The codebase is well-structured and production-quality in most areas: exception hierarchies are clean, the settings layer is robust, the ffmpeg integration has thorough retry/fallback logic, and the detection pipeline's reliability gating is sophisticated. The most important issue to address first is the **VideoCapture resource leak in `extract_main_frames`** (`detector/video_io.py`): the `_CachedCapture` is not released on exception paths, which under the retry loop could exhaust file descriptors on problematic videos. The second-most impactful structural issue is the **660-line `run_detection_pipeline` function** and its 500-line sibling `_classify_non_equirectangular` — both carry too many responsibilities to test or maintain safely. A positive observation: the conversion layer (`detector/projection_conversion.py`) is exemplary — well-documented, correctly separates the hardware-encoder and audio-failure retry paths, and handles the v360 hwaccel/software filter incompatibility with an accurate inline explanation.
