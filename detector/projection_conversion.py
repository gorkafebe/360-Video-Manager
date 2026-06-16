"""Projection conversion module.

Converts detected video projections to equirectangular using ffmpeg.

Projection type mapping
-----------------------
The following table documents the conversion strategy for each type:

| projection_type | Action                | ffmpeg v360 from→to        |
|-----------------|-----------------------|----------------------------|
| equirectangular | skip (already target) | –                          |
| stereo_equi     | skip (geometry is OK) | –                          |
| eac             | convert               | eac→equirect               |
| cubic           | convert               | c3x2→equirect              |
| unknown         | skip                  | –                          |

v360 input format identifiers used
-----------------------------------
- EAC  (equi-angular cubemap):  ``eac``
- Cubic 3×2 layout:             ``c3x2``

Reference:
  https://ffmpeg.org/ffmpeg-filters.html#v360

Notes on EAC
    The ffmpeg v360 EAC identifier is ``eac`` (equi-angular cubemap).
    This is the standard encoding used by platforms like YouTube 360.
    Note: ``e`` is ffmpeg's identifier for *equirectangular input*, which
    is a completely different format and must NOT be used for EAC sources.

Notes on cubic / cubemap
    The ``c3x2`` identifier represents a 3-column × 2-row cubemap layout.
    This matches the ``CUBEMAP_LAYOUT`` in ``projection_logic.py`` which
    uses (row, col) indices with 2 rows and 3 columns.
    ffmpeg v360 layout for c3x2 (rows from top):
      Row 0: RIGHT, LEFT,   TOP
      Row 1: BOTTOM, FRONT, BACK
    Adjust ``_V360_INPUT_FORMAT`` if a different cube variant is needed.

Stereo-equi note
    The current implementation intentionally does not flatten stereo-equi
    to mono equirectangular.  The geometry is already equirectangular, so
    no spatial transform is needed.  Stereo-to-mono flattening is out of
    scope and should be implemented as a separate, explicitly documented
    step.
"""

import logging
import os
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants — projection type → v360 input format string
# ---------------------------------------------------------------------------

# Only types that require an actual remapping are listed here.
# equirectangular and stereo_equi are handled by the skip path.
_V360_INPUT_FORMAT: Dict[str, str] = {
    # EAC (equi-angular cubemap) — YouTube 360 standard.
    # ffmpeg v360 identifier is "eac".  Do NOT use "e" — that is equirectangular input.
    "eac": "eac",
    # Cubic 3×2 layout as defined by CUBEMAP_LAYOUT in projection_logic.py
    # (2 rows × 3 columns of cube faces).
    "cubic": "c3x2",
}

# Stderr substrings that indicate an audio-encoding failure.
# When seen, the conversion should be retried without audio.
_AUDIO_FAILURE_SIGNALS: tuple = (
    "ambisonic",
    "unsupported channel layout",
    "channel layout",
)

_GENERIC_ENCODER_FAILURE_SIGNALS: tuple = (
    "encoder opening failed",
    "error initializing output stream",
    "error while opening encoder",
)

_AUDIO_CONTEXT_SIGNALS: tuple = (
    " audio",
    "audio ",
    " audio:",
    "aac",
)

_VIDEO_FAILURE_SIGNALS: tuple = (
    "height not divisible by 2",
    "width not divisible by 2",
    "invalid too big or non positive size",
    "error while filtering",
    "libx264",
    "v360",
)

# Projection types for which we skip conversion (geometry already suitable)
_SKIP_PROJECTION_TYPES = frozenset({"equirectangular", "stereo_equi"})

# Types considered "convertible" to equirectangular
_CONVERTIBLE_PROJECTION_TYPES = frozenset(_V360_INPUT_FORMAT.keys())

_HW_ENCODER_ORDER = ("h264_nvenc", "h264_qsv", "h264_videotoolbox", "h264_amf", "libx264")
_HW_ENCODERS = frozenset({"h264_nvenc", "h264_qsv", "h264_videotoolbox", "h264_amf"})

_HARDWARE_ENCODER_RUNTIME_FAILURE_SIGNALS: tuple = (
    "cannot load libcuda",
    "cuda is not available",
    "cannot init cuda",
    "no device available",
    "unsupported device",
    "device creation failed",
    "failed to initialize encoder",
    "error while opening encoder",
    "error initializing output stream",
)

_HARDWARE_BACKEND_CONTEXT_SIGNALS: tuple = (
    "nvenc",
    "qsv",
    "videotoolbox",
    "amf",
    "cuda",
)


@lru_cache(maxsize=1)
def detect_ffmpeg_h264_encoder() -> str:
    """Detect best usable ffmpeg H.264 encoder, preferring hardware backends."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "libx264"
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for candidate in _HW_ENCODER_ORDER:
        if candidate in text and _is_encoder_runtime_usable(candidate):
            return candidate
    return "libx264"


def _is_encoder_runtime_usable(encoder: str) -> bool:
    """Return True if *encoder* appears usable for a tiny encode in this runtime."""
    if encoder == "libx264":
        return True
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=1",
                "-frames:v",
                "1",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _is_hardware_encoder_runtime_failure(stderr: str, encoder: str) -> bool:
    """Return True when *stderr* indicates hardware encoder runtime failure."""
    if encoder not in _HW_ENCODERS:
        return False
    s = (stderr or "").lower()
    if not s:
        return False
    if any(sig in s for sig in _HARDWARE_ENCODER_RUNTIME_FAILURE_SIGNALS):
        return True
    if "error while opening encoder" in s or "error initializing output stream" in s:
        return any(sig in s for sig in _HARDWARE_BACKEND_CONTEXT_SIGNALS)
    return False


def _extract_video_encoder(cmd: List[str]) -> Optional[str]:
    """Return the value passed to ``-c:v`` in *cmd*, if present."""
    try:
        idx = cmd.index("-c:v")
    except ValueError:
        return None
    return cmd[idx + 1] if idx + 1 < len(cmd) else None


# ---------------------------------------------------------------------------
# ffmpeg availability
# ---------------------------------------------------------------------------

def is_ffmpeg_available() -> bool:
    """Return True if ffmpeg is found and executable on the PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def ensure_ffmpeg_available() -> None:
    """Raise ``RuntimeError`` if ffmpeg is not available."""
    if not is_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not available on PATH. "
            "Install ffmpeg to enable projection conversion."
        )


# ---------------------------------------------------------------------------
# Output path construction
# ---------------------------------------------------------------------------

def build_equirectangular_output_path(
    video_path: str,
    output_dir: Optional[str] = None,
    suffix: str = "_equirectangular",
) -> str:
    """Build a deterministic output path for the equirectangular-converted video.

    Examples::

        build_equirectangular_output_path("foo/bar/input.mov")
        # → "foo/bar/input_equirectangular.mp4"

        build_equirectangular_output_path("video.mp4", output_dir="/tmp/out")
        # → "/tmp/out/video_equirectangular.mp4"

    The output is always ``*.mp4`` regardless of the source container.

    Args:
        video_path: Path to the source video.
        output_dir: If provided, the converted file is placed here.
            Otherwise it is placed next to the source file.
        suffix: String appended to the base filename before the extension.

    Returns:
        Absolute output path string.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_name = f"{base}{suffix}.mp4"
    if output_dir:
        return os.path.join(output_dir, out_name)
    source_dir = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(source_dir, out_name)


# ---------------------------------------------------------------------------
# v360 filter and ffmpeg command construction
# ---------------------------------------------------------------------------

def _is_audio_failure(stderr: str) -> bool:
    """Return True if *stderr* contains signals of an audio-encoding failure.

    Used to decide whether a failed ffmpeg run should be retried video-only.
    Ambisonic audio (4+ channels, non-standard layout) is common in 360°
    content and cannot be encoded to standard AAC with default settings.

    Args:
        stderr: Captured stderr text from the failed ffmpeg run.

    Returns:
        True when stderr indicates an audio-related encoder failure.
    """
    s = stderr.lower()

    if any(sig in s for sig in _VIDEO_FAILURE_SIGNALS):
        return False

    if "ambisonic" in s:
        return True

    if "unsupported channel layout" in s:
        return True

    if "channel layout" in s and any(sig in s for sig in _AUDIO_CONTEXT_SIGNALS):
        return True

    if any(sig in s for sig in _GENERIC_ENCODER_FAILURE_SIGNALS):
        return any(sig in s for sig in _AUDIO_CONTEXT_SIGNALS)

    return False


def build_v360_filter_for_projection(projection_type: str) -> Optional[str]:
    """Return the ffmpeg v360 video-filter string for the given projection type.

    Returns ``None`` for types that do not require a geometric transform
    (equirectangular, stereo_equi) or for unknown types.

    v360 syntax used::

        v360=<input_format>:equirect

    This maps the detected source projection to standard equirectangular
    (equidistant cylindrical) output.

    Args:
        projection_type: One of the known detection types.

    Returns:
        A filter string like ``"v360=eac:equirect"`` or ``None``.
    """
    v360_in = _V360_INPUT_FORMAT.get(projection_type)
    if v360_in is None:
        return None
    # libx264 with yuv420p requires even width/height. Some source dimensions
    # can produce odd outputs after v360, so pad to even dimensions without
    # geometric distortion.
    return (
        f"v360={v360_in}:equirect,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2"
    )


def build_ffmpeg_command_for_projection(
    input_path: str,
    output_path: str,
    projection_type: str,
    drop_audio: bool = False,
    video_encoder: Optional[str] = None,
) -> List[str]:
    """Build an ffmpeg command list to convert *input_path* to equirectangular.

    The command always includes:
    - ``-y`` to overwrite existing output deterministically,
    - ``-loglevel error`` to suppress noise,
    - yuv420p pixel format for broad MP4 compatibility,
    - libx264 + veryfast preset for convertible types.

    Hardware acceleration note:
    ``-hwaccel auto`` is **only** applied when no software filter is needed
    (stream-copy path).  When the v360 geometric filter is used, hardware
    decoding must be disabled because hardware decoders produce device-memory
    frames that the software ``v360`` filter cannot read.  Leaving
    ``-hwaccel auto`` in the v360 path causes ffmpeg to open (and truncate)
    the output file and then exit immediately with an error, producing an
    empty output file.

    Audio strategy for geometric conversion (``drop_audio=False``):
    - ``-map 0:a?`` makes the audio stream optional (no error if absent),
    - ``-c:a aac -b:a 192k`` re-encodes to AAC for MP4 container compatibility.
    - If AAC encoding fails (e.g. ambisonic audio), pass ``drop_audio=True``
      to retry with ``-an`` (video-only output).

    For types that require a v360 geometric transform (eac, cubic) a
    ``-vf`` filter is added.  For skip-eligible types this function should
    not normally be called; see :func:`convert_detected_projection_to_equirectangular`.

    Args:
        input_path: Path to the source video.
        output_path: Desired path for the equirectangular output.
        projection_type: Detected projection type.
        drop_audio: When True, suppress all audio in the output (``-an``).
            Use this on retry when standard AAC encoding fails.

    Returns:
        List of strings forming the ffmpeg command.

    Raises:
        ValueError: If the projection type is not supported for conversion.
    """
    v360_filter = build_v360_filter_for_projection(projection_type)
    if v360_filter is None and projection_type not in _SKIP_PROJECTION_TYPES:
        raise ValueError(
            f"No ffmpeg conversion strategy defined for projection type: {projection_type!r}"
        )

    selected_encoder = video_encoder or detect_ffmpeg_h264_encoder()

    if v360_filter:
        # Geometric remapping required — use software decoding so that the
        # v360 filter receives CPU frames.  Hardware-decoded frames live in
        # device memory and cannot be fed directly to software filters;
        # including -hwaccel auto here causes ffmpeg to create (truncate) the
        # output file and then exit immediately with an error, leaving an empty
        # file on disk.
        cmd: List[str] = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", input_path,
            "-map", "0:v",
            "-vf", v360_filter,
            "-c:v", selected_encoder,
            "-pix_fmt", "yuv420p",
        ]
        if selected_encoder == "libx264":
            cmd += ["-preset", "veryfast", "-crf", "18"]
        if drop_audio:
            # Video-only output: drop audio (safe fallback for ambisonic /
            # unsupported channel layout streams).
            cmd += ["-an"]
        else:
            # Transcode audio to AAC for MP4 container compatibility.
            # ``-map 0:a?`` makes audio optional — sources without any audio
            # stream succeed without an error.
            cmd += [
                "-map", "0:a?",
                "-c:a", "aac",
                "-b:a", "192k",
            ]
    else:
        # No geometric transform needed — stream-copy to normalise container.
        # Hardware acceleration is safe here because no software filter is applied.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-hwaccel", "auto",
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "copy",
        ]

    cmd.append(output_path)
    return cmd


# ---------------------------------------------------------------------------
# ffmpeg execution
# ---------------------------------------------------------------------------

def run_ffmpeg_command(
    cmd: List[str],
    timeout_seconds: int = 3600,
) -> Dict[str, Any]:
    """Execute an ffmpeg command and return a status dict.

    Never raises; all errors are captured in the returned dict.

    Args:
        cmd: ffmpeg command as a list of strings.
        timeout_seconds: Hard timeout in seconds.

    Returns:
        Dict with keys: ``success``, ``returncode``, ``stdout``, ``stderr``,
        ``error``.
    """
    result: Dict[str, Any] = {
        "success": False,
        "returncode": -1,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout or ""
        result["stderr"] = proc.stderr or ""
        result["success"] = proc.returncode == 0
        if not result["success"]:
            result["error"] = f"ffmpeg exited with code {proc.returncode}"
    except subprocess.TimeoutExpired:
        result["error"] = f"ffmpeg timed out after {timeout_seconds}s"
    except FileNotFoundError:
        result["error"] = "ffmpeg executable not found"
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Conversion result builder helpers
# ---------------------------------------------------------------------------

def _make_skip_result(
    projection_type: str,
    input_path: str,
    reason: str,
    ffmpeg_available: bool = True,
) -> Dict[str, Any]:
    """Return a well-formed skipped-conversion result dict."""
    return {
        "attempted": False,
        "success": False,
        "skipped": True,
        "reason": reason,
        "input_path": input_path,
        "output_path": None,
        "projection_type": projection_type,
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_command": None,
        "stderr": "",
        "stdout": "",
        "error": None,
    }


def _make_failure_result(
    projection_type: str,
    input_path: str,
    output_path: Optional[str],
    cmd: Optional[List[str]],
    error: str,
    stderr: str = "",
    stdout: str = "",
) -> Dict[str, Any]:
    """Return a well-formed failed-conversion result dict."""
    return {
        "attempted": True,
        "success": False,
        "skipped": False,
        "reason": "conversion_failed",
        "input_path": input_path,
        "output_path": output_path,
        "projection_type": projection_type,
        "ffmpeg_available": True,
        "ffmpeg_command": cmd,
        "stderr": stderr,
        "stdout": stdout,
        "error": error,
    }


def _make_success_result(
    projection_type: str,
    input_path: str,
    output_path: str,
    cmd: List[str],
    stdout: str = "",
    stderr: str = "",
) -> Dict[str, Any]:
    """Return a well-formed successful-conversion result dict."""
    return {
        "attempted": True,
        "success": True,
        "skipped": False,
        "reason": "converted_to_equirectangular",
        "input_path": input_path,
        "output_path": output_path,
        "projection_type": projection_type,
        "ffmpeg_available": True,
        "ffmpeg_command": cmd,
        "stderr": stderr,
        "stdout": stdout,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert_detected_projection_to_equirectangular(
    video_path: str,
    projection_type: str,
    output_dir: Optional[str] = None,
    name_source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a video with the given detected projection to equirectangular.

    This is the only function that ``pipeline.py`` needs to call.

    Behaviour by projection type
    ----------------------------
    - **unknown**: skip, reason ``projection_unknown``
    - **equirectangular**: skip, reason ``already_equirectangular``
    - **stereo_equi**: skip, reason ``already_equirectangular_stereo_layout``
    - **eac**: convert via ``ffmpeg v360=eac:equirect``
    - **cubic**: convert via ``ffmpeg v360=c3x2:equirect``

    The result dict is always safe to return even when ffmpeg is missing or
    the conversion fails — detection results are never affected.

    Args:
        video_path: Path to the actual input file that ffmpeg will read.
            When a compatibility-transcoded intermediate was produced by
            ``video_io.convert_video_codec``, pass that path here.
        projection_type: Detected projection type string.
        output_dir: Optional directory for the converted output file.
            The directory is created automatically if it does not exist.
            If omitted the output is placed next to *name_source_path* (or
            *video_path* when *name_source_path* is not provided).
        name_source_path: Optional original source path used solely for
            computing the output file name.  Pass the user-facing original
            video path here when *video_path* points to a temporary
            compatibility-transcoded intermediate so that the output file
            carries the original video name rather than the temp-file name.

    Returns:
        A conversion result dict with at minimum the keys:
        ``attempted``, ``success``, ``skipped``, ``reason``,
        ``input_path``, ``output_path``, ``projection_type``,
        ``ffmpeg_available``, ``ffmpeg_command``, ``stderr``, ``stdout``,
        ``error``.
    """
    # --- Skip path: projection unknown ------------------------------------- #
    if not projection_type or projection_type == "unknown":
        logger.info("[CONVERSION] Skipping conversion — projection is unknown.")
        return _make_skip_result(
            projection_type=projection_type or "unknown",
            input_path=video_path,
            reason="projection_unknown",
        )

    # --- Skip path: already equirectangular geometry ----------------------- #
    if projection_type in _SKIP_PROJECTION_TYPES:
        if projection_type == "stereo_equi":
            reason = "already_equirectangular_stereo_layout"
            msg = (
                "[CONVERSION] Skipping conversion — projection is stereo_equi "
                "(geometry already equirectangular; stereo-to-mono flattening not performed)."
            )
        else:
            reason = "already_equirectangular"
            msg = "[CONVERSION] Skipping conversion — projection is already equirectangular."
        logger.info(msg)
        return _make_skip_result(
            projection_type=projection_type,
            input_path=video_path,
            reason=reason,
        )

    # --- Convertible path -------------------------------------------------- #
    if projection_type not in _CONVERTIBLE_PROJECTION_TYPES:
        logger.warning(
            "[CONVERSION] Projection type %r has no defined conversion path. Skipping.",
            projection_type,
        )
        return _make_skip_result(
            projection_type=projection_type,
            input_path=video_path,
            reason=f"unsupported_projection_type:{projection_type}",
        )

    # Check ffmpeg availability
    if not is_ffmpeg_available():
        logger.warning("[CONVERSION] ffmpeg not found — conversion skipped.")
        return _make_skip_result(
            projection_type=projection_type,
            input_path=video_path,
            reason="ffmpeg_unavailable",
            ffmpeg_available=False,
        )

    # Build output path — use original source name when available so that
    # temp-file intermediates do not pollute the output file name.
    naming_path = name_source_path if name_source_path else video_path
    output_path = build_equirectangular_output_path(naming_path, output_dir=output_dir)

    # Ensure output directory exists before invoking ffmpeg.
    out_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        logger.error("[CONVERSION] Cannot create output directory %s: %s", out_dir, exc)
        return _make_failure_result(
            projection_type=projection_type,
            input_path=video_path,
            output_path=output_path,
            cmd=None,
            error=f"cannot_create_output_dir:{exc}",
        )

    # Safety: refuse to overwrite the source
    if os.path.abspath(output_path) == os.path.abspath(video_path):
        logger.error("[CONVERSION] Output path equals input path — aborting to protect source.")
        return _make_failure_result(
            projection_type=projection_type,
            input_path=video_path,
            output_path=output_path,
            cmd=None,
            error="output_path_equals_input_path",
        )

    try:
        cmd = build_ffmpeg_command_for_projection(video_path, output_path, projection_type)
    except ValueError as exc:
        logger.error("[CONVERSION] Cannot build ffmpeg command: %s", exc)
        return _make_failure_result(
            projection_type=projection_type,
            input_path=video_path,
            output_path=output_path,
            cmd=None,
            error=str(exc),
        )

    logger.info(
        "[CONVERSION] Converting %s → equirectangular via ffmpeg (projection=%s, encoder=%s, audio=with_aac).",
        os.path.basename(video_path),
        projection_type,
        _extract_video_encoder(cmd) or "copy",
    )
    logger.debug("[CONVERSION] Command: %s", " ".join(cmd))

    ffmpeg_result = run_ffmpeg_command(cmd)

    # --- Hardware-encoder runtime fallback --------------------------------- #
    selected_encoder = _extract_video_encoder(cmd) or "copy"
    if (
        not ffmpeg_result["success"]
        and _is_hardware_encoder_runtime_failure(ffmpeg_result["stderr"], selected_encoder)
        and selected_encoder != "libx264"
    ):
        logger.warning(
            "[CONVERSION] Encoder %s failed at runtime; retrying with libx264.",
            selected_encoder,
        )
        try:
            fallback_cmd = build_ffmpeg_command_for_projection(
                video_path,
                output_path,
                projection_type,
                video_encoder="libx264",
            )
            cmd = fallback_cmd
            ffmpeg_result = run_ffmpeg_command(cmd)
        except ValueError as exc:
            logger.error("[CONVERSION] Cannot build libx264 fallback command: %s", exc)
            return _make_failure_result(
                projection_type=projection_type,
                input_path=video_path,
                output_path=output_path,
                cmd=None,
                error=str(exc),
                stderr=ffmpeg_result["stderr"],
                stdout=ffmpeg_result["stdout"],
            )

    # --- Audio-failure retry ------------------------------------------------ #
    # 360° videos commonly carry ambisonic audio (first-order / higher-order
    # ambisonics, non-standard channel layouts) that standard AAC encoders
    # cannot process.  When ffmpeg fails with audio-related errors, retry
    # with audio dropped so the geometric transform still succeeds.
    if not ffmpeg_result["success"] and _is_audio_failure(ffmpeg_result["stderr"]):
        logger.warning(
            "[CONVERSION] Audio encoding failed (likely ambisonic / unsupported "
            "channel layout). Retrying video-only (audio will be dropped)."
        )
        try:
            cmd_no_audio = build_ffmpeg_command_for_projection(
                video_path,
                output_path,
                projection_type,
                drop_audio=True,
                video_encoder=_extract_video_encoder(cmd),
            )
        except ValueError as exc:
            logger.error("[CONVERSION] Cannot build video-only command: %s", exc)
            return _make_failure_result(
                projection_type=projection_type,
                input_path=video_path,
                output_path=output_path,
                cmd=cmd,
                error=str(exc),
                stderr=ffmpeg_result["stderr"],
                stdout=ffmpeg_result["stdout"],
            )
        logger.debug("[CONVERSION] Retry command (no audio): %s", " ".join(cmd_no_audio))
        ffmpeg_result = run_ffmpeg_command(cmd_no_audio)
        if ffmpeg_result["success"]:
            logger.info("[CONVERSION] Retry succeeded (audio dropped) → %s", os.path.basename(output_path))
        cmd = cmd_no_audio

    if ffmpeg_result["success"]:
        logger.info(
            "[CONVERSION] Conversion succeeded → %s", os.path.basename(output_path)
        )
        return _make_success_result(
            projection_type=projection_type,
            input_path=video_path,
            output_path=output_path,
            cmd=cmd,
            stdout=ffmpeg_result["stdout"],
            stderr=ffmpeg_result["stderr"],
        )
    else:
        stderr_excerpt = (ffmpeg_result["stderr"] or "")[-500:]
        logger.error(
            "[CONVERSION] Conversion failed (code=%s): %s",
            ffmpeg_result["returncode"],
            stderr_excerpt,
        )
        # ffmpeg opens (truncates) the output file before it starts encoding.
        # When the command fails the output file is left empty on disk.  Remove
        # it so that callers do not encounter a zero-byte artefact.
        # Use a single os.stat() call to avoid a TOCTOU race between existence
        # and size checks.
        try:
            if os.stat(output_path).st_size == 0:
                os.remove(output_path)
                logger.debug("[CONVERSION] Removed empty output file: %s", output_path)
        except FileNotFoundError:
            pass  # file was never created — nothing to clean up
        except OSError as exc:
            logger.warning("[CONVERSION] Could not remove empty output file %s: %s", output_path, exc)
        return _make_failure_result(
            projection_type=projection_type,
            input_path=video_path,
            output_path=output_path,
            cmd=cmd,
            error=ffmpeg_result["error"] or "ffmpeg_nonzero_exit",
            stderr=ffmpeg_result["stderr"],
            stdout=ffmpeg_result["stdout"],
        )
