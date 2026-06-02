"""Unified video-processing pipeline.

This module is the single orchestration layer that ties together every
stage of the 360-Video-Manager workflow:

1. Search / URL resolution  (optional – caller may supply a URL directly)
2. Download
3. Codec normalisation
4. Preview frame extraction  (UI thumbnails — separate from analysis frames)
5. Projection detection
6. Equirectangular conversion  (when required and requested)
7. MediaCMS upload  (optional)
8. Job manifest persistence

Usage
-----
.. code-block:: python

    from workflows.unified_pipeline import process_video_job, JobOptions

    options = JobOptions(
        source_url="https://youtu.be/XXXXXXXXXXX",
        upload=True,
        convert_if_needed=True,
    )
    result = process_video_job(options)
    print(result.projection_type, result.confidence)

The :class:`JobOptions` dataclass exposes all knobs.  For simpler use-cases
you can call the individual stage functions directly.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from config.settings import get_settings
from core.models import DetectorStats, JobResult, UploadResult
from core.job_manifest import save_job_manifest, new_job_id
from utils.exceptions import (
    DownloadError,
    FrameExtractorError,
    MediaCMSError,
    WorkflowError,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JobOptions:
    """Configuration for a single video-processing job.

    Fields
    ------
    source_url
        YouTube URL or search query.  Exactly one of *source_url* or
        *local_video_path* must be set.
    local_video_path
        Path to an already-downloaded video.  Skips the download step.
    output_dir
        Directory for downloaded and converted files.  Defaults to the
        configured downloads directory.
    upload
        Whether to upload the final asset to MediaCMS.
    convert_if_needed
        Whether to convert non-equirectangular detections to equirectangular
        using ffmpeg v360.
    confidence_threshold
        Minimum detection confidence to attempt conversion (default: 0.5).
    num_detection_frames
        Number of frames passed to the detector pipeline (default: 10).
    preview_frames
        Number of preview frames to extract for the UI (default: 5).
    upload_title
        Title for the MediaCMS upload.  Defaults to the video filename.
    upload_description
        Description for the MediaCMS upload.
    upload_playlist
        Existing MediaCMS playlist name or ID to add the video to.
    upload_new_playlist
        When set, create a new playlist with this name.
    upload_category
        Existing MediaCMS category ID to assign to the uploaded video.
    upload_tags
        Optional personalized tags to attach to the uploaded video.
    save_manifest
        Whether to save a JSON job manifest (default: True).
    progress_callback
        Optional callable for download-progress notifications.
    force_full_codec_normalization
        Force the legacy full-file codec normalization transcode before
        detection. Disabled by default to prefer sampled frame extraction.
    """

    source_url: Optional[str] = None
    local_video_path: Optional[str] = None
    output_dir: Optional[str] = None
    upload: bool = False
    convert_if_needed: bool = True
    confidence_threshold: float = 0.5
    num_detection_frames: int = 10
    preview_frames: int = 5
    upload_title: Optional[str] = None
    upload_description: str = ""
    upload_playlist: Optional[str] = None
    upload_new_playlist: Optional[str] = None
    upload_category: Optional[str] = None
    upload_tags: Optional[List[str]] = None
    save_manifest: bool = True
    progress_callback: Optional[Callable] = None
    force_full_codec_normalization: bool = False


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------

def _stage_search_and_resolve(options: JobOptions) -> tuple[str, Optional[str]]:
    """Resolve source_url to a canonical YouTube URL and optional title.

    Returns (url, title).
    """
    url = options.source_url
    title: Optional[str] = None

    if url is None:
        raise WorkflowError("No source URL provided.")

    # If the query looks like a plain URL, return it directly.
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        logger.info("Source URL recognised as a direct link: %s", url)
        return url, title

    # Otherwise treat it as a search query.
    logger.info("Searching YouTube for: %s", url)
    from core.youtube import search_videos
    from utils.exceptions import YouTubeAPIError, NoYouTubeAPIKeyError

    try:
        results = search_videos(url)
    except (NoYouTubeAPIKeyError, YouTubeAPIError) as exc:
        raise WorkflowError(f"YouTube search failed: {exc}") from exc

    if not results:
        raise WorkflowError(f"No 360° videos found for query: {url!r}")

    best = results[0]
    resolved_url = best.get("url") or ""
    if not resolved_url:
        raise WorkflowError("Search result has no URL.")

    title = best.get("title")
    logger.info("Search resolved to: %s (%s)", title, resolved_url)
    return resolved_url, title


def _stage_download(
    url: str,
    output_dir: str,
    progress_callback: Optional[Callable],
) -> str:
    """Download *url* and return the local video path."""
    from core.downloader import download_video

    logger.info("Downloading: %s → %s", url, output_dir)
    try:
        path = download_video(url, output_dir=output_dir, progress_callback=progress_callback)
        logger.info("Download complete: %s", path)
        return path
    except DownloadError as exc:
        raise WorkflowError(f"Download failed: {exc}") from exc


def _stage_normalize_codec(video_path: str) -> str:
    """Ensure the video is decodable by OpenCV; convert codec if necessary.

    This is the **single canonical normalisation step**.  Neither the
    detector nor the GUI should call codec conversion independently.

    Returns the path to the normalised video (may be the same file).
    """
    from detector.video_io import convert_video_codec, FrameExtractorError as DetectorFEError

    logger.info("Normalising codec: %s", video_path)
    try:
        normalised = convert_video_codec(video_path)
        if normalised != video_path:
            logger.info("Codec conversion produced: %s", normalised)
        else:
            logger.info("Codec is already compatible; no conversion needed.")
        return normalised
    except DetectorFEError as exc:
        raise FrameExtractorError(f"Codec normalisation failed: {exc}") from exc


def _log_codec_telemetry(video_path: str, context: str) -> Dict[str, Any]:
    from detector.video_io import probe_video_stream

    metadata = probe_video_stream(video_path)
    logger.info(
        "[CODEC][%s] path=%s codec=%s pix_fmt=%s %sx%s fps=%.3f duration=%.2fs frames=%s source=%s",
        context,
        video_path,
        metadata.get("codec_name") or "unknown",
        metadata.get("pix_fmt") or "unknown",
        metadata.get("width") or 0,
        metadata.get("height") or 0,
        float(metadata.get("fps") or 0.0),
        float(metadata.get("duration") or 0.0),
        metadata.get("total_frames") or 0,
        metadata.get("source") or "unknown",
    )
    return metadata


def _stage_preview_frames(
    video_path: str,
    num_frames: int,
    output_dir: str,
) -> List[str]:
    """Extract preview/thumbnail frames for the UI."""
    from core.preview_frames import extract_preview_frames

    preview_dir = os.path.join(output_dir, "previews")
    logger.info("Extracting %d preview frames to %s", num_frames, preview_dir)
    return extract_preview_frames(
        video_path, num_frames=num_frames, output_dir=preview_dir
    )


def _stage_detect_projection(
    video_path: str,
    num_frames: int,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the robust detector pipeline on *video_path*."""
    from detector.pipeline import run_detection_with_retries

    logger.info("Running projection detection on: %s", video_path)

    result, debug_context = run_detection_with_retries(
        video_path,
        num_frames=num_frames,
        debug_base_dir=debug_dir,
    )
    if debug_context:
        logger.info("Debug output directory: %s", debug_context.get("run_dir"))
    logger.info(
        "Detection complete: projection=%s confidence=%.1f%%",
        result.get("projection_type", "unknown"),
        float(result.get("confidence", 0.0)) * 100,
    )
    return result


def _stage_convert_to_equirectangular(
    video_path: str,
    projection_type: str,
    confidence: float,
    output_dir: Optional[str],
    confidence_threshold: float,
) -> Optional[str]:
    """Convert *video_path* to equirectangular if warranted.

    Returns the converted path, or None when conversion was skipped.
    """
    if projection_type in ("equirectangular", "stereo_equi"):
        logger.info(
            "Conversion skipped: projection=%s does not require remapping.",
            projection_type,
        )
        return None

    if projection_type == "unknown":
        logger.warning(
            "Projection detection returned 'unknown'; falling back to EAC for conversion."
        )
        projection_type = "eac"

    if confidence < confidence_threshold:
        logger.info(
            "Conversion skipped: confidence=%.2f < threshold=%.2f.",
            confidence,
            confidence_threshold,
        )
        return None

    from detector.projection_conversion import convert_detected_projection_to_equirectangular

    logger.info(
        "Converting %s → equirectangular (confidence=%.1f%%)", projection_type, confidence * 100
    )
    converted = convert_detected_projection_to_equirectangular(
        video_path=video_path,
        projection_type=projection_type,
        output_dir=output_dir,
    )
    output_path = converted.get("output_path") if converted.get("success") else None
    if output_path:
        logger.info("Conversion complete: %s", output_path)
        return output_path

    logger.warning("Conversion returned unchanged path or None; treating as skipped.")
    return None


def _stage_upload(
    video_path: str,
    title: str,
    description: str,
    playlist: Optional[str],
    new_playlist: Optional[str],
    category: Optional[str],
    tags: Optional[List[str]],
) -> UploadResult:
    """Upload *video_path* to MediaCMS."""
    from core.uploader import upload_video_asset

    logger.info("Uploading: %s  title=%r", video_path, title)
    try:
        return upload_video_asset(
            video_path=video_path,
            title=title,
            description=description,
            playlist_id=playlist,
            new_playlist_name=new_playlist,
            category_id=category,
            tags=tags,
        )
    except MediaCMSError as exc:
        logger.error("Upload failed: %s", exc)
        return UploadResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Unified entry-point
# ---------------------------------------------------------------------------

def process_video_job(options: JobOptions) -> JobResult:
    """Execute the full 360-video processing workflow described by *options*.

    Steps
    -----
    1. Search / URL resolution (if source_url is not a direct link)
    2. Download (skipped when *local_video_path* is provided)
    3. Codec normalisation
    4. Preview frame extraction
    5. Projection detection
    6. Equirectangular conversion (optional)
    7. Upload to MediaCMS (optional)
    8. Job manifest persistence (optional)

    Returns a :class:`~core.models.JobResult` regardless of outcome.
    The ``success`` flag is True only when no stage raised a fatal error.
    """
    cfg = get_settings()
    cfg.ensure_runtime_dirs()
    force_full_codec_normalization = bool(
        options.force_full_codec_normalization or getattr(cfg, "force_full_codec_normalization", False)
    )

    result = JobResult(
        job_id=new_job_id(),
        source_url=options.source_url,
    )

    output_dir = options.output_dir or cfg.downloads_dir
    stage_timings: Dict[str, float] = {}

    def _time_stage(stage_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            stage_timings[stage_name] = elapsed
            logger.info("[TIMING] %s=%.3fs", stage_name, elapsed)

    try:
        # ------------------------------------------------------------------ #
        # Stage 1: Resolve source URL (skip if local file provided)
        # ------------------------------------------------------------------ #
        if options.local_video_path:
            video_path = options.local_video_path
            if not os.path.exists(video_path):
                raise WorkflowError(f"local_video_path does not exist: {video_path}")
            result.original_video_path = video_path
            logger.info("Using local video: %s", video_path)
        else:
            url, title = _time_stage("search_resolve", _stage_search_and_resolve, options)
            result.source_url = url
            if title:
                result.source_title = title

            # ------------------------------------------------------------------ #
            # Stage 2: Download
            # ------------------------------------------------------------------ #
            video_path = _time_stage(
                "download",
                _stage_download,
                url,
                output_dir,
                options.progress_callback,
            )
            result.original_video_path = video_path
            if not result.source_title:
                result.source_title = os.path.splitext(os.path.basename(video_path))[0]

        _log_codec_telemetry(video_path, "source")

        # ------------------------------------------------------------------ #
        # Stage 3: Codec normalisation (feature-flagged full transcode fallback)
        # ------------------------------------------------------------------ #
        normalized_path = video_path
        if force_full_codec_normalization:
            normalized_path = _time_stage("normalize_codec", _stage_normalize_codec, video_path)
        else:
            logger.info(
                "[CODEC] Full-file normalization disabled; detection will use sampled extraction fallback when needed."
            )
        result.normalized_video_path = normalized_path
        _log_codec_telemetry(normalized_path, "analysis_input")

        # ------------------------------------------------------------------ #
        # Stage 4: Preview frame extraction (UI / thumbnails)
        # ------------------------------------------------------------------ #
        result.preview_frame_paths = _time_stage(
            "preview_frames",
            _stage_preview_frames,
            normalized_path,
            options.preview_frames,
            output_dir,
        )

        # ------------------------------------------------------------------ #
        # Stage 5: Projection detection
        # ------------------------------------------------------------------ #
        debug_dir = cfg.debug_output_dir
        detection = _time_stage(
            "detect_projection",
            _stage_detect_projection,
            normalized_path,
            options.num_detection_frames,
            debug_dir,
        )
        result.projection_type = detection.get("projection_type", "unknown")
        result.confidence = float(detection.get("confidence", 0.0))
        raw_stats = detection.get("stats", {})
        if raw_stats:
            result.detector_stats = DetectorStats.from_dict(raw_stats)

        # ------------------------------------------------------------------ #
        # Stage 6: Equirectangular conversion (optional)
        # ------------------------------------------------------------------ #
        converted_path: Optional[str] = None
        if options.convert_if_needed:
            converted_path = _time_stage(
                "convert_equirectangular",
                _stage_convert_to_equirectangular,
                video_path=normalized_path,
                projection_type=result.projection_type,
                confidence=result.confidence,
                output_dir=output_dir,
                confidence_threshold=options.confidence_threshold,
            )
        result.converted_video_path = converted_path
        if converted_path:
            _log_codec_telemetry(converted_path, "converted_output")

        # ------------------------------------------------------------------ #
        # Stage 7: Upload (optional)
        # ------------------------------------------------------------------ #
        if options.upload:
            asset_to_upload = result.final_video_path
            if asset_to_upload and os.path.exists(asset_to_upload):
                upload_title = (
                    options.upload_title
                    or result.source_title
                    or os.path.splitext(os.path.basename(asset_to_upload))[0]
                )
                result.upload_result = _time_stage(
                    "upload",
                    _stage_upload,
                    video_path=asset_to_upload,
                    title=upload_title,
                    description=options.upload_description,
                    playlist=options.upload_playlist,
                    new_playlist=options.upload_new_playlist,
                    category=options.upload_category,
                    tags=options.upload_tags,
                )
            else:
                logger.warning("Upload requested but no valid asset path found; skipping.")

        result.success = True
        if stage_timings:
            logger.info(
                "[TIMING] total=%.3fs detail=%s",
                sum(stage_timings.values()),
                ", ".join(f"{name}:{duration:.3f}s" for name, duration in stage_timings.items()),
            )
        logger.info("Job %s completed successfully.", result.job_id)

    except WorkflowError as exc:
        result.success = False
        result.error = str(exc)
        logger.error("Job %s failed: %s", result.job_id, exc)

    except Exception as exc:
        result.success = False
        result.error = f"Unexpected error: {exc}"
        logger.exception("Job %s encountered an unexpected error.", result.job_id)

    finally:
        # ------------------------------------------------------------------ #
        # Stage 8: Persist job manifest
        # ------------------------------------------------------------------ #
        if options.save_manifest:
            save_job_manifest(result, cfg.jobs_dir)

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper — mirrors the old process_downloaded_video API
# ---------------------------------------------------------------------------

def analyze_video_projection(video_path: str, num_frames: int = 10) -> Dict[str, Any]:
    """Convenience wrapper: run detection only, no download or upload.

    Returns the raw detection dict from the detector pipeline.
    """
    from detector.pipeline import run_detection_pipeline
    return run_detection_pipeline(video_path, num_frames=num_frames)
