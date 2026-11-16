"""Detector package for 360-video projection analysis.

Public API
----------
.. code-block:: python

    from detector import analyze_video_projection, convert_to_equirectangular

    result = analyze_video_projection("path/to/video.mp4")
    # result["projection_type"] -> "equirectangular" | "eac" | "cubic" | "stereo_equi" | "unknown"

    if result["projection_type"] not in ("equirectangular", "stereo_equi", "unknown"):
        out_path = convert_to_equirectangular(
            "path/to/video.mp4",
            result["projection_type"],
        )
"""

from .pipeline import run_detection_pipeline
from .video_io import convert_video_codec, extract_main_frames, FrameExtractorError
from .projection_conversion import convert_detected_projection_to_equirectangular


def analyze_video_projection(video_path: str, num_frames: int = 10, **kwargs):
    """Run the full projection-detection pipeline on *video_path*.

    This is the canonical entry-point for the detector.  All keyword
    arguments are forwarded to :func:`detector.pipeline.run_detection_pipeline`.

    Returns a dict with keys:
        ``projection_type``, ``confidence``, ``frames_analyzed``,
        ``frames_with_line``, ``motion_reliable``, ``stats``, and others.
    """
    return run_detection_pipeline(video_path, num_frames=num_frames, **kwargs)


def convert_to_equirectangular(
    video_path: str,
    projection_type: str,
    output_dir=None,
):
    """Convert *video_path* to equirectangular if the projection requires it.

    Thin wrapper over the full conversion logic in
    :func:`detector.projection_conversion.convert_detected_projection_to_equirectangular`.

    Returns the path to the converted file on success, or *video_path*
    unchanged when conversion is not needed or fails.
    """
    result = convert_detected_projection_to_equirectangular(
        video_path=video_path,
        projection_type=projection_type,
        output_dir=output_dir,
    )
    if result.get("success") and result.get("output_path"):
        return result["output_path"]
    return video_path


__all__ = [
    "analyze_video_projection",
    "convert_to_equirectangular",
    "run_detection_pipeline",
    "convert_video_codec",
    "extract_main_frames",
    "FrameExtractorError",
    "convert_detected_projection_to_equirectangular",
]
