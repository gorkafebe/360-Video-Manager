"""Data models for the 360-Video-Manager pipeline.

:class:`JobResult` is the canonical result type returned by the unified
processing pipeline.  All fields are optional (None) when the corresponding
pipeline stage was skipped or failed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class DetectorStats:
    """Statistics returned by the projection-detection pipeline."""

    total_frames_extracted: int = 0
    black_frames: int = 0
    valid_frames_used: int = 0
    discarded_frames: int = 0
    pairs_total: int = 0
    pairs_valid: int = 0
    pairs_invalid: int = 0
    regions_valid: int = 0
    regions_invalid: int = 0
    avg_eac_score: str = "N/A"
    avg_cubic_score: str = "N/A"
    score_margin: str = "N/A"
    final_classification: str = "unknown"
    final_confidence: float = 0.0
    equi_usable_frames: int = 0
    equi_strong_frames: int = 0
    equi_mean_score: str = "N/A"
    equi_median_score: str = "N/A"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DetectorStats":
        """Build a :class:`DetectorStats` from a raw stats dict."""
        return cls(
            total_frames_extracted=int(d.get("total_frames_extracted", 0)),
            black_frames=int(d.get("black_frames", 0)),
            valid_frames_used=int(d.get("valid_frames_used", 0)),
            discarded_frames=int(d.get("discarded_frames", 0)),
            pairs_total=int(d.get("pairs_total", 0)),
            pairs_valid=int(d.get("pairs_valid", 0)),
            pairs_invalid=int(d.get("pairs_invalid", 0)),
            regions_valid=int(d.get("regions_valid", 0)),
            regions_invalid=int(d.get("regions_invalid", 0)),
            avg_eac_score=str(d.get("avg_eac_score", "N/A")),
            avg_cubic_score=str(d.get("avg_cubic_score", "N/A")),
            score_margin=str(d.get("score_margin", "N/A")),
            final_classification=str(d.get("final_classification", "unknown")),
            final_confidence=float(d.get("final_confidence", 0.0)),
            equi_usable_frames=int(d.get("equi_usable_frames", 0)),
            equi_strong_frames=int(d.get("equi_strong_frames", 0)),
            equi_mean_score=str(d.get("equi_mean_score", "N/A")),
            equi_median_score=str(d.get("equi_median_score", "N/A")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class UploadResult:
    """Result of a MediaCMS upload operation."""

    success: bool = False
    status_code: Optional[int] = None
    media_url: Optional[str] = None
    friendly_token: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class JobResult:
    """Canonical result for a single video-processing job.

    Fields
    ------
    success
        True only when the entire pipeline (download → upload) completed
        without fatal errors.
    source_url
        Original YouTube URL or search query.
    source_title
        Title of the source video as returned by the YouTube API.
    original_video_path
        Path to the downloaded video file.
    normalized_video_path
        Path to the codec-normalised video used as the *analysis* input.  Equals
        *original_video_path* when no compatibility transcode was needed; when it
        differs it is a temporary, audio-stripped intermediate and is never
        delivered to the user.
    conversion_skipped_reason
        Why equirectangular conversion did not produce a file, or ``None`` when
        it did.  ``"projection_unknown"`` means the detector could not determine
        the source geometry, so the video was deliberately left untouched.
    projection_type
        One of ``"equirectangular"``, ``"stereo_equi"``, ``"eac"``,
        ``"cubic"``, ``"unknown"``.
    confidence
        Detection confidence in [0, 1].
    converted_video_path
        Path to the equirectangular-converted video, or ``None`` when
        conversion was not performed.
    preview_frame_paths
        Paths to thumbnail/preview frames extracted for the UI.
    detector_stats
        Structured stats from the detection pipeline.
    upload_result
        Result of the MediaCMS upload, or ``None`` when upload was skipped.
    error
        Error message when *success* is False.
    job_id
        Unique run identifier (timestamp-based).
    manifest_path
        Path where the JSON manifest was saved.
    """

    success: bool = False
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    original_video_path: Optional[str] = None
    normalized_video_path: Optional[str] = None
    projection_type: str = "unknown"
    confidence: float = 0.0
    converted_video_path: Optional[str] = None
    conversion_skipped_reason: Optional[str] = None
    preview_frame_paths: List[str] = dataclasses.field(default_factory=list)
    detector_stats: Optional[DetectorStats] = None
    upload_result: Optional[UploadResult] = None
    error: Optional[str] = None
    job_id: Optional[str] = None
    manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @property
    def final_video_path(self) -> Optional[str]:
        """The video to upload / offer to the user.

        Priority: converted > original.

        ``normalized_video_path`` is deliberately **not** a candidate. When it
        differs from the original it is a throwaway compatibility transcode made
        only so OpenCV could decode the source for analysis: re-encoded at CRF 23
        and stripped of audio (``-an``). Delivering that instead of the source
        would silently ship the user a degraded, silent copy of their video.
        """
        return self.converted_video_path or self.original_video_path
