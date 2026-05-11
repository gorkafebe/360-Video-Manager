"""Application exception hierarchy for 360-Video-Manager.

All custom exceptions live here so they can be imported from a single
location regardless of which subsystem raises them.
"""


# ---------------------------------------------------------------------------
# YouTube / search errors
# ---------------------------------------------------------------------------

class YouTubeError(Exception):
    """Base exception for YouTube-related errors."""


class NoYouTubeAPIKeyError(YouTubeError):
    """Raised when the YouTube API key is missing from the environment."""


class InvalidYouTubeURLError(YouTubeError):
    """Raised when a URL cannot be parsed to extract a YouTube video ID."""

    # Backward-compatible alias kept for legacy callers.
    pass


# Alias used in older code
InvalidYouTubeURLException = InvalidYouTubeURLError


class YouTubeAPIError(YouTubeError):
    """General YouTube Data API error."""


class YouTubeQuotaError(YouTubeAPIError):
    """Raised when the YouTube API quota has been exceeded."""


# ---------------------------------------------------------------------------
# Download errors
# ---------------------------------------------------------------------------

class DownloadError(Exception):
    """Raised when a video download fails."""


# Backward-compatible alias
DownloadErrorCustom = DownloadError


# ---------------------------------------------------------------------------
# Video processing / frame extraction errors
# ---------------------------------------------------------------------------

class VideoProcessingError(Exception):
    """Base exception for video-processing errors."""


class FrameExtractorError(VideoProcessingError):
    """Raised when frame extraction or codec conversion fails."""


class VideoAnalysisError(VideoProcessingError):
    """Raised when video projection analysis fails."""


# ---------------------------------------------------------------------------
# Upload / MediaCMS errors
# ---------------------------------------------------------------------------

class MediaCMSError(Exception):
    """Raised when a MediaCMS API operation fails."""


# ---------------------------------------------------------------------------
# Workflow / orchestration errors
# ---------------------------------------------------------------------------

class WorkflowError(Exception):
    """Raised when the unified pipeline encounters an unrecoverable error."""
