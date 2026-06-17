"""YouTube search and metadata client.

Public API
----------
.. code-block:: python

    from core.youtube import search_videos, get_video_thumbnail_urls

    results = search_videos("Mount Everest 360")
    for r in results:
        print(r["title"], r["url"])
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from utils.exceptions import (
    InvalidYouTubeURLError,
    NoYouTubeAPIKeyError,
    YouTubeAPIError,
    YouTubeQuotaError,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key(api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key
    from config.settings import get_settings
    key = get_settings().youtube_api_key
    if not key:
        raise NoYouTubeAPIKeyError("YOUTUBE_API_KEY is not set in the environment.")
    return key


def extract_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from *url*.

    Raises:
        InvalidYouTubeURLError: When the URL cannot be parsed.
    """
    if not url or not isinstance(url, str):
        raise InvalidYouTubeURLError("URL must be a non-empty string.")
    pattern = r"(?:v=|/|embed/|youtu\.be/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise InvalidYouTubeURLError(f"Could not extract a video ID from: {url!r}")
    return match.group(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_videos(
    query: str,
    api_key: Optional[str] = None,
    youtube_client: Any = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search YouTube for 360° videos matching *query*.

    When *query* is a YouTube URL the video is looked up directly.

    Args:
        query: Search text or a YouTube video/watch URL.
        api_key: YouTube Data API v3 key.  Falls back to ``YOUTUBE_API_KEY``
            environment variable.
        youtube_client: Optional pre-built ``googleapiclient`` client (for
            testing).
        max_results: Maximum number of search results to request.

    Returns:
        List of dicts with keys ``id``, ``title``, ``channel``, ``url``.
        Only videos with ``contentDetails.projection == "360"`` are included.

    Raises:
        NoYouTubeAPIKeyError: When no API key is available.
        YouTubeAPIError: On API-level errors.
        YouTubeQuotaError: When the daily quota is exceeded.
    """
    try:
        from googleapiclient.discovery import build  # type: ignore
        from googleapiclient.errors import HttpError  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client is required for YouTube search. "
            "Install it with: pip install google-api-python-client"
        ) from exc

    key = _get_api_key(api_key)

    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string.")

    try:
        client = youtube_client or build("youtube", "v3", developerKey=key)
        all_video_ids: list = []
        _seen_ids: set = set()

        is_url = "youtube.com" in query.lower() or "youtu.be" in query.lower()
        if is_url:
            video_id = extract_video_id(query)
            all_video_ids.append(video_id)
            _seen_ids.add(video_id)
        else:
            search_response = client.search().list(
                part="snippet",
                q=f"{query} 360",
                type="video",
                maxResults=max_results,
            ).execute()

            items = search_response.get("items")
            if items is None:
                raise YouTubeAPIError("Invalid search response: 'items' key missing.")

            for item in items:
                v_id = item.get("id", {}).get("videoId")
                if v_id and v_id not in _seen_ids:
                    all_video_ids.append(v_id)
                    _seen_ids.add(v_id)

        if not all_video_ids:
            return []

        details_response = client.videos().list(
            part="snippet,contentDetails",
            id=",".join(all_video_ids),
        ).execute()

        details_items = details_response.get("items")
        if details_items is None:
            raise YouTubeAPIError("Invalid details response: 'items' key missing.")

        results = []
        for item in details_items:
            content = item.get("contentDetails", {})
            snippet = item.get("snippet", {})
            video_id = item.get("id")
            if content.get("projection") == "360" and video_id:
                results.append({
                    "id": video_id,
                    "title": snippet.get("title"),
                    "channel": snippet.get("channelTitle"),
                    "url": f"https://youtu.be/{video_id}",
                })

        return results

    except ImportError:
        raise
    except (YouTubeAPIError, YouTubeQuotaError, InvalidYouTubeURLError, NoYouTubeAPIKeyError):
        raise
    except Exception as exc:
        try:
            from googleapiclient.errors import HttpError  # type: ignore
            if isinstance(exc, HttpError):
                content_bytes = getattr(exc, "content", b"")
                try:
                    decoded = (
                        content_bytes.decode("utf-8")
                        if isinstance(content_bytes, (bytes, bytearray))
                        else str(content_bytes)
                    )
                except Exception:
                    decoded = str(content_bytes)
                if "quotaExceeded" in decoded or "dailyLimitExceeded" in decoded:
                    raise YouTubeQuotaError("YouTube API quota exceeded.") from exc
                raise YouTubeAPIError(f"YouTube API HTTP error: {exc}") from exc
        except ImportError:
            pass
        raise YouTubeAPIError(f"Unexpected YouTube API error: {exc}") from exc


def get_video_thumbnail_urls(video_id: str) -> Dict[str, str]:
    """Return thumbnail URLs for *video_id* at various quality levels.

    Raises:
        InvalidYouTubeURLError: When *video_id* is not a valid 11-char ID.
    """
    if not video_id or not isinstance(video_id, str) or len(video_id) != 11:
        raise InvalidYouTubeURLError(f"Invalid video ID: {video_id!r}")

    base = f"https://img.youtube.com/vi/{video_id}"
    return {
        "maxresdefault": f"{base}/maxresdefault.jpg",
        "sddefault": f"{base}/sddefault.jpg",
        "hqdefault": f"{base}/hqdefault.jpg",
        "mqdefault": f"{base}/mqdefault.jpg",
        "default": f"{base}/default.jpg",
    }
