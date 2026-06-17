"""Video downloader using yt-dlp.

Public API
----------
.. code-block:: python

    from core.downloader import download_video

    path = download_video("https://youtu.be/XXXXXXXXXXX")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Iterable, Optional

from utils.exceptions import DownloadError


logger = logging.getLogger(__name__)


def _iter_candidate_output_paths(ydl: Any, info: Dict[str, Any]) -> Iterable[str]:
    """Yield candidate output file paths for a completed yt-dlp download."""
    requested_downloads = info.get("requested_downloads")
    if isinstance(requested_downloads, list):
        for entry in requested_downloads:
            if not isinstance(entry, dict):
                continue
            for key in ("filepath", "filename", "_filename"):
                value = entry.get(key)
                if value:
                    yield str(value)

    for key in ("filepath", "filename", "_filename"):
        value = info.get(key)
        if value:
            yield str(value)

    try:
        prepared = ydl.prepare_filename(info)
    except Exception:
        prepared = None

    if prepared:
        yield prepared
        base, _ = os.path.splitext(prepared)
        yield f"{base}.mp4"
        info_mp4 = dict(info)
        info_mp4["ext"] = "mp4"
        try:
            yield ydl.prepare_filename(info_mp4)
        except Exception:
            pass


def _resolve_downloaded_output_path(ydl: Any, info: Dict[str, Any]) -> Optional[str]:
    """Resolve the final output file path emitted by yt-dlp."""
    seen = set()
    for candidate in _iter_candidate_output_paths(ydl, info):
        if not candidate:
            continue
        path = os.path.abspath(candidate)
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    return None


def _collect_output_candidates(ydl: Any, info: Dict[str, Any]) -> list[str]:
    """Collect unique absolute candidate paths for logging/debugging."""
    seen = set()
    paths: list[str] = []
    for candidate in _iter_candidate_output_paths(ydl, info):
        if not candidate:
            continue
        path = os.path.abspath(candidate)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def download_video(
    url: str,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    format_spec: str = "bestvideo+bestaudio/best",
) -> str:
    """Download a video from *url* using yt-dlp.

    Args:
        url: YouTube (or other yt-dlp-supported) URL.
        output_dir: Directory to save the downloaded file.  Defaults to the
            configured downloads directory.
        progress_callback: Optional callable receiving yt-dlp progress hook
            dicts.
        format_spec: yt-dlp format selector string.

    Returns:
        Absolute path to the downloaded video file.

    Raises:
        DownloadError: On any download failure.
    """
    try:
        from yt_dlp import YoutubeDL  # type: ignore
        from yt_dlp import DownloadError as YtDlpDownloadError  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "yt-dlp is required for downloading. Install it with: pip install yt-dlp"
        ) from exc

    if not url or not isinstance(url, str):
        raise ValueError("url must be a non-empty string.")

    if output_dir is None:
        from config.settings import get_settings
        output_dir = get_settings().downloads_dir

    os.makedirs(output_dir, exist_ok=True)

    def _hook(d: dict) -> None:
        if progress_callback:
            progress_callback(d)
        else:
            if d.get("status") == "downloading":
                pct = d.get("_percent_str", "?%").strip()
                logger.debug("Downloading: %s", pct)

    ydl_opts = {
        "format": format_spec,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "retries": 20,
        "fragment_retries": 20,
        "socket_timeout": 60,
        "progress_hooks": [_hook],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = _resolve_downloaded_output_path(ydl, info)
            if not filename or not os.path.exists(filename):
                checked = _collect_output_candidates(ydl, info)
                logger.error("yt-dlp completed but output file was not found. Checked: %s", checked)
                raise DownloadError(
                    f"Downloaded file not found after yt-dlp completed. Checked paths: {checked}"
                )
            logger.info("Downloaded: %s", filename)
            return filename

    except YtDlpDownloadError as exc:
        logger.error("yt-dlp download error: %s", exc)
        raise DownloadError(f"yt-dlp error: {exc}") from exc
    except DownloadError:
        raise
    except Exception as exc:
        logger.exception("Unexpected download error")
        raise DownloadError(f"Unexpected download error: {exc}") from exc
