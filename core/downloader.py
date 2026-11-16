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
import shutil
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from utils.exceptions import DownloadError


logger = logging.getLogger(__name__)


# yt-dlp supports these external JavaScript runtimes; it only auto-enables `deno`.
# Order = preference. `deno` first because it is yt-dlp's default and best-tested;
# `node`/`bun` are honoured when present. Solving YouTube's `n`-signature / PO
# tokens needs one of these — without it yt-dlp falls back to a degraded client
# (e.g. android_vr) whose media URLs YouTube rejects mid-download with HTTP 403.
_JS_RUNTIME_CANDIDATES: Tuple[str, ...] = ("deno", "node", "bun")

# Extra locations to probe beyond PATH — user-local installs are common and are
# exactly the case that bites here (node present under ~/.local but not on PATH).
_JS_RUNTIME_EXTRA_DIRS: Tuple[str, ...] = (
    os.path.expanduser("~/.deno/bin"),
    os.path.expanduser("~/.local/bin"),
    os.path.expanduser("~/.local/node/bin"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
)

# HTTP 403 mid-download is almost always YouTube throttling a range request.
# Downloading in bounded chunks lets yt-dlp re-request the failing range (backed
# by `retries`/`fragment_retries`) instead of the whole transfer dying.
_HTTP_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MiB


def _find_runtime_binary(name: str) -> Optional[str]:
    """Return a usable path to JS runtime *name*, searching PATH then known dirs."""
    found = shutil.which(name)
    if found:
        return found
    exe = f"{name}.exe" if os.name == "nt" else name
    for directory in _JS_RUNTIME_EXTRA_DIRS:
        candidate = os.path.join(directory, exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def discover_js_runtimes() -> Dict[str, Dict[str, Any]]:
    """Build a yt-dlp ``js_runtimes`` mapping from whatever runtimes are installed.

    Returns a dict like ``{"node": {"path": "/home/u/.local/node/bin/node"}}`` in
    yt-dlp's expected ``{runtime: {config}}`` shape, or ``{}`` when none is found.
    A runtime already on PATH is passed without an explicit path so yt-dlp resolves
    it itself; one found only in a known extra directory is pinned by path.
    """
    runtimes: Dict[str, Dict[str, Any]] = {}
    for name in _JS_RUNTIME_CANDIDATES:
        on_path = shutil.which(name)
        if on_path:
            runtimes[name] = {}
            continue
        located = _find_runtime_binary(name)
        if located:
            runtimes[name] = {"path": located}
    return runtimes


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
        # Re-request bounded ranges rather than letting one 403 kill the whole
        # transfer; combined with retries this recovers from YouTube throttling.
        "http_chunk_size": _HTTP_CHUNK_SIZE,
        "socket_timeout": 60,
        "progress_hooks": [_hook],
    }

    # Enable a JavaScript runtime when one is available. Without it, modern
    # YouTube extraction is degraded (yt-dlp deprecation warning) and the fallback
    # client's URLs 403 partway through a download. yt-dlp only auto-enables deno,
    # so node/bun installs — and user-local deno not on PATH — need to be passed
    # explicitly here.
    js_runtimes = discover_js_runtimes()
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes
        logger.info("Enabled JavaScript runtime(s) for yt-dlp: %s", ", ".join(sorted(js_runtimes)))
    else:
        logger.warning(
            "No JavaScript runtime (deno/node/bun) found for yt-dlp. YouTube "
            "extraction is degraded and downloads may fail with HTTP 403 partway "
            "through. Install deno (https://deno.land) or node and ensure it is on "
            "PATH."
        )

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
        message = f"yt-dlp error: {exc}"
        if "403" in str(exc) or "Forbidden" in str(exc):
            hint = (
                "HTTP 403 mid-download usually means YouTube throttled the format "
                "or yt-dlp is using a degraded client. "
            )
            if not js_runtimes:
                hint += "No JavaScript runtime was available — install deno/node. "
            hint += "Updating yt-dlp (pip install -U yt-dlp) also often resolves this."
            logger.error(hint)
            message = f"{message}\n{hint}"
        raise DownloadError(message) from exc
    except DownloadError:
        raise
    except Exception as exc:
        logger.exception("Unexpected download error")
        raise DownloadError(f"Unexpected download error: {exc}") from exc
