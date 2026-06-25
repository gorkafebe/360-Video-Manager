"""MediaCMS upload client.

Public API
----------
.. code-block:: python

    from core.uploader import upload_video_asset

    result = upload_video_asset(
        video_path="output.mp4",
        title="My 360 Video",
        description="...",
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse

from utils.exceptions import MediaCMSError
from core.models import UploadResult


logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 300
_REQUEST_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
_RESPONSE_PREVIEW_MAX_LENGTH = 200

# A fixed read timeout is too short for large video uploads on slow
# connections (the body can still be sending when it fires) yet needlessly
# long for small ones, so the upload read timeout scales with file size.
_UPLOAD_SECONDS_PER_MB = 2  # ~0.5 MB/s minimum assumed upload throughput
_UPLOAD_MAX_READ_TIMEOUT = 3600

_UPLOAD_RETRY_MAX_ATTEMPTS = 3       # total attempts including the first (2 retries)
_UPLOAD_RETRY_BACKOFF_SECONDS = 2.0  # base delay; doubles each retry (2s, 4s)


def _upload_timeout(video_path: str) -> tuple[int, int]:
    """Return a (connect, read) timeout scaled to *video_path*'s size."""
    try:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
    except OSError:
        return _REQUEST_TIMEOUT
    read_timeout = min(
        _UPLOAD_MAX_READ_TIMEOUT,
        max(_READ_TIMEOUT, int(size_mb * _UPLOAD_SECONDS_PER_MB)),
    )
    return (_CONNECT_TIMEOUT, read_timeout)


def _is_disguised_timeout(exc: Exception) -> bool:
    """Detect a send-phase socket timeout hiding inside a ConnectionError.

    urllib3 only maps a socket timeout to ``requests.Timeout`` when it occurs
    while waiting for a response. A timeout while still pushing the upload
    body (common for large files on slow connections) is instead wrapped as
    a generic ConnectionError whose inner cause is a TimeoutError.
    """
    cause = exc.args[0] if exc.args else None
    inner_args = getattr(cause, "args", ())
    return any(isinstance(arg, TimeoutError) for arg in inner_args)


def _get_auth():
    """Return HTTPBasicAuth from settings, or None when credentials are missing."""
    try:
        from requests.auth import HTTPBasicAuth  # type: ignore
    except ImportError:
        return None
    from config.settings import get_settings
    cfg = get_settings()
    if cfg.cms_user and cfg.cms_password:
        return HTTPBasicAuth(cfg.cms_user, cfg.cms_password)
    return None


def _get_api_url(api_url: Optional[str] = None) -> str:
    if api_url:
        return api_url
    from config.settings import get_settings
    url = get_settings().cms_api_url
    if not url:
        raise MediaCMSError(
            "CMS_API_URL is not set. "
            "Set it in your .env file or environment variables."
        )
    return url


def _build_endpoint(api_url: str, path: str) -> str:
    """Build an absolute endpoint URL from *api_url* and a *path* suffix.

    Related endpoints are resolved under the same API namespace as *api_url*.
    For example, when ``api_url`` is ``.../api/v1/media``, ``/playlists`` is
    resolved as ``.../api/v1/playlists``.

    Example::

        _build_endpoint("https://cms.example.com/api/v1/media", "/playlists/")
        # -> "https://cms.example.com/api/v1/playlists/"
    """
    parsed = urlparse(api_url)
    api_path = parsed.path.rstrip("/")
    namespace_path = api_path.rsplit("/", 1)[0] if "/" in api_path else ""
    base = f"{parsed.scheme}://{parsed.netloc}{namespace_path}"
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _safe_json_response(resp, endpoint: str, operation: str):
    """Parse JSON response and log diagnostics if parsing fails."""
    try:
        return resp.json()
    except ValueError as exc:
        preview = (resp.text or "")[:_RESPONSE_PREVIEW_MAX_LENGTH]
        logger.error(
            "%s: invalid JSON from %s (status=%s, body=%r): %s",
            operation,
            endpoint,
            resp.status_code,
            preview,
            exc,
        )
        return None


def _normalise_tags(tags: Optional[List[Any]]) -> List[str]:
    """Return cleaned upload tags with duplicates and blanks removed."""
    clean: List[str] = []
    seen = set()
    for raw in tags or []:
        if not isinstance(raw, (str, int, float)):
            logger.debug("Ignoring non-primitive tag value: %r", raw)
            continue
        tag = str(raw).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        clean.append(tag)
    return clean


def get_playlists(api_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the list of existing MediaCMS playlists."""
    import requests  # type: ignore
    url = _get_api_url(api_url)
    endpoint = _build_endpoint(url, "/playlists")
    auth = _get_auth()
    try:
        resp = requests.get(endpoint, auth=auth, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = _safe_json_response(resp, endpoint, "get_playlists")
            if data is None:
                return []
            return data.get("results", data) if isinstance(data, dict) else data
        logger.warning(
            "get_playlists: unexpected status %s from %s body=%r",
            resp.status_code,
            endpoint,
            (resp.text or "")[:_RESPONSE_PREVIEW_MAX_LENGTH],
        )
        return []
    except requests.Timeout:
        logger.error("Timeout fetching playlists.")
        return []
    except Exception as exc:
        logger.error("Error fetching playlists: %s", exc)
        return []


def create_playlist(title: str, api_url: Optional[str] = None) -> Optional[str]:
    """Create a new public playlist and return its ID, or None on failure."""
    import requests  # type: ignore
    url = _get_api_url(api_url)
    endpoint = _build_endpoint(url, "/playlists/")
    auth = _get_auth()
    try:
        resp = requests.post(
            endpoint,
            json={"title": title, "privacy": "public"},
            auth=auth,
            timeout=_REQUEST_TIMEOUT,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("id")
        logger.warning("create_playlist: status %s", resp.status_code)
        return None
    except requests.Timeout:
        logger.error("Timeout creating playlist.")
        return None
    except Exception as exc:
        logger.error("Error creating playlist: %s", exc)
        return None


def add_to_playlist(
    video_token: str,
    playlist_id: str,
    api_url: Optional[str] = None,
) -> bool:
    """Add *video_token* to *playlist_id*.  Returns True on success."""
    import requests  # type: ignore
    url = _get_api_url(api_url)
    auth = _get_auth()
    endpoint = _build_endpoint(url, f"/playlists/{playlist_id}")
    try:
        resp = requests.put(
            endpoint,
            json={"type": "add", "media_friendly_token": video_token},
            auth=auth,
            timeout=_REQUEST_TIMEOUT,
        )
        logger.info("add_to_playlist: status %s", resp.status_code)
        return resp.status_code in (200, 201, 204)
    except Exception as exc:
        logger.error("Error adding to playlist: %s", exc)
        return False


def upload_video_asset(
    video_path: str,
    title: str,
    description: str = "",
    api_url: Optional[str] = None,
    playlist_id: Optional[str] = None,
    new_playlist_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> UploadResult:
    """Upload *video_path* to MediaCMS and optionally add it to a playlist.

    Args:
        video_path: Absolute path to the video file.
        title: Title to use in MediaCMS.
        description: Optional description.
        api_url: MediaCMS API endpoint.  Falls back to ``CMS_API_URL`` env var.
        playlist_id: Existing playlist identifier to add the video to.
        new_playlist_name: When set, create a new playlist with this name and
            add the video to it.
        tags: Optional personalized tags to assign to the uploaded media.

    Returns:
        :class:`~core.models.UploadResult` describing the outcome.

    Raises:
        MediaCMSError: On unrecoverable upload failures.  A send-phase
            socket timeout (the body never fully reached the server) is
            retried internally before raising.
    """
    import requests  # type: ignore
    if not video_path or not os.path.exists(video_path):
        return UploadResult(
            success=False, error=f"Video file not found: {video_path}"
        )

    url = _get_api_url(api_url)
    auth = _get_auth()

    from config.settings import get_settings
    token = get_settings().cms_token
    headers = {
        "accept": "application/json",
        **({"X-CSRFToken": token} if token else {}),
    }
    data = {"title": title, "description": description}
    clean_tags = _normalise_tags(tags)
    if clean_tags:
        data["tags"] = ",".join(clean_tags)

    try:
        for attempt in range(1, _UPLOAD_RETRY_MAX_ATTEMPTS + 1):
            try:
                with open(video_path, "rb") as fh:
                    files = {"media_file": (os.path.basename(video_path), fh, "video/mp4")}
                    response = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data,
                        auth=auth,
                        timeout=_upload_timeout(video_path),
                    )
                break
            except requests.ConnectionError as exc:
                if not _is_disguised_timeout(exc) or attempt >= _UPLOAD_RETRY_MAX_ATTEMPTS:
                    raise
                delay = _UPLOAD_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Upload attempt %d/%d hit a disguised send-phase timeout; retrying in %.0fs.",
                    attempt, _UPLOAD_RETRY_MAX_ATTEMPTS, delay,
                )
                time.sleep(delay)

        if response.status_code not in (200, 201):
            return UploadResult(
                success=False,
                status_code=response.status_code,
                error=f"Upload failed with status {response.status_code}: {response.text[:_RESPONSE_PREVIEW_MAX_LENGTH]}",
            )

        resp_json = response.json()
        friendly_token = resp_json.get("friendly_token")
        media_url = resp_json.get("url") or resp_json.get("media_url")

        # Handle playlist assignment
        target_playlist_id = playlist_id
        if new_playlist_name and not target_playlist_id:
            target_playlist_id = create_playlist(new_playlist_name, api_url=url)

        if friendly_token and target_playlist_id:
            add_to_playlist(friendly_token, target_playlist_id, api_url=url)

        logger.info("Uploaded: %s → token=%s", title, friendly_token)
        return UploadResult(
            success=True,
            status_code=response.status_code,
            media_url=media_url,
            friendly_token=friendly_token,
        )

    except requests.Timeout as exc:
        raise MediaCMSError(
            "Upload timed out. Check CMS_API_URL and network connectivity."
        ) from exc
    except requests.ConnectionError as exc:
        if _is_disguised_timeout(exc):
            raise MediaCMSError(
                "Upload timed out or the connection was lost mid-transfer. "
                "Check CMS_API_URL, network connectivity, and file size."
            ) from exc
        raise MediaCMSError(f"Upload error: {exc}") from exc
    except MediaCMSError:
        raise
    except Exception as exc:
        logger.exception("Critical error in upload_video_asset")
        raise MediaCMSError(f"Upload error: {exc}") from exc
