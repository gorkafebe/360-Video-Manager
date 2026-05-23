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
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse

from utils.exceptions import MediaCMSError
from core.models import UploadResult


logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 300
_REQUEST_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)


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

    Strips the path component of *api_url* down to the scheme+host so that
    the result is independent of whether *api_url* ends with ``/media``,
    ``/api/v1/media``, or any other suffix.

    Example::

        _build_endpoint("https://cms.example.com/api/v1/media", "/playlists/")
        # -> "https://cms.example.com/playlists/"
    """
    parsed = urlparse(api_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    # Ensure path starts with a slash for urljoin to work correctly.
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def get_playlists(api_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the list of existing MediaCMS playlists."""
    import requests  # type: ignore
    url = _get_api_url(api_url)
    endpoint = _build_endpoint(url, "/playlists")
    auth = _get_auth()
    try:
        resp = requests.get(endpoint, auth=auth, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", data) if isinstance(data, dict) else data
        logger.warning("get_playlists: unexpected status %s", resp.status_code)
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


def get_categories(api_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the list of existing MediaCMS categories."""
    import requests  # type: ignore
    url = _get_api_url(api_url)
    endpoint = _build_endpoint(url, "/categories")
    auth = _get_auth()
    try:
        resp = requests.get(endpoint, auth=auth, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", data) if isinstance(data, dict) else data
        logger.warning("get_categories: unexpected status %s", resp.status_code)
        return []
    except requests.Timeout:
        logger.error("Timeout fetching categories.")
        return []
    except Exception as exc:
        logger.error("Error fetching categories: %s", exc)
        return []


def create_category(title: str, api_url: Optional[str] = None) -> Optional[str]:
    """Create a new category and return its ID, or None on failure."""
    import requests  # type: ignore
    url = _get_api_url(api_url)
    endpoint = _build_endpoint(url, "/categories/")
    auth = _get_auth()
    try:
        resp = requests.post(
            endpoint,
            json={"title": title},
            auth=auth,
            timeout=_REQUEST_TIMEOUT,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("id")
        logger.warning("create_category: status %s", resp.status_code)
        return None
    except requests.Timeout:
        logger.error("Timeout creating category.")
        return None
    except Exception as exc:
        logger.error("Error creating category: %s", exc)
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
    category_id: Optional[str] = None,
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
        category_id: Optional MediaCMS category ID for metadata assignment.
        tags: Optional personalized tags to assign to the uploaded media.

    Returns:
        :class:`~core.models.UploadResult` describing the outcome.

    Raises:
        MediaCMSError: On unrecoverable upload failures.
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
    if category_id:
        data["category"] = str(category_id)
    clean_tags = [t.strip() for t in (tags or []) if str(t).strip()]
    if clean_tags:
        data["tags"] = ",".join(clean_tags)

    try:
        with open(video_path, "rb") as fh:
            files = {"media_file": (os.path.basename(video_path), fh, "video/mp4")}
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                auth=auth,
                timeout=_REQUEST_TIMEOUT,
            )

        if response.status_code not in (200, 201):
            return UploadResult(
                success=False,
                status_code=response.status_code,
                error=f"Upload failed with status {response.status_code}: {response.text[:200]}",
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
    except MediaCMSError:
        raise
    except Exception as exc:
        logger.exception("Critical error in upload_video_asset")
        raise MediaCMSError(f"Upload error: {exc}") from exc


# Backward-compatible aliases
post_video = upload_video_asset
