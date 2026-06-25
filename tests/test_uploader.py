"""Tests for core/uploader.py endpoint construction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import requests

from core.uploader import (
    _build_endpoint,
    _is_disguised_timeout,
    _upload_timeout,
    get_playlists,
    upload_video_asset,
)
from utils.exceptions import MediaCMSError


class BuildEndpointTests(unittest.TestCase):
    """_build_endpoint must resolve URLs under the CMS API namespace."""

    def test_standard_media_path(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists")
        self.assertEqual(result, "https://cms.example.com/api/v1/playlists")

    def test_trailing_slash_on_api_url(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media/", "/playlists")
        self.assertEqual(result, "https://cms.example.com/api/v1/playlists")

    def test_path_without_media_suffix(self):
        """Works even when CMS_API_URL does not end with /media."""
        result = _build_endpoint("https://cms.example.com/v2/videos", "/playlists")
        self.assertEqual(result, "https://cms.example.com/v2/playlists")

    def test_root_path_api_url(self):
        result = _build_endpoint("https://cms.example.com/media", "/playlists/")
        self.assertEqual(result, "https://cms.example.com/playlists/")

    def test_playlist_id_path(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists/abc123")
        self.assertEqual(result, "https://cms.example.com/api/v1/playlists/abc123")

    def test_path_without_leading_slash_is_normalised(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media", "playlists")
        self.assertEqual(result, "https://cms.example.com/api/v1/playlists")

    def test_host_preserved_exactly(self):
        result = _build_endpoint("https://my.cms.internal:8080/api/media", "/playlists")
        self.assertEqual(result, "https://my.cms.internal:8080/api/playlists")

    def test_create_playlist_endpoint(self):
        """create_playlist uses /playlists/ (trailing slash) for POST."""
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists/")
        self.assertTrue(result.endswith("/playlists/"))

    def test_get_playlists_endpoint(self):
        """get_playlists uses /playlists (no trailing slash) for GET."""
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists")
        self.assertTrue(result.endswith("/playlists"))
        self.assertFalse(result.endswith("/playlists/"))

class PlaylistsApiTests(unittest.TestCase):
    @patch("requests.get")
    def test_get_playlists_handles_invalid_json_response(self, mock_get):
        mock_resp = MagicMock(status_code=200, text="<!doctype html>")
        mock_resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = mock_resp

        with self.assertLogs("core.uploader", level="ERROR") as logs:
            playlists = get_playlists(api_url="https://cms.example.com/api/v1/media")

        self.assertEqual(playlists, [])
        mock_get.assert_called_once_with(
            "https://cms.example.com/api/v1/playlists",
            auth=mock_get.call_args.kwargs["auth"],
            timeout=mock_get.call_args.kwargs["timeout"],
        )
        self.assertTrue(any("invalid JSON from https://cms.example.com/api/v1/playlists" in m for m in logs.output))


class UploadMetadataTests(unittest.TestCase):
    def setUp(self):
        self.fake_settings = SimpleNamespace(
            cms_api_url="https://cms.example.com/api/v1/media",
            cms_token=None,
            cms_user=None,
            cms_password=None,
        )

    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_upload_with_tags_includes_tags_field(
        self,
        _mock_exists,
        mock_post,
        mock_get_settings,
    ):
        mock_get_settings.return_value = self.fake_settings
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {
            "friendly_token": "tok-125",
            "media_url": "https://cms.example.com/media/tok-125",
        }
        mock_post.return_value = mock_resp

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="Session Upload",
                description="desc",
                api_url="https://cms.example.com/api/v1/media",
                tags=["anxiety"],
            )

        self.assertTrue(result.success)
        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("data", call_kwargs)
        self.assertEqual(call_kwargs["data"]["tags"], "anxiety")

    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_upload_includes_tags(
        self,
        _mock_exists,
        mock_post,
        mock_get_settings,
    ):
        mock_get_settings.return_value = self.fake_settings
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {
            "friendly_token": "tok-123",
            "media_url": "https://cms.example.com/media/tok-123",
        }
        mock_post.return_value = mock_resp

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="Session Upload",
                description="desc",
                api_url="https://cms.example.com/api/v1/media",
                tags=["anxiety", " breathing ", ""],
            )

        self.assertTrue(result.success)
        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("data", call_kwargs)
        self.assertEqual(call_kwargs["data"]["tags"], "anxiety,breathing")

    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_upload_normalises_tags_and_drops_duplicates(
        self,
        _mock_exists,
        mock_post,
        mock_get_settings,
    ):
        mock_get_settings.return_value = self.fake_settings
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {
            "friendly_token": "tok-124",
            "media_url": "https://cms.example.com/media/tok-124",
        }
        mock_post.return_value = mock_resp

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="Session Upload",
                description="desc",
                api_url="https://cms.example.com/api/v1/media",
                tags=[" anxiety ", "", "anxiety", 123, "123", 12.5, {"bad": "tag"}],
            )

        self.assertTrue(result.success)
        call_kwargs = mock_post.call_args.kwargs
        # int 123 and string "123" collapse to the same final tag after normalisation.
        self.assertEqual(call_kwargs["data"]["tags"], "anxiety,123,12.5")

    @patch("config.settings.get_settings")
    @patch("core.uploader.add_to_playlist")
    @patch("core.uploader.create_playlist")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_upload_with_new_playlist_creates_then_adds(
        self,
        _mock_exists,
        mock_post,
        mock_create_playlist,
        mock_add_to_playlist,
        mock_get_settings,
    ):
        mock_get_settings.return_value = self.fake_settings
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"friendly_token": "tok-999", "media_url": "https://cms/m/tok-999"}
        mock_post.return_value = mock_resp
        mock_create_playlist.return_value = "pl-new"
        mock_add_to_playlist.return_value = True

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="T",
                api_url="https://cms.example.com/api/v1/media",
                new_playlist_name="Playlist X",
            )

        self.assertTrue(result.success)
        mock_create_playlist.assert_called_once_with("Playlist X", api_url="https://cms.example.com/api/v1/media")
        mock_add_to_playlist.assert_called_once_with("tok-999", "pl-new", api_url="https://cms.example.com/api/v1/media")

    @patch("config.settings.get_settings")
    @patch("core.uploader.add_to_playlist")
    @patch("core.uploader.create_playlist")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_upload_with_existing_playlist_skips_new_playlist_creation(
        self,
        _mock_exists,
        mock_post,
        mock_create_playlist,
        mock_add_to_playlist,
        mock_get_settings,
    ):
        mock_get_settings.return_value = self.fake_settings
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"friendly_token": "tok-321", "media_url": "https://cms/m/tok-321"}
        mock_post.return_value = mock_resp
        mock_add_to_playlist.return_value = True

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="T",
                api_url="https://cms.example.com/api/v1/media",
                playlist_id="pl-existing",
                new_playlist_name="Ignored Name",
            )

        self.assertTrue(result.success)
        mock_create_playlist.assert_not_called()
        mock_add_to_playlist.assert_called_once_with("tok-321", "pl-existing", api_url="https://cms.example.com/api/v1/media")


class UploadTimeoutTests(unittest.TestCase):
    """_upload_timeout scales the read timeout with file size."""

    def test_missing_file_falls_back_to_default(self):
        self.assertEqual(_upload_timeout("/no/such/file.mp4"), (10, 300))

    @patch("os.path.getsize", return_value=10 * 1024 * 1024)
    def test_small_file_uses_floor_timeout(self, _mock_getsize):
        self.assertEqual(_upload_timeout("/tmp/small.mp4"), (10, 300))

    @patch("os.path.getsize", return_value=5000 * 1024 * 1024)
    def test_large_file_scales_up(self, _mock_getsize):
        self.assertEqual(_upload_timeout("/tmp/large.mp4"), (10, 3600))


class DisguisedTimeoutTests(unittest.TestCase):
    """A send-phase socket timeout is wrapped by requests as ConnectionError."""

    def test_connection_error_wrapping_timeout_error_is_detected(self):
        protocol_error = Exception("Connection aborted.", TimeoutError("timed out"))
        exc = requests.ConnectionError(protocol_error)
        self.assertTrue(_is_disguised_timeout(exc))

    def test_connection_error_without_timeout_cause_is_not_detected(self):
        protocol_error = Exception("Connection aborted.", ValueError("bad chunk"))
        exc = requests.ConnectionError(protocol_error)
        self.assertFalse(_is_disguised_timeout(exc))

    @patch("core.uploader.time.sleep")
    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_upload_reports_clear_message_on_disguised_timeout(
        self,
        _mock_exists,
        mock_post,
        mock_get_settings,
        _mock_sleep,
    ):
        mock_get_settings.return_value = SimpleNamespace(
            cms_api_url="https://cms.example.com/api/v1/media",
            cms_token=None,
            cms_user=None,
            cms_password=None,
        )
        protocol_error = Exception("Connection aborted.", TimeoutError("timed out"))
        mock_post.side_effect = requests.ConnectionError(protocol_error)

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            with self.assertRaises(MediaCMSError) as ctx:
                upload_video_asset(
                    video_path="/tmp/video.mp4",
                    title="T",
                    api_url="https://cms.example.com/api/v1/media",
                )

        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertEqual(mock_post.call_count, 3)


class UploadRetryTests(unittest.TestCase):
    """Only the verified-safe disguised send-phase timeout is retried."""

    def setUp(self):
        self.fake_settings = SimpleNamespace(
            cms_api_url="https://cms.example.com/api/v1/media",
            cms_token=None,
            cms_user=None,
            cms_password=None,
        )

    @patch("core.uploader.time.sleep")
    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_disguised_timeout_then_success_retries_and_succeeds(
        self, _mock_exists, mock_post, mock_get_settings, mock_sleep,
    ):
        mock_get_settings.return_value = self.fake_settings
        protocol_error = Exception("Connection aborted.", TimeoutError("timed out"))
        success_resp = MagicMock(status_code=201)
        success_resp.json.return_value = {
            "friendly_token": "tok-1",
            "media_url": "https://cms.example.com/m/tok-1",
        }
        mock_post.side_effect = [requests.ConnectionError(protocol_error), success_resp]

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="T",
                api_url="https://cms.example.com/api/v1/media",
            )

        self.assertTrue(result.success)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("core.uploader.time.sleep")
    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_disguised_timeout_exhausts_retries_raises_original_message(
        self, _mock_exists, mock_post, mock_get_settings, _mock_sleep,
    ):
        mock_get_settings.return_value = self.fake_settings
        protocol_error = Exception("Connection aborted.", TimeoutError("timed out"))
        mock_post.side_effect = requests.ConnectionError(protocol_error)

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            with self.assertRaises(MediaCMSError) as ctx:
                upload_video_asset(
                    video_path="/tmp/video.mp4",
                    title="T",
                    api_url="https://cms.example.com/api/v1/media",
                )

        self.assertEqual(
            str(ctx.exception),
            "Upload timed out or the connection was lost mid-transfer. "
            "Check CMS_API_URL, network connectivity, and file size.",
        )
        self.assertEqual(mock_post.call_count, 3)

    @patch("core.uploader.time.sleep")
    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_plain_timeout_is_not_retried(
        self, _mock_exists, mock_post, mock_get_settings, mock_sleep,
    ):
        mock_get_settings.return_value = self.fake_settings
        mock_post.side_effect = requests.Timeout("read timed out")

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            with self.assertRaises(MediaCMSError) as ctx:
                upload_video_asset(
                    video_path="/tmp/video.mp4",
                    title="T",
                    api_url="https://cms.example.com/api/v1/media",
                )

        self.assertEqual(
            str(ctx.exception),
            "Upload timed out. Check CMS_API_URL and network connectivity.",
        )
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("core.uploader.time.sleep")
    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_plain_connection_error_is_not_retried(
        self, _mock_exists, mock_post, mock_get_settings, mock_sleep,
    ):
        mock_get_settings.return_value = self.fake_settings
        protocol_error = Exception("Connection aborted.", ValueError("bad chunk"))
        mock_post.side_effect = requests.ConnectionError(protocol_error)

        with patch("builtins.open", mock_open(read_data=b"video-bytes")):
            with self.assertRaises(MediaCMSError) as ctx:
                upload_video_asset(
                    video_path="/tmp/video.mp4",
                    title="T",
                    api_url="https://cms.example.com/api/v1/media",
                )

        self.assertNotIn("timed out", str(ctx.exception).lower())
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("core.uploader.time.sleep")
    @patch("config.settings.get_settings")
    @patch("requests.post")
    @patch("os.path.exists", return_value=True)
    def test_retry_reopens_file_fresh_each_attempt(
        self, _mock_exists, mock_post, mock_get_settings, _mock_sleep,
    ):
        mock_get_settings.return_value = self.fake_settings
        protocol_error = Exception("Connection aborted.", TimeoutError("timed out"))
        success_resp = MagicMock(status_code=201)
        success_resp.json.return_value = {
            "friendly_token": "tok-2",
            "media_url": "https://cms.example.com/m/tok-2",
        }
        mock_post.side_effect = [requests.ConnectionError(protocol_error), success_resp]

        m_open = mock_open(read_data=b"video-bytes")
        with patch("builtins.open", m_open):
            result = upload_video_asset(
                video_path="/tmp/video.mp4",
                title="T",
                api_url="https://cms.example.com/api/v1/media",
            )

        self.assertTrue(result.success)
        self.assertEqual(m_open.call_count, 2)


if __name__ == "__main__":
    unittest.main()
