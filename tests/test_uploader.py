from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import requests

from core.uploader import upload_video_asset
from utils.exceptions import MediaCMSError


class UploadVideoAssetTests(unittest.TestCase):
    def test_uses_configured_upload_timeout(self):
        fake_cfg = SimpleNamespace(
            cms_token="token",
            cms_upload_connect_timeout=180,
            cms_upload_read_timeout=2400,
        )
        fake_response = MagicMock()
        fake_response.status_code = 201
        fake_response.json.return_value = {"friendly_token": "abc", "url": "https://cms/media/abc"}

        with (
            patch("core.uploader._get_api_url", return_value="https://cms/api/v1/media"),
            patch("core.uploader._get_auth", return_value=None),
            patch("config.settings.get_settings", return_value=fake_cfg),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=b"video-bytes")),
            patch("requests.post", return_value=fake_response) as mock_post,
        ):
            result = upload_video_asset("/tmp/video.mp4", "title")

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 201)
        self.assertEqual(mock_post.call_args.kwargs["timeout"], (180, 2400))

    def test_connection_error_raises_clear_message(self):
        fake_cfg = SimpleNamespace(
            cms_token=None,
            cms_upload_connect_timeout=120,
            cms_upload_read_timeout=1800,
        )

        with (
            patch("core.uploader._get_api_url", return_value="https://cms/api/v1/media"),
            patch("core.uploader._get_auth", return_value=None),
            patch("config.settings.get_settings", return_value=fake_cfg),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=b"video-bytes")),
            patch("requests.post", side_effect=requests.ConnectionError("timed out")),
        ):
            with self.assertRaises(MediaCMSError) as ctx:
                upload_video_asset("/tmp/video.mp4", "title")

        self.assertIn("CMS_UPLOAD_CONNECT_TIMEOUT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
