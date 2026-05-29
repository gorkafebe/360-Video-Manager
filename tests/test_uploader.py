"""Tests for core/uploader.py endpoint construction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

from core.uploader import (
    _build_endpoint,
    create_category,
    get_categories,
    get_playlists,
    upload_video_asset,
)


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

    def test_get_categories_endpoint(self):
        """get_categories uses /categories (no trailing slash) for GET."""
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/categories")
        self.assertTrue(result.endswith("/categories"))
        self.assertFalse(result.endswith("/categories/"))

    def test_create_category_endpoint(self):
        """create_category uses /categories/ (trailing slash) for POST."""
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/categories/")
        self.assertTrue(result.endswith("/categories/"))


class CategoriesApiTests(unittest.TestCase):
    @patch("requests.get")
    def test_get_categories_returns_results_list(self, mock_get):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"results": [{"id": "c1", "title": "Patient A"}]}
        mock_get.return_value = mock_resp

        categories = get_categories(api_url="https://cms.example.com/api/v1/media")
        self.assertEqual(categories, [{"id": "c1", "title": "Patient A"}])
        mock_get.assert_called_once_with(
            "https://cms.example.com/api/v1/categories",
            auth=mock_get.call_args.kwargs["auth"],
            timeout=mock_get.call_args.kwargs["timeout"],
        )

    @patch("requests.get")
    def test_get_categories_handles_invalid_json_response(self, mock_get):
        mock_resp = MagicMock(status_code=200, text="<html>not-json</html>")
        mock_resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = mock_resp

        with self.assertLogs("core.uploader", level="ERROR") as logs:
            categories = get_categories(api_url="https://cms.example.com/api/v1/media")

        self.assertEqual(categories, [])
        mock_get.assert_called_once_with(
            "https://cms.example.com/api/v1/categories",
            auth=mock_get.call_args.kwargs["auth"],
            timeout=mock_get.call_args.kwargs["timeout"],
        )
        self.assertTrue(any("invalid JSON from https://cms.example.com/api/v1/categories" in m for m in logs.output))

    @patch("requests.post")
    def test_create_category_fallbacks_to_plural_without_trailing_slash_on_404(self, mock_post):
        api_url = "https://cms.example.com/api/v1/media"
        first = MagicMock(status_code=404, text="not found")
        second = MagicMock(status_code=201, text='{"id":"cat-9"}')
        second.json.return_value = {"id": "cat-9"}
        mock_post.side_effect = [first, second]

        category_id = create_category("Test Category 9", api_url=api_url)

        self.assertEqual(category_id, "cat-9")
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_post.call_args_list[0].args[0], _build_endpoint(api_url, "/categories/"))
        self.assertEqual(mock_post.call_args_list[1].args[0], _build_endpoint(api_url, "/categories"))

    @patch("requests.post")
    def test_create_category_fallbacks_to_singular_endpoint_after_plural_variants(self, mock_post):
        api_url = "https://cms.example.com/api/v1/media"
        first = MagicMock(status_code=404, text="not found")
        second = MagicMock(status_code=404, text="still not found")
        third = MagicMock(status_code=201, text='{"id":"cat-10"}')
        third.json.return_value = {"id": "cat-10"}
        mock_post.side_effect = [first, second, third]

        category_id = create_category("Test Category 10", api_url=api_url)

        self.assertEqual(category_id, "cat-10")
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_post.call_args_list[0].args[0], _build_endpoint(api_url, "/categories/"))
        self.assertEqual(mock_post.call_args_list[1].args[0], _build_endpoint(api_url, "/categories"))
        self.assertEqual(mock_post.call_args_list[2].args[0], _build_endpoint(api_url, "/category/"))

    @patch("requests.post")
    def test_create_category_logs_status_and_body_preview(self, mock_post):
        mock_resp = MagicMock(status_code=403, text="forbidden body")
        mock_post.return_value = mock_resp

        with self.assertLogs("core.uploader", level="WARNING") as logs:
            category_id = create_category("Test Category 10", api_url="https://cms.example.com/api/v1/media")

        self.assertIsNone(category_id)
        self.assertTrue(any("create_category: status 403" in m for m in logs.output))
        self.assertTrue(any("forbidden body" in m for m in logs.output))


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
    def test_upload_includes_category_and_tags(
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
                category_id="cat-1",
                tags=["anxiety", " breathing ", ""],
            )

        self.assertTrue(result.success)
        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("data", call_kwargs)
        self.assertEqual(call_kwargs["data"]["category"], "cat-1")
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


if __name__ == "__main__":
    unittest.main()
