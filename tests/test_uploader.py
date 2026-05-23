"""Tests for core/uploader.py endpoint construction."""

from __future__ import annotations

import unittest

from core.uploader import _build_endpoint


class BuildEndpointTests(unittest.TestCase):
    """_build_endpoint must produce correct absolute URLs regardless of api_url shape."""

    def test_standard_media_path(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists")
        self.assertEqual(result, "https://cms.example.com/playlists")

    def test_trailing_slash_on_api_url(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media/", "/playlists")
        self.assertEqual(result, "https://cms.example.com/playlists")

    def test_path_without_media_suffix(self):
        """Works even when CMS_API_URL does not end with /media."""
        result = _build_endpoint("https://cms.example.com/v2/videos", "/playlists")
        self.assertEqual(result, "https://cms.example.com/playlists")

    def test_root_path_api_url(self):
        result = _build_endpoint("https://cms.example.com/media", "/playlists/")
        self.assertEqual(result, "https://cms.example.com/playlists/")

    def test_playlist_id_path(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists/abc123")
        self.assertEqual(result, "https://cms.example.com/playlists/abc123")

    def test_path_without_leading_slash_is_normalised(self):
        result = _build_endpoint("https://cms.example.com/api/v1/media", "playlists")
        self.assertEqual(result, "https://cms.example.com/playlists")

    def test_host_preserved_exactly(self):
        result = _build_endpoint("https://my.cms.internal:8080/api/media", "/playlists")
        self.assertEqual(result, "https://my.cms.internal:8080/playlists")

    def test_create_playlist_endpoint(self):
        """create_playlist uses /playlists/ (trailing slash) for POST."""
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists/")
        self.assertTrue(result.endswith("/playlists/"))

    def test_get_playlists_endpoint(self):
        """get_playlists uses /playlists (no trailing slash) for GET."""
        result = _build_endpoint("https://cms.example.com/api/v1/media", "/playlists")
        self.assertTrue(result.endswith("/playlists"))
        self.assertFalse(result.endswith("/playlists/"))


if __name__ == "__main__":
    unittest.main()
