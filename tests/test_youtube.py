"""Tests for core/youtube.py: ordered de-duplication and 360° filtering."""

from __future__ import annotations

import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

from core.youtube import search_videos, extract_video_id
from utils.exceptions import InvalidYouTubeURLError


def _make_search_item(video_id: str) -> dict:
    return {"id": {"videoId": video_id}, "snippet": {"title": f"Video {video_id}"}}


def _make_details_item(video_id: str, projection: str = "360") -> dict:
    return {
        "id": video_id,
        "snippet": {"title": f"Video {video_id}", "channelTitle": "TestChannel"},
        "contentDetails": {"projection": projection},
    }


def _build_mock_client(search_ids: list, detail_projections: dict) -> MagicMock:
    """Return a mock youtube client that simulates search + video details."""
    client = MagicMock()

    search_items = [_make_search_item(vid) for vid in search_ids]
    client.search.return_value.list.return_value.execute.return_value = {
        "items": search_items
    }

    detail_items = [
        _make_details_item(vid, detail_projections.get(vid, "360"))
        for vid in search_ids
    ]
    client.videos.return_value.list.return_value.execute.return_value = {
        "items": detail_items
    }

    return client


class SearchVideosOrderTest(unittest.TestCase):
    """search_videos must preserve API result order and de-duplicate."""

    def _call(self, search_ids, detail_projections=None):
        if detail_projections is None:
            detail_projections = {vid: "360" for vid in search_ids}
        client = _build_mock_client(search_ids, detail_projections)
        with patch("core.youtube._get_api_key", return_value="fake_key"):
            return search_videos("test query", youtube_client=client)

    def test_order_preserved(self):
        ids = ["AAA", "BBB", "CCC"]
        results = self._call(ids)
        returned_ids = [r["id"] for r in results]
        # order must match the search response order
        self.assertEqual(returned_ids, ids)

    def test_dedup_preserves_first_occurrence(self):
        """Duplicate IDs from search must be de-duplicated, keeping first position."""
        # Simulate a search response with a duplicate (real API won't do this,
        # but the dedup logic must be robust).
        ids = ["AAA", "BBB", "AAA", "CCC"]
        unique_ids = ["AAA", "BBB", "CCC"]
        client = _build_mock_client(ids, {vid: "360" for vid in unique_ids})
        # Patch the search items to include the duplicate manually
        search_items = [_make_search_item(vid) for vid in ids]
        client.search.return_value.list.return_value.execute.return_value = {
            "items": search_items
        }
        detail_items = [_make_details_item(vid) for vid in unique_ids]
        client.videos.return_value.list.return_value.execute.return_value = {
            "items": detail_items
        }
        with patch("core.youtube._get_api_key", return_value="fake_key"):
            results = search_videos("test query", youtube_client=client)
        returned_ids = [r["id"] for r in results]
        # No duplicates
        self.assertEqual(len(returned_ids), len(set(returned_ids)))
        # AAA comes before CCC
        self.assertLess(returned_ids.index("AAA"), returned_ids.index("CCC"))

    def test_non_360_videos_excluded(self):
        """Only videos with projection==360 must be returned."""
        ids = ["AAA", "BBB", "CCC"]
        projections = {"AAA": "360", "BBB": "rectangular", "CCC": "360"}
        results = self._call(ids, projections)
        returned_ids = [r["id"] for r in results]
        self.assertIn("AAA", returned_ids)
        self.assertNotIn("BBB", returned_ids)
        self.assertIn("CCC", returned_ids)

    def test_empty_search_returns_empty_list(self):
        client = MagicMock()
        client.search.return_value.list.return_value.execute.return_value = {"items": []}
        with patch("core.youtube._get_api_key", return_value="fake_key"):
            results = search_videos("nothing here", youtube_client=client)
        self.assertEqual(results, [])

    def test_result_has_required_keys(self):
        results = self._call(["AAA"])
        self.assertEqual(len(results), 1)
        self.assertIn("id", results[0])
        self.assertIn("title", results[0])
        self.assertIn("channel", results[0])
        self.assertIn("url", results[0])
        self.assertTrue(results[0]["url"].startswith("https://youtu.be/"))

    def test_build_disables_discovery_cache(self):
        built_client = _build_mock_client(["AAA"], {"AAA": "360"})
        discovery_module = ModuleType("googleapiclient.discovery")
        discovery_module.build = MagicMock(return_value=built_client)
        errors_module = ModuleType("googleapiclient.errors")
        errors_module.HttpError = RuntimeError
        package_module = ModuleType("googleapiclient")

        with (
            patch.dict(
                "sys.modules",
                {
                    "googleapiclient": package_module,
                    "googleapiclient.discovery": discovery_module,
                    "googleapiclient.errors": errors_module,
                },
            ),
            patch("core.youtube._get_api_key", return_value="fake_key"),
        ):
            results = search_videos("test query")

        self.assertEqual([item["id"] for item in results], ["AAA"])
        discovery_module.build.assert_called_once_with(
            "youtube",
            "v3",
            developerKey="fake_key",
            cache_discovery=False,
        )


class ExtractVideoIdTests(unittest.TestCase):
    """extract_video_id must handle various URL formats."""

    def test_watch_url(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_short_url(self):
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_embed_url(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_invalid_url_raises(self):
        with self.assertRaises(InvalidYouTubeURLError):
            extract_video_id("https://example.com/notavideo")

    def test_empty_string_raises(self):
        with self.assertRaises(InvalidYouTubeURLError):
            extract_video_id("")


if __name__ == "__main__":
    unittest.main()
