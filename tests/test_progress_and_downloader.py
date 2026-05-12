import os
import tempfile
import unittest

from app.gui.progress_utils import (
    compute_progress_update_delay_ms,
    extract_download_progress_fraction,
)
from core.downloader import _resolve_downloaded_output_path


class _FakeYDL:
    def prepare_filename(self, info):
        title = info.get("title", "video")
        ext = info.get("ext", "webm")
        return os.path.join(info.get("output_dir", ""), f"{title}.{ext}")


class ProgressParsingTests(unittest.TestCase):
    def test_prefers_byte_counters(self):
        frac = extract_download_progress_fraction(
            {
                "downloaded_bytes": 50,
                "total_bytes": 200,
                "_percent_str": "1.0%",
            }
        )
        self.assertAlmostEqual(frac, 0.25)

    def test_falls_back_to_percent_string(self):
        frac = extract_download_progress_fraction({"_percent_str": "33.3%"})
        self.assertAlmostEqual(frac, 0.333, places=3)

    def test_invalid_payload_returns_none(self):
        self.assertIsNone(extract_download_progress_fraction({"_percent_str": "??"}))

    def test_clamps_progress_to_one(self):
        frac = extract_download_progress_fraction(
            {"downloaded_bytes": 250, "total_bytes": 200}
        )
        self.assertEqual(frac, 1.0)

    def test_coalescing_delay_respects_interval(self):
        delay = compute_progress_update_delay_ms(
            last_update_monotonic=10.0,
            now_monotonic=10.03,
            min_interval_ms=120,
        )
        self.assertGreaterEqual(delay, 90)
        self.assertLessEqual(delay, 91)

    def test_coalescing_delay_can_be_immediate(self):
        delay = compute_progress_update_delay_ms(
            last_update_monotonic=10.0,
            now_monotonic=10.2,
            min_interval_ms=120,
        )
        self.assertEqual(delay, 0)


class DownloaderOutputResolutionTests(unittest.TestCase):
    def test_uses_requested_download_filepath_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "merged.mp4")
            with open(final_path, "w", encoding="utf-8") as f:
                f.write("ok")
            info = {
                "requested_downloads": [{"filepath": final_path}],
                "title": "video",
                "ext": "webm",
                "output_dir": tmpdir,
            }
            resolved = _resolve_downloaded_output_path(_FakeYDL(), info)
            self.assertEqual(resolved, os.path.abspath(final_path))

    def test_falls_back_to_mp4_variant_of_prepared_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = os.path.join(tmpdir, "video.mp4")
            with open(mp4_path, "w", encoding="utf-8") as f:
                f.write("ok")
            info = {"title": "video", "ext": "webm", "output_dir": tmpdir}
            resolved = _resolve_downloaded_output_path(_FakeYDL(), info)
            self.assertEqual(resolved, os.path.abspath(mp4_path))


if __name__ == "__main__":
    unittest.main()
