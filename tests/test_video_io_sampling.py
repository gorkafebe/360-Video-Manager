import tempfile
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from detector.video_io import (
    FrameExtractorError,
    clear_probe_cache,
    extract_main_frames,
    extract_secondary_frames,
    _extract_batch_frames_ffmpeg,
    open_video_capture,
)


class _FakeCapture:
    def __init__(self, opened: bool = False):
        self._opened = opened

    def isOpened(self):
        return self._opened

    def read(self):
        return False, None

    def set(self, _prop, _value):
        return None

    def get(self, _prop):
        return 0.0

    def release(self):
        return None


class _FrameLike:
    def copy(self):
        return self


class VideoIOSamplingFallbackTests(unittest.TestCase):
    def setUp(self):
        clear_probe_cache()

    def tearDown(self):
        clear_probe_cache()

    @patch("detector.video_io._extract_batch_frames_ffmpeg")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io.cv2.VideoCapture")
    def test_extract_main_frames_uses_ffmpeg_sampling_when_opencv_decode_fails(
        self,
        mock_capture,
        mock_probe,
        mock_extract_batch,
    ):
        mock_capture.return_value = _FakeCapture(opened=False)
        mock_probe.return_value = {
            "total_frames": 240,
            "fps": 24.0,
            "duration": 10.0,
        }
        mock_extract_batch.return_value = [_FrameLike(), _FrameLike(), _FrameLike(), _FrameLike()]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            result = extract_main_frames(tmp_video.name, num_frames=4, guardar_frames=False)

        self.assertEqual(len(result["frames"]), 4)
        self.assertEqual(result["fallback_ffmpeg_frames"], 4)
        self.assertEqual(result["video_path_procesado"], tmp_video.name)
        mock_extract_batch.assert_called_once()

    @patch("detector.video_io._run_ffprobe_json")
    def test_probe_cache_avoids_repeated_ffprobe_calls(self, mock_ffprobe):
        mock_ffprobe.return_value = {
            "streams": [{"codec_type": "video", "avg_frame_rate": "24/1", "nb_frames": "240",
                          "duration": "10.0", "width": 1920, "height": 1080,
                          "codec_name": "h264", "pix_fmt": "yuv420p"}],
            "format": {"duration": "10.0"},
        }
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            from detector.video_io import probe_video_stream
            result1 = probe_video_stream(tmp_video.name)
            result2 = probe_video_stream(tmp_video.name)

        mock_ffprobe.assert_called_once()
        self.assertEqual(result1["fps"], 24.0)
        self.assertEqual(result2["fps"], 24.0)

    @patch("detector.video_io._run_ffprobe_json")
    def test_probe_cache_bypassed_on_mtime_change(self, mock_ffprobe):
        mock_ffprobe.return_value = {
            "streams": [{"codec_type": "video", "avg_frame_rate": "24/1", "nb_frames": "120",
                          "duration": "5.0", "width": 1280, "height": 720,
                          "codec_name": "h264", "pix_fmt": "yuv420p"}],
            "format": {"duration": "5.0"},
        }
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            path = tmp_video.name

        import os
        import time
        from detector.video_io import probe_video_stream
        probe_video_stream(path)
        # Simulate mtime change by touching the file — the cache key will differ
        time.sleep(0.01)
        os.utime(path, None)
        probe_video_stream(path)

        # Both calls should have gone to ffprobe because mtime changed the cache key
        self.assertEqual(mock_ffprobe.call_count, 2)
        os.unlink(path)

    @patch("detector.video_io._extract_batch_frames_ffmpeg")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io.cv2.VideoCapture")
    def test_shared_cap_session_opens_opencv_once(
        self,
        mock_capture,
        mock_probe,
        mock_extract_batch,
    ):
        """cv2.VideoCapture should be called exactly once when cap_session is shared."""
        mock_capture.return_value = _FakeCapture(opened=False)
        mock_probe.return_value = {
            "total_frames": 120,
            "fps": 24.0,
            "duration": 5.0,
        }
        mock_extract_batch.return_value = [_FrameLike(), _FrameLike()]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            with open_video_capture(tmp_video.name) as cap_session:
                extract_main_frames(
                    tmp_video.name, num_frames=2, guardar_frames=False, cap_session=cap_session
                )
                extract_secondary_frames(
                    tmp_video.name,
                    frame_positions=[10, 20],
                    total_frames=120,
                    cap_session=cap_session,
                )

        # cv2.VideoCapture opened once (for the shared session), not once per function
        mock_capture.assert_called_once_with(tmp_video.name)

    @patch("detector.video_io.subprocess.run")
    def test_batch_ffmpeg_timeout_reports_diagnostics(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=120)

        frames, diag = _extract_batch_frames_ffmpeg(
            "/tmp/video.mp4",
            [1.0, 2.0],
            return_diagnostics=True,
        )

        self.assertEqual(frames, [None, None])
        self.assertTrue(diag["timeout"])
        self.assertEqual(diag["error_code"], "ffmpeg_batch_timeout")

    @patch("detector.video_io.subprocess.run")
    def test_batch_ffmpeg_empty_output_reports_diagnostics(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")

        frames, diag = _extract_batch_frames_ffmpeg(
            "/tmp/video.mp4",
            [1.0, 2.0],
            return_diagnostics=True,
        )

        self.assertEqual(frames, [None, None])
        self.assertFalse(diag["timeout"])
        self.assertEqual(diag["error_code"], "ffmpeg_batch_no_output")

    @patch("detector.video_io._extract_single_frame_ffmpeg_with_diagnostics")
    @patch("detector.video_io._extract_batch_frames_ffmpeg")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io.cv2.VideoCapture")
    def test_extract_main_frames_falls_back_to_single_ffmpeg_after_batch_failure(
        self,
        mock_capture,
        mock_probe,
        mock_extract_batch,
        mock_extract_single,
    ):
        mock_capture.return_value = _FakeCapture(opened=False)
        mock_probe.return_value = {
            "total_frames": 240,
            "fps": 24.0,
            "duration": 10.0,
        }
        mock_extract_batch.return_value = (
            [None, None],
            {"timeout": False, "error_code": "ffmpeg_batch_no_output"},
        )
        mock_extract_single.side_effect = [
            (_FrameLike(), {"timeout": False, "decoded": True, "timestamp": 1.0}),
            (_FrameLike(), {"timeout": False, "decoded": True, "timestamp": 2.0}),
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            result = extract_main_frames(tmp_video.name, num_frames=2, guardar_frames=False)

        self.assertEqual(len(result["frames"]), 2)
        self.assertEqual(result["fallback_ffmpeg_single_frames"], 2)
        self.assertEqual(result["fallback_ffmpeg_frames"], 0)
        self.assertEqual(mock_extract_single.call_count, 2)

    @patch("detector.video_io._extract_single_frame_ffmpeg_with_diagnostics")
    @patch("detector.video_io._extract_batch_frames_ffmpeg")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io.cv2.VideoCapture")
    def test_extract_main_frames_hard_fail_includes_enriched_error_context(
        self,
        mock_capture,
        mock_probe,
        mock_extract_batch,
        mock_extract_single,
    ):
        mock_capture.return_value = _FakeCapture(opened=False)
        mock_probe.return_value = {
            "total_frames": 240,
            "fps": 24.0,
            "duration": 10.0,
        }
        mock_extract_batch.return_value = (
            [None, None],
            {"timeout": True, "error_code": "ffmpeg_batch_timeout"},
        )
        mock_extract_single.side_effect = [
            (None, {"timeout": True, "decoded": False, "timestamp": 1.0}),
            (None, {"timeout": False, "decoded": False, "timestamp": 2.0}),
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            with self.assertRaises(FrameExtractorError) as exc:
                extract_main_frames(tmp_video.name, num_frames=2, guardar_frames=False)

        self.assertEqual(exc.exception.code, "frame_extraction_timeout")
        self.assertIn("attempts", exc.exception.details)
        self.assertTrue(exc.exception.details["attempts"])


if __name__ == "__main__":
    unittest.main()
