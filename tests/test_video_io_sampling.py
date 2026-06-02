import tempfile
import unittest
from unittest.mock import patch

from detector.video_io import extract_main_frames


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
    @patch("detector.video_io._extract_single_frame_ffmpeg")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io.cv2.VideoCapture")
    def test_extract_main_frames_uses_ffmpeg_sampling_when_opencv_decode_fails(
        self,
        mock_capture,
        mock_probe,
        mock_extract_ffmpeg,
    ):
        mock_capture.return_value = _FakeCapture(opened=False)
        mock_probe.return_value = {
            "total_frames": 240,
            "fps": 24.0,
            "duration": 10.0,
        }
        mock_extract_ffmpeg.return_value = _FrameLike()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            result = extract_main_frames(tmp_video.name, num_frames=4, guardar_frames=False)

        self.assertEqual(len(result["frames"]), 4)
        self.assertEqual(result["fallback_ffmpeg_frames"], 4)
        self.assertEqual(result["video_path_procesado"], tmp_video.name)


if __name__ == "__main__":
    unittest.main()
