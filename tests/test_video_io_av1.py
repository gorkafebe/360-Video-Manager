import subprocess
import unittest
from unittest.mock import MagicMock, patch, call

from detector.video_io import (
    FrameExtractorError,
    clear_probe_cache,
    convert_video_codec,
    _extract_batch_frames_ffmpeg,
    _extract_batch_frames_ffmpeg_parallel,
    MAX_SINGLE_PASS_FRAMES,
)


class _FakeCapture:
    def __init__(self, opened: bool = True, readable: bool = True):
        self._opened = opened
        self._readable = readable

    def isOpened(self):
        return self._opened

    def read(self):
        return self._readable, MagicMock() if self._readable else None

    def set(self, _prop, _value):
        return None

    def get(self, _prop):
        return 0.0

    def release(self):
        return None


_AV1_PROBE = {
    "codec_name": "av1",
    "total_frames": 240,
    "fps": 24.0,
    "width": 1920,
    "height": 960,
    "duration": 10.0,
}

_H264_PROBE = {
    "codec_name": "h264",
    "total_frames": 240,
    "fps": 24.0,
    "width": 1920,
    "height": 960,
    "duration": 10.0,
}


class AV1TranscodeCommandTests(unittest.TestCase):
    def setUp(self):
        clear_probe_cache()

    def tearDown(self):
        clear_probe_cache()

    @patch("detector.video_io.os.path.exists", return_value=True)
    @patch("detector.video_io.detect_ffmpeg_h264_encoder", return_value="libx264")
    @patch("detector.video_io.cv2.VideoCapture")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io._can_decode_with_opencv", return_value=False)
    @patch("detector.video_io.subprocess.run")
    def test_av1_input_omits_hwaccel_from_transcode_command(
        self, mock_run, _mock_can_decode, mock_probe, mock_cap, _mock_encoder, _mock_exists
    ):
        mock_probe.return_value = _AV1_PROBE
        mock_cap.return_value = _FakeCapture(opened=True, readable=True)
        mock_run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"ffmpeg version 6", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""),
        ]

        convert_video_codec("fake_av1.mkv")

        transcode_call = mock_run.call_args_list[1]
        cmd = transcode_call[0][0]
        self.assertNotIn("-hwaccel", cmd, "AV1 input must not use -hwaccel auto")

    @patch("detector.video_io.os.path.exists", return_value=True)
    @patch("detector.video_io.detect_ffmpeg_h264_encoder", return_value="libx264")
    @patch("detector.video_io.cv2.VideoCapture")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io._can_decode_with_opencv", return_value=False)
    @patch("detector.video_io.subprocess.run")
    def test_h264_input_retains_hwaccel_in_transcode_command(
        self, mock_run, _mock_can_decode, mock_probe, mock_cap, _mock_encoder, _mock_exists
    ):
        mock_probe.return_value = _H264_PROBE
        mock_cap.return_value = _FakeCapture(opened=True, readable=True)
        mock_run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"ffmpeg version 6", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""),
        ]

        convert_video_codec("fake_h264.mp4")

        transcode_call = mock_run.call_args_list[1]
        cmd = transcode_call[0][0]
        self.assertIn("-hwaccel", cmd, "Non-AV1 input must still use -hwaccel auto")
        hwaccel_idx = cmd.index("-hwaccel")
        self.assertEqual(cmd[hwaccel_idx + 1], "auto")

    @patch("detector.video_io.os.path.exists", return_value=True)
    @patch("detector.video_io.detect_ffmpeg_h264_encoder", return_value="libx264")
    @patch("detector.video_io.cv2.VideoCapture")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io._can_decode_with_opencv", return_value=False)
    @patch("detector.video_io.subprocess.run")
    def test_av1_transcode_does_not_raise_on_success(
        self, mock_run, _mock_can_decode, mock_probe, mock_cap, _mock_encoder, _mock_exists
    ):
        mock_probe.return_value = _AV1_PROBE
        mock_cap.return_value = _FakeCapture(opened=True, readable=True)
        mock_run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"ffmpeg version 6", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""),
        ]

        result = convert_video_codec("fake_av1.mkv")
        self.assertIsInstance(result, str)

    @patch("detector.video_io.os.path.exists", return_value=True)
    @patch("detector.video_io.detect_ffmpeg_h264_encoder", return_value="libx264")
    @patch("detector.video_io.cv2.VideoCapture")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io._can_decode_with_opencv", return_value=False)
    @patch("detector.video_io.subprocess.run")
    def test_libaom_av1_codec_name_also_omits_hwaccel(
        self, mock_run, _mock_can_decode, mock_probe, mock_cap, _mock_encoder, _mock_exists
    ):
        probe = dict(_AV1_PROBE, codec_name="libaom-av1")
        mock_probe.return_value = probe
        mock_cap.return_value = _FakeCapture(opened=True, readable=True)
        mock_run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"ffmpeg version 6", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""),
        ]

        convert_video_codec("fake_av1_libaom.mkv")

        transcode_call = mock_run.call_args_list[1]
        cmd = transcode_call[0][0]
        self.assertNotIn("-hwaccel", cmd)

    @patch("detector.video_io.os.path.exists", return_value=True)
    @patch("detector.video_io.detect_ffmpeg_h264_encoder", return_value="libx264")
    @patch("detector.video_io.cv2.VideoCapture")
    @patch("detector.video_io.probe_video_stream")
    @patch("detector.video_io._can_decode_with_opencv", return_value=False)
    @patch("detector.video_io.subprocess.run")
    def test_av1_ffmpeg_failure_raises_frame_extractor_error(
        self, mock_run, _mock_can_decode, mock_probe, mock_cap, _mock_encoder, _mock_exists
    ):
        mock_probe.return_value = _AV1_PROBE
        mock_cap.return_value = _FakeCapture(opened=False, readable=False)
        mock_run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"ffmpeg version 6", stderr=b""),
            subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"decoder error"),
        ]

        with self.assertRaises(FrameExtractorError):
            convert_video_codec("fake_av1_bad.mkv")


class ParallelExtractionRoutingTests(unittest.TestCase):
    def setUp(self):
        clear_probe_cache()

    def tearDown(self):
        clear_probe_cache()

    @patch("detector.video_io._extract_batch_frames_ffmpeg_parallel")
    @patch("detector.video_io.subprocess.run")
    def test_large_batch_routes_to_parallel_path(self, mock_run, mock_parallel):
        timestamps = [float(i) for i in range(MAX_SINGLE_PASS_FRAMES + 1)]
        mock_parallel.return_value = [None] * len(timestamps)

        result = _extract_batch_frames_ffmpeg("fake.mp4", timestamps)

        mock_parallel.assert_called_once()
        mock_run.assert_not_called()

    @patch("detector.video_io._extract_batch_frames_ffmpeg_parallel")
    @patch("detector.video_io.subprocess.run")
    def test_small_batch_skips_parallel_path(self, mock_run, mock_parallel):
        timestamps = [float(i) for i in range(MAX_SINGLE_PASS_FRAMES)]
        mock_run.return_value = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"")

        _extract_batch_frames_ffmpeg("fake.mp4", timestamps)

        mock_parallel.assert_not_called()

    @patch("detector.video_io._extract_batch_frames_ffmpeg_parallel")
    @patch("detector.video_io.subprocess.run")
    def test_parallel_failure_falls_back_to_select_filter(self, mock_run, mock_parallel):
        timestamps = [float(i) for i in range(MAX_SINGLE_PASS_FRAMES + 1)]
        mock_parallel.side_effect = RuntimeError("parallel failed")
        mock_run.return_value = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"")

        result = _extract_batch_frames_ffmpeg("fake.mp4", timestamps)

        mock_parallel.assert_called_once()
        mock_run.assert_called_once()

    @patch("detector.video_io._extract_batch_frames_ffmpeg_parallel")
    @patch("detector.video_io.subprocess.run")
    def test_large_batch_with_return_diagnostics_includes_mode_parallel(
        self, mock_run, mock_parallel
    ):
        timestamps = [float(i) for i in range(MAX_SINGLE_PASS_FRAMES + 1)]
        mock_parallel.return_value = [None] * len(timestamps)

        frames, diag = _extract_batch_frames_ffmpeg("fake.mp4", timestamps, return_diagnostics=True)

        self.assertEqual(diag["mode"], "parallel")
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
