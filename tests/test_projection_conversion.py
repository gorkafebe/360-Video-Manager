"""Tests for projection conversion ffmpeg command and retry behavior."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

def _fake_callable(*args, **kwargs):
    return None


fake_cv2 = types.ModuleType("cv2")
fake_cv2.__getattr__ = lambda _name: _fake_callable

fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
fake_numpy.mean = lambda values: (sum(values) / len(values)) if values else 0.0
fake_numpy.median = lambda values: sorted(values)[len(values) // 2] if values else 0.0
fake_numpy.pi = 3.141592653589793
fake_numpy.__getattr__ = lambda _name: _fake_callable

sys.modules.setdefault("cv2", fake_cv2)
sys.modules.setdefault("numpy", fake_numpy)

from detector.projection_conversion import (
    _is_audio_failure,
    build_ffmpeg_command_for_projection,
    build_v360_filter_for_projection,
    convert_detected_projection_to_equirectangular,
)


class ProjectionConversionCommandTests(unittest.TestCase):
    def test_build_v360_filter_includes_even_padding(self):
        filt = build_v360_filter_for_projection("eac")
        self.assertEqual(
            filt,
            "v360=eac:equirect,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2",
        )

    def test_build_ffmpeg_command_uses_padded_v360_filter(self):
        cmd = build_ffmpeg_command_for_projection(
            "/tmp/input.mp4",
            "/tmp/output.mp4",
            "cubic",
        )
        self.assertIn("-vf", cmd)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("v360=c3x2:equirect", vf)
        self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2", vf)


class ProjectionConversionRetryTests(unittest.TestCase):
    def test_is_audio_failure_false_for_video_dimension_error(self):
        stderr = (
            "[libx264 @ 0x1] height not divisible by 2 (3840x1707)\n"
            "Error initializing output stream 0:0 -- Error while opening encoder "
            "for output stream #0:0"
        )
        self.assertFalse(_is_audio_failure(stderr))

    @patch("detector.projection_conversion.run_ffmpeg_command")
    @patch("detector.projection_conversion.is_ffmpeg_available", return_value=True)
    def test_dimension_error_does_not_trigger_audio_drop_retry(
        self,
        _mock_ffmpeg_available,
        mock_run_ffmpeg_command,
    ):
        mock_run_ffmpeg_command.return_value = {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": (
                "[libx264 @ 0x1] height not divisible by 2 (3840x1707)\n"
                "Error initializing output stream 0:0 -- Error while opening encoder "
                "for output stream #0:0"
            ),
            "error": "ffmpeg exited with code 1",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            with open(input_path, "wb"):
                pass

            result = convert_detected_projection_to_equirectangular(
                video_path=input_path,
                projection_type="eac",
                output_dir=tmpdir,
            )

        self.assertFalse(result["success"])
        self.assertEqual(mock_run_ffmpeg_command.call_count, 1)
        called_cmd = mock_run_ffmpeg_command.call_args_list[0].args[0]
        self.assertNotIn("-an", called_cmd)


if __name__ == "__main__":
    unittest.main()
