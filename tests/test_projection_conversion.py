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
    _is_hardware_encoder_runtime_failure,
    _is_audio_failure,
    build_ffmpeg_command_for_projection,
    build_v360_filter_for_projection,
    convert_detected_projection_to_equirectangular,
    get_conversion_output_profile,
)


class ProjectionConversionCommandTests(unittest.TestCase):
    def test_build_v360_filter_includes_even_padding(self):
        filt = build_v360_filter_for_projection("eac")
        self.assertEqual(
            filt,
            "v360=eac:equirect,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2,scale=4320:2160:flags=lanczos",
        )

    def test_build_ffmpeg_command_uses_padded_v360_filter(self):
        with patch("detector.projection_conversion.detect_ffmpeg_h264_encoder", return_value="libx264"):
            cmd = build_ffmpeg_command_for_projection(
                "/tmp/input.mp4",
                "/tmp/output.mp4",
                "cubic",
            )
        self.assertIn("-vf", cmd)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("v360=c3x2:equirect", vf)
        self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2", vf)
        self.assertIn("scale=4320:2160:flags=lanczos", vf)
        self.assertIn("-preset", cmd)
        self.assertIn("medium", cmd)
        self.assertIn("-crf", cmd)
        self.assertIn("16", cmd)
        # Hardware acceleration must NOT be present when the v360 software filter
        # is used: hardware-decoded frames stay in device memory and cannot be
        # consumed by the software filter, which would cause ffmpeg to create an
        # empty output file and exit with an error.
        self.assertNotIn("-hwaccel", cmd)

    def test_build_ffmpeg_command_allows_target_profile_override(self):
        with patch("detector.projection_conversion.detect_ffmpeg_h264_encoder", return_value="libx264"):
            cmd = build_ffmpeg_command_for_projection(
                "/tmp/input.mp4",
                "/tmp/output.mp4",
                "eac",
                target_width=4096,
                target_height=2048,
                crf=14,
                preset="slow",
            )
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("scale=4096:2048:flags=lanczos", vf)
        self.assertIn("slow", cmd)
        self.assertIn("14", cmd)

    def test_build_ffmpeg_command_stream_copy_uses_hwaccel(self):
        """Stream-copy path (no v360 filter) may safely use hardware acceleration."""
        with patch("detector.projection_conversion.detect_ffmpeg_h264_encoder", return_value="libx264"):
            cmd = build_ffmpeg_command_for_projection(
                "/tmp/input.mp4",
                "/tmp/output.mp4",
                # equirectangular has no v360 filter → stream-copy path
                "equirectangular",
            )
        self.assertNotIn("-vf", cmd)
        self.assertIn("-hwaccel", cmd)
        self.assertIn("auto", cmd)


class ProjectionConversionRetryTests(unittest.TestCase):
    @patch("config.settings.get_settings")
    def test_conversion_output_profile_loaded_from_settings(self, mock_get_settings):
        mock_get_settings.return_value = types.SimpleNamespace(
            conversion_target_width=4096,
            conversion_target_height=2048,
            conversion_crf=14,
            conversion_preset="slow",
        )
        profile = get_conversion_output_profile()
        self.assertEqual(profile["target_width"], 4096)
        self.assertEqual(profile["target_height"], 2048)
        self.assertEqual(profile["crf"], 14)
        self.assertEqual(profile["preset"], "slow")

    @patch("config.settings.get_settings", side_effect=ImportError("settings unavailable"))
    def test_conversion_output_profile_falls_back_when_settings_unavailable(self, _mock_get_settings):
        profile = get_conversion_output_profile()
        self.assertEqual(profile["target_width"], 4320)
        self.assertEqual(profile["target_height"], 2160)
        self.assertEqual(profile["crf"], 16)
        self.assertEqual(profile["preset"], "medium")

    @patch("config.settings.get_settings")
    def test_conversion_output_profile_normalizes_invalid_values(self, mock_get_settings):
        mock_get_settings.return_value = types.SimpleNamespace(
            conversion_target_width="invalid",
            conversion_target_height=-1,
            conversion_crf="999",
            conversion_preset="",
        )
        profile = get_conversion_output_profile()
        self.assertEqual(profile["target_width"], 4320)
        self.assertEqual(profile["target_height"], 2160)
        self.assertEqual(profile["crf"], 51)
        self.assertEqual(profile["preset"], "medium")

    def test_hardware_encoder_runtime_failure_detected_for_nvenc_cuda_error(self):
        stderr = "[h264_nvenc @ 0x1] Cannot load libcuda.so.1\nError while opening encoder"
        self.assertTrue(_is_hardware_encoder_runtime_failure(stderr, "h264_nvenc"))

    def test_hardware_encoder_runtime_failure_not_detected_for_libx264(self):
        stderr = "Error while opening encoder for output stream #0:0"
        self.assertFalse(_is_hardware_encoder_runtime_failure(stderr, "libx264"))

    def test_is_audio_failure_false_for_video_dimension_error(self):
        stderr = (
            "[libx264 @ 0x1] height not divisible by 2 (3840x1707)\n"
            "Error initializing output stream 0:0 -- Error while opening encoder "
            "for output stream #0:0"
        )
        self.assertFalse(_is_audio_failure(stderr))

    @patch("detector.projection_conversion.run_ffmpeg_command")
    @patch("detector.projection_conversion.is_ffmpeg_available", return_value=True)
    @patch("detector.projection_conversion.detect_ffmpeg_h264_encoder", return_value="h264_nvenc")
    def test_hw_encoder_failure_retries_with_libx264(
        self,
        _mock_detect_encoder,
        _mock_ffmpeg_available,
        mock_run_ffmpeg_command,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            with open(input_path, "wb"):
                pass

            output_path = os.path.join(tmpdir, "input_equirectangular.mp4")
            with open(output_path, "wb") as f:
                f.write(b"x")

            mock_run_ffmpeg_command.side_effect = [
                {
                    "success": False,
                    "returncode": 255,
                    "stdout": "",
                    "stderr": "[h264_nvenc @ 0x1] Cannot load libcuda.so.1\nError while opening encoder",
                    "error": "ffmpeg exited with code 255",
                },
                {
                    "success": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                },
            ]

            result = convert_detected_projection_to_equirectangular(
                video_path=input_path,
                projection_type="eac",
                output_dir=tmpdir,
            )

        self.assertTrue(result["success"])
        self.assertEqual(mock_run_ffmpeg_command.call_count, 2)
        first_cmd = mock_run_ffmpeg_command.call_args_list[0].args[0]
        second_cmd = mock_run_ffmpeg_command.call_args_list[1].args[0]
        self.assertIn("h264_nvenc", first_cmd)
        self.assertIn("libx264", second_cmd)
        self.assertEqual(result["output_path"], output_path)

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


    @patch("detector.projection_conversion.run_ffmpeg_command")
    @patch("detector.projection_conversion.is_ffmpeg_available", return_value=True)
    def test_empty_output_file_removed_on_ffmpeg_failure(
        self,
        _mock_ffmpeg_available,
        mock_run_ffmpeg_command,
    ):
        """When ffmpeg fails, any empty output file it created must be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            with open(input_path, "wb"):
                pass

            # Pre-create the empty output file to simulate what ffmpeg does
            # with -y before it fails: it opens/truncates the output file first.
            from detector.projection_conversion import build_equirectangular_output_path
            output_path = build_equirectangular_output_path(input_path, output_dir=tmpdir)
            with open(output_path, "wb"):
                pass  # create zero-byte file

            mock_run_ffmpeg_command.return_value = {
                "success": False,
                "returncode": 1,
                "stdout": "",
                "stderr": "v360: some filter error",
                "error": "ffmpeg exited with code 1",
            }

            result = convert_detected_projection_to_equirectangular(
                video_path=input_path,
                projection_type="eac",
                output_dir=tmpdir,
            )

            self.assertFalse(result["success"])
            self.assertIsNotNone(result["output_path"])
            self.assertFalse(
                os.path.exists(output_path),
                "Empty output file should have been removed after ffmpeg failure",
            )


if __name__ == "__main__":
    unittest.main()
