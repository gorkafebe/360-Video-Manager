import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


def _fake_callable(*args, **kwargs):
    return None


fake_cv2 = types.ModuleType("cv2")
fake_cv2.imwrite = _fake_callable
fake_cv2.line = _fake_callable
fake_cv2.circle = _fake_callable
fake_cv2.VideoCapture = _fake_callable
fake_cv2.CAP_PROP_FRAME_COUNT = 7
fake_cv2.CAP_PROP_FPS = 5
fake_cv2.CAP_PROP_POS_FRAMES = 1
fake_cv2.__getattr__ = lambda _name: _fake_callable

fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
fake_numpy.mean = lambda values: (sum(values) / len(values)) if values else 0.0
fake_numpy.median = lambda values: sorted(values)[len(values) // 2] if values else 0.0
fake_numpy.pi = 3.141592653589793
fake_numpy.__getattr__ = lambda _name: _fake_callable

sys.modules.setdefault("cv2", fake_cv2)
sys.modules.setdefault("numpy", fake_numpy)

from detector.debug_utils import (
    create_run_debug_dir,
    save_frame_debug,
    save_line_detected_frame,
    save_line_visual_debug,
    save_stereo_halves,
)
from detector.pipeline import run_detection_with_retries
from detector.video_io import extract_main_frames


class _FakeFrame:
    def __init__(self, name: str, shape=None):
        self.name = name
        self.shape = shape or (8, 8, 3)

    def copy(self):
        return self


class _FakeCapture:
    def __init__(self, total_frames: int, fps: float):
        self.total_frames = total_frames
        self.fps = fps
        self.current_pos = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set(self, _prop, value):
        self.current_pos = int(value)
        return True

    def read(self):
        return True, _FakeFrame(f"frame-{self.current_pos}")

    def release(self):
        return None


class SaveFrameDebugContractTests(unittest.TestCase):
    @patch("detector.debug_utils.log_success")
    @patch("detector.debug_utils.log_discard")
    @patch("detector.debug_utils.cv2.imwrite", return_value=True)
    @patch("detector.debug_utils.os.makedirs")
    def test_save_frame_debug_uses_cv2_imwrite_and_success_logging(
        self,
        mock_makedirs,
        mock_imwrite,
        mock_log_discard,
        mock_log_success,
    ):
        frame = object()
        filepath = "/tmp/debug/main/frame.jpg"

        ok = save_frame_debug(filepath, frame, "MAIN")

        self.assertTrue(ok)
        mock_makedirs.assert_called_once_with("/tmp/debug/main", exist_ok=True)
        mock_imwrite.assert_called_once_with(filepath, frame)
        mock_log_success.assert_called_once_with(f"[MAIN] SAVED -> {filepath}")
        mock_log_discard.assert_not_called()

    @patch("detector.debug_utils.log_success")
    @patch("detector.debug_utils.log_discard")
    @patch("detector.debug_utils.cv2.imwrite", return_value=False)
    @patch("detector.debug_utils.os.makedirs")
    def test_save_frame_debug_preserves_failure_logging(
        self,
        mock_makedirs,
        mock_imwrite,
        mock_log_discard,
        mock_log_success,
    ):
        frame = object()
        filepath = "/tmp/debug/main/frame.jpg"

        ok = save_frame_debug(filepath, frame, "SECONDARY")

        self.assertFalse(ok)
        mock_makedirs.assert_called_once_with("/tmp/debug/main", exist_ok=True)
        mock_imwrite.assert_called_once_with(filepath, frame)
        mock_log_discard.assert_called_once_with(f"[SECONDARY] SAVE_FAILED -> {filepath}")
        mock_log_success.assert_not_called()


class DebugDirectoryContractTests(unittest.TestCase):
    def test_create_run_debug_dir_preserves_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            context = create_run_debug_dir("/videos/sample-video.mp4", tmpdir)

            self.assertEqual(os.path.basename(context["run_dir"]), context["run_id"])
            self.assertTrue(os.path.isdir(context["main_frames_dir"]))
            self.assertTrue(os.path.isdir(context["secondary_frames_dir"]))
            self.assertTrue(os.path.isdir(context["motion_vectors_dir"]))
            self.assertTrue(os.path.isdir(context["line_detection_dir"]))
            self.assertTrue(os.path.isdir(context["logs_dir"]))
            self.assertEqual(os.path.basename(context["main_frames_dir"]), "main_frames")
            self.assertEqual(os.path.basename(context["secondary_frames_dir"]), "secondary_frames")
            self.assertEqual(os.path.basename(context["motion_vectors_dir"]), "motion_vectors")
            self.assertEqual(os.path.basename(context["line_detection_dir"]), "line_detection")
            self.assertEqual(os.path.basename(context["logs_dir"]), "logs")
            self.assertEqual(context["video_tag"], context["run_id"])


class DetectionDebugHookContractTests(unittest.TestCase):
    @patch("detector.pipeline.run_detection_pipeline")
    @patch("detector.pipeline.extract_main_frames")
    @patch("detector.pipeline.create_run_debug_dir")
    @patch("detector.pipeline._build_detection_retry_plan")
    def test_run_detection_with_retries_preserves_debug_save_hook_configuration(
        self,
        mock_retry_plan,
        mock_create_run_debug_dir,
        mock_extract_main_frames,
        mock_run_detection_pipeline,
    ):
        debug_context = {
            "run_dir": "/tmp/debug/run",
            "run_id": "run",
            "main_frames_dir": "/tmp/debug/run/main_frames",
            "secondary_frames_dir": "/tmp/debug/run/secondary_frames",
            "motion_vectors_dir": "/tmp/debug/run/motion_vectors",
            "line_detection_dir": "/tmp/debug/run/line_detection",
            "logs_dir": "/tmp/debug/run/logs",
            "secondary_sequences_dir": "/tmp/debug/run/secondary_frames",
            "motion_visualizations_dir": "/tmp/debug/run/motion_vectors",
            "horizontal_line_dir": "/tmp/debug/run/line_detection",
            "video_tag": "run",
        }
        mock_retry_plan.return_value = [
            {
                "num_frames": 10,
                "paso_frames_secundarios": 5,
                "min_frames_with_line_required": 7,
            }
        ]
        mock_create_run_debug_dir.return_value = debug_context
        mock_extract_main_frames.return_value = {
            "frames": ["frame-1"],
            "frames_metadata": [{"position": 10}],
            "video_name": "video",
        }
        mock_run_detection_pipeline.return_value = {
            "projection_type": "cubic",
            "confidence": 0.9,
            "motion_reliable": True,
            "motion_reliability_reason": "",
        }

        result, returned_context = run_detection_with_retries(
            "/tmp/video.mp4",
            num_frames=10,
            debug_base_dir="/tmp/debug",
        )

        self.assertEqual(result["projection_type"], "cubic")
        self.assertEqual(returned_context, debug_context)
        mock_create_run_debug_dir.assert_called_once_with("/tmp/video.mp4", "/tmp/debug")
        extract_kwargs = mock_extract_main_frames.call_args.kwargs
        self.assertEqual(extract_kwargs["output_dir"], "/tmp/debug/run/main_frames")
        self.assertTrue(extract_kwargs["guardar_frames"])
        self.assertEqual(extract_kwargs["frame_filename_prefix"], "run")
        self.assertIsNotNone(extract_kwargs["save_image_fn"])

    @patch("detector.pipeline.run_detection_pipeline")
    @patch("detector.pipeline.extract_main_frames")
    @patch("detector.pipeline.create_run_debug_dir")
    @patch("detector.pipeline._build_detection_retry_plan")
    def test_run_detection_with_retries_keeps_debug_saving_disabled_without_debug_dir(
        self,
        mock_retry_plan,
        mock_create_run_debug_dir,
        mock_extract_main_frames,
        mock_run_detection_pipeline,
    ):
        mock_retry_plan.return_value = [
            {
                "num_frames": 10,
                "paso_frames_secundarios": 5,
                "min_frames_with_line_required": 7,
            }
        ]
        mock_extract_main_frames.return_value = {
            "frames": ["frame-1"],
            "frames_metadata": [{"position": 10}],
            "video_name": "video",
        }
        mock_run_detection_pipeline.return_value = {
            "projection_type": "unknown",
            "confidence": 0.1,
            "motion_reliable": True,
            "motion_reliability_reason": "",
        }

        _result, returned_context = run_detection_with_retries("/tmp/video.mp4", num_frames=10)

        self.assertEqual(returned_context, {})
        mock_create_run_debug_dir.assert_not_called()
        extract_kwargs = mock_extract_main_frames.call_args.kwargs
        self.assertIsNone(extract_kwargs["output_dir"])
        self.assertFalse(extract_kwargs["guardar_frames"])
        self.assertIsNone(extract_kwargs["frame_filename_prefix"])
        self.assertIsNone(extract_kwargs["save_image_fn"])


class DebugFilenameContractTests(unittest.TestCase):
    @patch("detector.debug_utils.save_frame_debug", return_value=True)
    def test_save_line_detected_frame_preserves_detected_filename_pattern(self, mock_save_frame_debug):
        frame = _FakeFrame("line-frame")
        line_data = {"x1": 0, "y1": 4, "x2": 7, "y2": 4, "center_y": 4, "video_position": 12}

        result = save_line_detected_frame(frame, line_data, frame_idx=2, output_dir="/tmp/lines")

        self.assertEqual(result, "/tmp/lines/line_main_002_video_000012_detected.jpg")
        mock_save_frame_debug.assert_called_once()
        self.assertEqual(mock_save_frame_debug.call_args.args[0], result)

    @patch("detector.debug_utils.save_frame_debug", return_value=True)
    def test_save_line_visual_debug_preserves_horizontal_and_vertical_names(self, mock_save_frame_debug):
        frame = _FakeFrame("visual-frame")

        with patch("detector.debug_utils.draw_search_band", return_value=frame):
            horizontal = save_line_visual_debug(
                frame=frame,
                frame_idx=1,
                output_dir="/tmp/lines",
                debug_line_info={},
                found=True,
            )
            vertical = save_line_visual_debug(
                frame=frame,
                frame_idx=1,
                output_dir="/tmp/lines",
                debug_line_info={},
                found=False,
                line_orientation="vertical",
            )

        self.assertEqual(horizontal, "/tmp/lines/frame_01_line_debug_found.jpg")
        self.assertEqual(vertical, "/tmp/lines/frame_01_vertical_line_debug_not_found.jpg")
        self.assertEqual(mock_save_frame_debug.call_args_list[0].args[0], horizontal)
        self.assertEqual(mock_save_frame_debug.call_args_list[1].args[0], vertical)

    @patch("detector.debug_utils.save_frame_debug")
    def test_save_stereo_halves_preserves_filename_pattern(self, mock_save_frame_debug):
        frame = _FakeFrame("stereo-frame")

        save_stereo_halves(frame_idx=0, top=frame, bottom=frame, output_dir="/tmp/stereo")

        saved_paths = [call.args[0] for call in mock_save_frame_debug.call_args_list]
        self.assertEqual(
            saved_paths,
            [
                "/tmp/stereo/top_half_frame_001.jpg",
                "/tmp/stereo/bottom_half_frame_001.jpg",
            ],
        )


class MainFrameFilenameContractTests(unittest.TestCase):
    @patch("detector.video_io.cv2.VideoCapture")
    def test_extract_main_frames_preserves_main_frame_filename_pattern(
        self,
        mock_video_capture,
    ):
        mock_video_capture.return_value = _FakeCapture(total_frames=100, fps=10.0)
        save_image_fn = MagicMock(return_value=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            output_dir = os.path.join(tmpdir, "debug", "main_frames")
            with open(video_path, "wb"):
                pass

            with patch("detector.video_io.convert_video_codec", return_value=video_path):
                result = extract_main_frames(
                    video_path,
                    num_frames=3,
                    output_dir=output_dir,
                    guardar_frames=True,
                    frame_filename_prefix="debug-run",
                    save_image_fn=save_image_fn,
                )

        saved_paths = [call.args[0] for call in save_image_fn.call_args_list]
        self.assertEqual(
            saved_paths,
            [
                os.path.join(output_dir, "debug-run_main_frame_001_video_000010_used.jpg"),
                os.path.join(output_dir, "debug-run_main_frame_002_video_000039_used.jpg"),
                os.path.join(output_dir, "debug-run_main_frame_003_video_000069_used.jpg"),
            ],
        )
        self.assertEqual(result["frames_paths"], saved_paths)


if __name__ == "__main__":
    unittest.main()
