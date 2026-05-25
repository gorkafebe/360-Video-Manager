import sys
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

from detector.pipeline import run_detection_with_retries
from workflows.unified_pipeline import _stage_detect_projection


class DetectionRetryTests(unittest.TestCase):
    @patch("detector.pipeline.run_detection_pipeline")
    @patch("detector.pipeline.extract_main_frames")
    def test_run_detection_with_retries_uses_wider_spacing_after_unreliable_motion(
        self,
        mock_extract_main_frames,
        mock_run_detection_pipeline,
    ):
        mock_extract_main_frames.side_effect = [
            {
                "frames": ["frame-1"],
                "frames_metadata": [{"position": 10}],
                "video_name": "video",
            },
            {
                "frames": ["frame-2"],
                "frames_metadata": [{"position": 20}],
                "video_name": "video",
            },
        ]
        mock_run_detection_pipeline.side_effect = [
            {
                "projection_type": "unknown",
                "confidence": 0.2,
                "motion_reliable": False,
                "motion_reliability_reason": "few_valid_pairs:1<4",
                "motion_pairs_valid": 1,
            },
            {
                "projection_type": "cubic",
                "confidence": 0.9,
                "motion_reliable": True,
                "motion_reliability_reason": "",
                "motion_pairs_valid": 5,
            },
        ]

        result, debug_context = run_detection_with_retries("/tmp/video.mp4", num_frames=10)

        self.assertEqual(result["projection_type"], "cubic")
        self.assertEqual(debug_context, {})
        self.assertEqual(mock_run_detection_pipeline.call_count, 2)

        first_attempt = mock_run_detection_pipeline.call_args_list[0].kwargs
        second_attempt = mock_run_detection_pipeline.call_args_list[1].kwargs
        self.assertEqual(first_attempt["num_frames"], 10)
        self.assertEqual(first_attempt["paso_frames_secundarios"], 5)
        self.assertEqual(first_attempt["min_frames_with_line_required"], 7)
        self.assertEqual(second_attempt["num_frames"], 8)
        self.assertEqual(second_attempt["paso_frames_secundarios"], 9)
        self.assertEqual(second_attempt["min_frames_with_line_required"], 6)

    @patch("detector.pipeline.run_detection_with_retries")
    def test_stage_detect_projection_uses_retry_wrapper(self, mock_run_detection_with_retries):
        mock_run_detection_with_retries.return_value = (
            {
                "projection_type": "unknown",
                "confidence": 0.25,
            },
            {"run_dir": "/tmp/debug-run"},
        )

        result = _stage_detect_projection("/tmp/video.mp4", 12, "/tmp/debug")

        self.assertEqual(result["projection_type"], "unknown")
        mock_run_detection_with_retries.assert_called_once_with(
            "/tmp/video.mp4",
            num_frames=12,
            debug_base_dir="/tmp/debug",
        )


if __name__ == "__main__":
    unittest.main()
