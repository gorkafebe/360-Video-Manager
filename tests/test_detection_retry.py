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

from detector.pipeline import _resolve_motion_feature_flags, run_detection_with_retries
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

    @patch("detector.pipeline.run_detection_pipeline")
    @patch("detector.pipeline.extract_main_frames")
    def test_run_detection_with_retries_aggressively_increases_spacing_for_low_motion(
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
                "motion_reliability_reason": "low_motion_detected:ratio=0.90",
                "motion_low_detected": True,
                "motion_mean_pair_magnitude": 0.9,
                "motion_mean_pair_active_ratio": 0.03,
            },
            {
                "projection_type": "cubic",
                "confidence": 0.9,
                "motion_reliable": True,
                "motion_reliability_reason": "",
                "motion_low_detected": False,
                "motion_mean_pair_magnitude": 1.4,
                "motion_mean_pair_active_ratio": 0.08,
            },
        ]

        result, _debug_context = run_detection_with_retries("/tmp/video.mp4", num_frames=10)

        self.assertEqual(result["projection_type"], "cubic")
        self.assertEqual(mock_run_detection_pipeline.call_count, 2)
        first_attempt = mock_run_detection_pipeline.call_args_list[0].kwargs
        second_attempt = mock_run_detection_pipeline.call_args_list[1].kwargs
        self.assertEqual(first_attempt["paso_frames_secundarios"], 5)
        self.assertEqual(second_attempt["paso_frames_secundarios"], 11)

    @patch("detector.pipeline.run_detection_pipeline")
    @patch("detector.pipeline.extract_main_frames")
    def test_run_detection_with_retries_stops_early_when_low_motion_signal_stalls(
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
            {
                "frames": ["frame-3"],
                "frames_metadata": [{"position": 30}],
                "video_name": "video",
            },
        ]
        mock_run_detection_pipeline.side_effect = [
            {
                "projection_type": "unknown",
                "confidence": 0.2,
                "motion_reliable": False,
                "motion_reliability_reason": "low_motion_detected:ratio=0.90",
                "motion_low_detected": True,
                "motion_mean_pair_magnitude": 1.0,
                "motion_mean_pair_active_ratio": 0.04,
            },
            {
                "projection_type": "unknown",
                "confidence": 0.2,
                "motion_reliable": False,
                "motion_reliability_reason": "low_motion_detected:ratio=0.88",
                "motion_low_detected": True,
                "motion_mean_pair_magnitude": 1.0,
                "motion_mean_pair_active_ratio": 0.04,
            },
            {
                "projection_type": "eac",
                "confidence": 0.8,
                "motion_reliable": True,
                "motion_reliability_reason": "",
            },
        ]

        result, _debug_context = run_detection_with_retries("/tmp/video.mp4", num_frames=10)

        self.assertEqual(result["projection_type"], "unknown")
        self.assertEqual(mock_run_detection_pipeline.call_count, 2)

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


class MotionFeatureFlagResolutionTests(unittest.TestCase):
    def test_baseline_profile_keeps_safe_defaults(self):
        flags = _resolve_motion_feature_flags(
            {
                "motion_rollout_profile": "baseline",
                "flow_enable_refinement": False,
                "flow_enable_fb_check": False,
                "enable_geometry_evidence": False,
                "flow_fb_threshold": 1.5,
                "geometry_evidence_weight": 0.2,
            }
        )
        self.assertEqual(flags["profile"], "baseline")
        self.assertFalse(flags["enable_refinement"])
        self.assertFalse(flags["enable_fb_check"])
        self.assertFalse(flags["enable_geometry_evidence"])
        self.assertAlmostEqual(flags["fb_threshold"], 1.5)

    def test_robust_profile_enables_fb_and_geometry(self):
        flags = _resolve_motion_feature_flags({"motion_rollout_profile": "robust"})
        self.assertEqual(flags["profile"], "robust")
        self.assertTrue(flags["enable_fb_check"])
        self.assertTrue(flags["enable_geometry_evidence"])
        self.assertFalse(flags["enable_refinement"])

    def test_high_accuracy_profile_enables_all(self):
        flags = _resolve_motion_feature_flags({"motion_rollout_profile": "high_accuracy"})
        self.assertEqual(flags["profile"], "high_accuracy")
        self.assertTrue(flags["enable_refinement"])
        self.assertTrue(flags["enable_fb_check"])
        self.assertTrue(flags["enable_geometry_evidence"])


if __name__ == "__main__":
    unittest.main()
