import math
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


def _fake_callable(*args, **kwargs):
    return None


fake_cv2 = types.ModuleType("cv2")
fake_cv2.THRESH_BINARY = 0
fake_cv2.THRESH_OTSU = 0
fake_cv2.MORPH_RECT = 0
fake_cv2.MORPH_CLOSE = 0
fake_cv2.LSD_REFINE_STD = 0
fake_cv2.DIST_L12 = 0
fake_cv2.COLOR_BGR2GRAY = 0
fake_cv2.HISTCMP_CORREL = 0
fake_cv2.HISTCMP_BHATTACHARYYA = 0
fake_cv2.threshold = lambda *_a, **_k: (80.0, None)
fake_cv2.Canny = lambda *_a, **_k: None
fake_cv2.getStructuringElement = lambda *_a, **_k: None
fake_cv2.morphologyEx = lambda img, *_a, **_k: img
fake_cv2.__getattr__ = lambda _name: _fake_callable

fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
fake_numpy.mean = lambda values: (sum(values) / len(values)) if values else 0.0
fake_numpy.median = lambda values: sorted(values)[len(values) // 2] if values else 0.0
fake_numpy.pi = 3.141592653589793
fake_numpy.clip = lambda a, a_min, a_max: max(a_min, min(a_max, a))
fake_numpy.sqrt = lambda x: math.sqrt(float(x))
fake_numpy.degrees = lambda v: math.degrees(float(v))
fake_numpy.arctan2 = lambda y, x: math.atan2(float(y), float(x))
fake_numpy.__getattr__ = lambda _name: _fake_callable

sys.modules.setdefault("cv2", fake_cv2)
sys.modules.setdefault("numpy", fake_numpy)

from detector.pipeline import _resolve_motion_feature_flags, run_detection_with_retries
from detector.video_io import FrameExtractorError
from workflows.unified_pipeline import JobOptions, _stage_detect_projection, process_video_job
from core.models import UploadResult


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
        self.assertEqual(first_attempt["min_frames_with_line_required"], 5)  # max(2, int(10*0.55))
        self.assertEqual(second_attempt["num_frames"], 8)
        self.assertEqual(second_attempt["paso_frames_secundarios"], 9)
        self.assertEqual(second_attempt["min_frames_with_line_required"], 4)  # max(2, int(8*0.55))

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
    @patch(
        "detector.pipeline.get_opencv_capabilities",
        return_value={
            "has_optflow_module": True,
            "has_dis": True,
            "has_tvl1": True,
            "has_deepflow": True,
            "has_pcaflow": True,
            "has_sparse_to_dense": True,
            "has_variational_refinement": True,
            "has_usac_magsac": True,
        },
    )
    def test_default_profile_is_high_accuracy(self, _mock_caps):
        flags = _resolve_motion_feature_flags({})
        self.assertEqual(flags["profile"], "high_accuracy")
        self.assertTrue(flags["enable_refinement"])
        self.assertTrue(flags["enable_fb_check"])
        self.assertTrue(flags["enable_geometry_evidence"])
        self.assertEqual(flags["flow_algorithm"], "tvl1")
        self.assertTrue(flags["flow_fallback_chain"])
        self.assertIn("tier_b_flow_algorithms", flags["feature_tiers"])
        self.assertIn("tier_c_flow_algorithms", flags["feature_tiers"])

    @patch(
        "detector.pipeline.get_opencv_capabilities",
        return_value={
            "has_optflow_module": False,
            "has_dis": False,
            "has_tvl1": False,
            "has_deepflow": False,
            "has_pcaflow": False,
            "has_sparse_to_dense": False,
            "has_variational_refinement": False,
            "has_usac_magsac": False,
        },
    )
    def test_baseline_profile_keeps_safe_defaults(self, _mock_caps):
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
        self.assertEqual(flags["flow_algorithm"], "farneback")
        self.assertEqual(flags["flow_fallback_chain"], ["farneback"])

    @patch(
        "detector.pipeline.get_opencv_capabilities",
        return_value={
            "has_optflow_module": True,
            "has_dis": True,
            "has_tvl1": False,
            "has_deepflow": False,
            "has_pcaflow": False,
            "has_sparse_to_dense": False,
            "has_variational_refinement": False,
            "has_usac_magsac": False,
        },
    )
    def test_baseline_profile_allows_dis_when_requested_and_available(self, _mock_caps):
        flags = _resolve_motion_feature_flags(
            {
                "motion_rollout_profile": "baseline",
                "flow_algorithm": "dis",
            }
        )
        self.assertEqual(flags["flow_algorithm"], "dis")
        self.assertEqual(flags["flow_fallback_chain"][0], "dis")
        self.assertEqual(flags["flow_fallback_chain"][-1], "farneback")

    @patch(
        "detector.pipeline.get_opencv_capabilities",
        return_value={
            "has_optflow_module": True,
            "has_dis": True,
            "has_tvl1": True,
            "has_deepflow": True,
            "has_pcaflow": True,
            "has_sparse_to_dense": True,
            "has_variational_refinement": True,
            "has_usac_magsac": True,
        },
    )
    def test_robust_profile_enables_fb_and_geometry(self, _mock_caps):
        flags = _resolve_motion_feature_flags({"motion_rollout_profile": "robust"})
        self.assertEqual(flags["profile"], "robust")
        self.assertTrue(flags["enable_fb_check"])
        self.assertTrue(flags["enable_geometry_evidence"])
        self.assertTrue(flags["enable_refinement"])
        self.assertEqual(flags["flow_algorithm"], "tvl1")

    @patch(
        "detector.pipeline.get_opencv_capabilities",
        return_value={
            "has_optflow_module": True,
            "has_dis": True,
            "has_tvl1": True,
            "has_deepflow": True,
            "has_pcaflow": True,
            "has_sparse_to_dense": True,
            "has_variational_refinement": True,
            "has_usac_magsac": True,
        },
    )
    def test_high_accuracy_profile_enables_all(self, _mock_caps):
        flags = _resolve_motion_feature_flags({"motion_rollout_profile": "high_accuracy"})
        self.assertEqual(flags["profile"], "high_accuracy")
        self.assertTrue(flags["enable_refinement"])
        self.assertTrue(flags["enable_fb_check"])
        self.assertTrue(flags["enable_geometry_evidence"])

    @patch(
        "detector.pipeline.get_opencv_capabilities",
        return_value={
            "has_optflow_module": True,
            "has_dis": False,
            "has_tvl1": True,
            "has_deepflow": True,
            "has_pcaflow": True,
            "has_sparse_to_dense": True,
            "has_variational_refinement": True,
            "has_usac_magsac": True,
        },
    )
    def test_high_accuracy_prefers_tier_b_over_requested_tier_c(self, _mock_caps):
        flags = _resolve_motion_feature_flags(
            {
                "motion_rollout_profile": "high_accuracy",
                "flow_algorithm": "deepflow",
            }
        )
        self.assertEqual(flags["flow_algorithm"], "tvl1")
        self.assertEqual(flags["flow_fallback_chain"][0], "tvl1")


class UnifiedPipelineNormalizationModeTests(unittest.TestCase):
    @patch("workflows.unified_pipeline._stage_preview_frames", return_value=[])
    @patch(
        "workflows.unified_pipeline._stage_detect_projection",
        return_value={"projection_type": "equirectangular", "confidence": 0.9, "stats": {}},
    )
    @patch("workflows.unified_pipeline._stage_normalize_codec", return_value="/tmp/compat.mp4")
    def test_detection_path_skips_full_normalization_by_default(
        self,
        mock_normalize,
        _mock_detect,
        _mock_preview,
    ):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            result = process_video_job(
                JobOptions(
                    local_video_path=tmp_video.name,
                    convert_if_needed=False,
                    save_manifest=False,
                )
            )
        self.assertTrue(result.success)
        mock_normalize.assert_not_called()
        self.assertEqual(result.normalized_video_path, tmp_video.name)

    @patch("workflows.unified_pipeline._stage_preview_frames", return_value=[])
    @patch(
        "workflows.unified_pipeline._stage_detect_projection",
        return_value={"projection_type": "equirectangular", "confidence": 0.9, "stats": {}},
    )
    @patch("workflows.unified_pipeline._stage_normalize_codec", return_value="/tmp/compat.mp4")
    def test_force_flag_keeps_legacy_full_normalization(
        self,
        mock_normalize,
        _mock_detect,
        _mock_preview,
    ):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            result = process_video_job(
                JobOptions(
                    local_video_path=tmp_video.name,
                    convert_if_needed=False,
                    save_manifest=False,
                    force_full_codec_normalization=True,
                )
            )
        self.assertTrue(result.success)
        mock_normalize.assert_called_once_with(tmp_video.name)
        self.assertEqual(result.normalized_video_path, "/tmp/compat.mp4")

    @patch("workflows.unified_pipeline._stage_preview_frames", return_value=[])
    @patch("workflows.unified_pipeline._stage_normalize_codec", return_value="/tmp/compat_retry.mp4")
    @patch(
        "workflows.unified_pipeline._stage_detect_projection",
        side_effect=[
            FrameExtractorError(
                "No se pudieron extraer frames del vídeo",
                code="frame_extraction_timeout",
                details={"attempts": [{"attempt": 1}]},
            ),
            FrameExtractorError(
                "No se pudieron extraer frames del vídeo",
                code="frame_extraction_timeout",
                details={"attempts": [{"attempt": 1}, {"attempt": 2}]},
            ),
        ],
    )
    def test_detection_failure_after_normalization_retry_reports_enriched_error(
        self,
        mock_detect,
        mock_normalize,
        _mock_preview,
    ):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            result = process_video_job(
                JobOptions(
                    local_video_path=tmp_video.name,
                    convert_if_needed=False,
                    save_manifest=False,
                )
            )

        self.assertFalse(result.success)
        self.assertIn("after codec-normalization retry", result.error)
        self.assertIn("code=frame_extraction_timeout", result.error)
        self.assertEqual(mock_detect.call_count, 2)
        mock_normalize.assert_called_once_with(tmp_video.name)


class UnifiedPipelineUploadSelectionTests(unittest.TestCase):
    @patch(
        "workflows.unified_pipeline._stage_upload",
        return_value=UploadResult(success=True, friendly_token="tok"),
    )
    @patch(
        "workflows.unified_pipeline._stage_detect_projection",
        return_value={"projection_type": "eac", "confidence": 0.95, "stats": {}},
    )
    @patch("workflows.unified_pipeline._stage_convert_to_equirectangular")
    @patch("workflows.unified_pipeline._stage_preview_frames", return_value=[])
    @patch("workflows.unified_pipeline._log_codec_telemetry", return_value={})
    def test_upload_prefers_converted_path_when_available(
        self,
        _mock_codec,
        _mock_preview,
        mock_convert,
        _mock_detect,
        mock_upload,
    ):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video, tempfile.NamedTemporaryFile(
            suffix=".mp4"
        ) as tmp_converted:
            mock_convert.return_value = tmp_converted.name
            result = process_video_job(
                JobOptions(
                    local_video_path=tmp_video.name,
                    output_dir="/tmp",
                    upload=True,
                    convert_if_needed=True,
                    save_manifest=False,
                )
            )
        self.assertTrue(result.success)
        self.assertEqual(result.converted_video_path, tmp_converted.name)
        mock_upload.assert_called_once()
        self.assertEqual(mock_upload.call_args.kwargs["video_path"], tmp_converted.name)


from detector.pipeline import _build_detection_retry_plan


class BuildDetectionRetryPlanTests(unittest.TestCase):
    """Tests for the proportional min_frames_with_line_required in _build_detection_retry_plan."""

    def test_min_lines_proportional_for_10_frames(self):
        plan = _build_detection_retry_plan(10)
        # frame_plan = [10, 8, 6]; min_lines = max(2, int(fc*0.55))
        self.assertEqual(plan[0]["num_frames"], 10)
        self.assertEqual(plan[0]["min_frames_with_line_required"], 5)  # int(10*0.55)=5
        self.assertEqual(plan[1]["num_frames"], 8)
        self.assertEqual(plan[1]["min_frames_with_line_required"], 4)  # int(8*0.55)=4
        self.assertEqual(plan[2]["num_frames"], 6)
        self.assertEqual(plan[2]["min_frames_with_line_required"], 3)  # int(6*0.55)=3

    def test_min_lines_proportional_for_8_frames(self):
        plan = _build_detection_retry_plan(8)
        self.assertEqual(plan[0]["num_frames"], 8)
        self.assertEqual(plan[0]["min_frames_with_line_required"], 4)  # int(8*0.55)=4
        self.assertEqual(plan[1]["num_frames"], 6)
        self.assertEqual(plan[1]["min_frames_with_line_required"], 3)  # int(6*0.55)=3

    def test_min_lines_proportional_for_6_frames(self):
        plan = _build_detection_retry_plan(6)
        self.assertEqual(plan[0]["num_frames"], 6)
        self.assertEqual(plan[0]["min_frames_with_line_required"], 3)  # int(6*0.55)=3

    def test_min_lines_never_below_2(self):
        plan = _build_detection_retry_plan(2)
        for entry in plan:
            self.assertGreaterEqual(entry["min_frames_with_line_required"], 2)

    def test_secondary_plan_still_applied(self):
        plan = _build_detection_retry_plan(10)
        self.assertEqual(plan[0]["paso_frames_secundarios"], 5)
        self.assertEqual(plan[1]["paso_frames_secundarios"], 9)
        self.assertEqual(plan[2]["paso_frames_secundarios"], 13)


class MotionGateRatioFallbackTests(unittest.TestCase):
    """Tests for the ratio-aware motion gate in run_detection_pipeline."""

    def _make_fake_cap(self, frame_count: int = 5000):
        """Return a MagicMock VideoCapture that reports *frame_count* frames."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = float(frame_count)
        return cap

    def _make_line_response(self, has_line: bool, fft_confirmed: bool = True, confidence: float = 0.6) -> dict:
        """Return a single detect_horizontal_line mock response."""
        return {
            "has_horizontal_line": has_line,
            "debug_line_info": {},
            "fft_confirmed": fft_confirmed,
            "fft_confidence": confidence,
            "seam_center": 100,
            "has_vertical_line": False,
            "vertical_seam_center": None,
        }

    def _make_equi_aggregate(self) -> dict:
        """Return an aggregate_equirectangular_evidence mock response with no evidence."""
        return {
            "confidence": 0.0,
            "is_strong_evidence": False,
            "usable_frames": 6,
            "strong_frames": 0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "final_score": 0.0,
            "reason": "insufficient_evidence",
        }

    def _make_stereo_result(self) -> dict:
        """Return a detect_stereo mock response indicating no stereo."""
        return {
            "is_stereo": False,
            "frames_evaluados": 0,
            "frames_match": 0,
            "frames_no_match": 0,
            "avg_similarity": 0.0,
            "avg_bhattacharyya": 1.0,
            "avg_edge_similarity": 0.0,
            "avg_combined_score": 0.0,
            "match_ratio": 0.0,
            "min_match_ratio": 0.60,
            "longest_mismatch_streak": 0,
            "stability_ratio": 0.0,
            "min_frames_required": 3,
            "min_stability_ratio": 0.5,
            "edge_similarity_threshold": 0.3,
            "bhattacharyya_threshold": 0.30,
            "frame_details": [],
        }

    @patch("detector.pipeline.cv2.VideoCapture")
    @patch("detector.pipeline._classify_non_equirectangular")
    @patch("detector.pipeline.detect_stereo")
    @patch("detector.pipeline.detect_horizontal_line")
    @patch("detector.pipeline.frame_esta_en_negro")
    @patch("detector.pipeline.compute_frame_equirectangular_evidence")
    @patch("detector.pipeline.aggregate_equirectangular_evidence")
    @patch("detector.pipeline.evaluar_suficiencia_datos", return_value=[])
    @patch("detector.pipeline.extract_secondary_frames")
    @patch("detector.pipeline.save_line_visual_debug")
    @patch("detector.pipeline.save_frame_debug")
    def test_ratio_gate_fires_when_lines_below_absolute_threshold(
        self,
        _mock_save_frame,
        _mock_save_line,
        mock_extract_secondary,
        _mock_evaluar,
        mock_aggregate_equi,
        mock_compute_equi,
        mock_is_black,
        mock_detect_line,
        mock_detect_stereo,
        mock_classify,
        mock_video_capture,
    ):
        """5/6 valid frames with lines should enter motion analysis even when min_lines=6."""
        from detector.pipeline import run_detection_pipeline

        mock_video_capture.return_value = self._make_fake_cap()

        # 6 non-black frames, 5 have a horizontal line (ratio=0.833 >= 0.50)
        mock_is_black.return_value = False
        mock_detect_line.side_effect = [
            self._make_line_response(v) for v in [True, True, True, True, True, False]
        ]
        mock_compute_equi.return_value = {"confidence": 0.8, "is_equirectangular": False}
        mock_aggregate_equi.return_value = self._make_equi_aggregate()

        # secondary sequences: provide one sequence per main frame
        fake_seq = [{"position": i * 10 + 1, "frame": object(), "valid": True} for i in range(3)]
        mock_extract_secondary.return_value = [fake_seq] * 6

        mock_detect_stereo.return_value = self._make_stereo_result()

        mock_classify.return_value = {
            "classification": "eac",
            "reliable": True,
            "reliability_reason": "ok",
            "pares_totales": 3,
            "pares_validos": 3,
            "pares_invalidos": 0,
            "motion_confidence": 0.9,
            "avg_eac_score": 0.8,
            "avg_cubic_score": 0.2,
            "score_margin": 0.6,
            "total_regiones_validas": 12,
            "total_regiones_invalidas": 0,
            "motion_low_detected": False,
            "motion_low_ratio": 0.0,
            "motion_mean_pair_magnitude": 2.5,
            "motion_mean_pair_active_ratio": 0.15,
            "motion_pair_gaps_used": [5],
            "pair_geometry_mean_quality": 0.7,
        }

        frames = [MagicMock(**{"shape": (720, 1440, 3)}) for _ in range(6)]
        frames_metadata = [{"position": i * 100} for i in range(6)]

        result = run_detection_pipeline(
            "/tmp/video.mp4",
            frames=frames,
            frames_metadata=frames_metadata,
            video_name="test_video",
            min_frames_with_line_required=6,  # absolute threshold exceeds frames_with_line=5
            min_valid_pairs=2,
        )

        # classify_non_equirectangular must have been called (gate was passed via ratio)
        mock_classify.assert_called_once()
        self.assertEqual(result["projection_type"], "eac")

    @patch("detector.pipeline.cv2.VideoCapture")
    @patch("detector.pipeline._classify_non_equirectangular")
    @patch("detector.pipeline.detect_stereo")
    @patch("detector.pipeline.detect_horizontal_line")
    @patch("detector.pipeline.frame_esta_en_negro")
    @patch("detector.pipeline.compute_frame_equirectangular_evidence")
    @patch("detector.pipeline.aggregate_equirectangular_evidence")
    @patch("detector.pipeline.evaluar_suficiencia_datos", return_value=[])
    @patch("detector.pipeline.extract_secondary_frames")
    @patch("detector.pipeline.save_line_visual_debug")
    @patch("detector.pipeline.save_frame_debug")
    def test_gate_still_blocks_when_ratio_is_below_50_percent(
        self,
        _mock_save_frame,
        _mock_save_line,
        mock_extract_secondary,
        _mock_evaluar,
        mock_aggregate_equi,
        mock_compute_equi,
        mock_is_black,
        mock_detect_line,
        mock_detect_stereo,
        mock_classify,
        mock_video_capture,
    ):
        """2/6 valid frames with lines (ratio=0.33) should NOT trigger motion analysis."""
        from detector.pipeline import run_detection_pipeline

        mock_video_capture.return_value = self._make_fake_cap()
        mock_is_black.return_value = False
        mock_detect_line.side_effect = [
            self._make_line_response(v, fft_confirmed=False, confidence=0.4)
            for v in [True, True, False, False, False, False]
        ]
        mock_compute_equi.return_value = {"confidence": 0.4, "is_equirectangular": False}
        mock_aggregate_equi.return_value = self._make_equi_aggregate()
        mock_detect_stereo.return_value = self._make_stereo_result()
        mock_extract_secondary.return_value = [[]] * 6

        frames = [MagicMock(**{"shape": (720, 1440, 3)}) for _ in range(6)]
        frames_metadata = [{"position": i * 100} for i in range(6)]

        result = run_detection_pipeline(
            "/tmp/video.mp4",
            frames=frames,
            frames_metadata=frames_metadata,
            video_name="test_video",
            min_frames_with_line_required=6,
            min_valid_pairs=2,
        )

        # classify_non_equirectangular must NOT have been called
        mock_classify.assert_not_called()
        self.assertIn("insufficient_structural_frames", result.get("motion_reliability_reason", ""))


if __name__ == "__main__":
    unittest.main()
