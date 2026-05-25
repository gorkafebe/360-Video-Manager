"""Policy tests for non-equirectangular motion classification."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from detector.pipeline import _classify_non_equirectangular


def _pair_result(
    *,
    decision,
    score_eac_fused: float = 0.62,
    score_cubic_fused: float = 0.38,
    pair_mean_magnitude: float = 0.90,
    pair_mean_active_ratio: float = 0.04,
    pair_geometry_quality: float = 0.10,
):
    return {
        "decision": decision,
        "score_eac": score_eac_fused,
        "score_cubic": score_cubic_fused,
        "score_eac_fused": score_eac_fused,
        "score_cubic_fused": score_cubic_fused,
        "valid_regions": 3 if decision is not None else 0,
        "invalid_regions": 3 if decision is not None else 0,
        "pair_mean_magnitude": pair_mean_magnitude,
        "pair_mean_active_ratio": pair_mean_active_ratio,
        "pair_motion_strength": pair_mean_magnitude * pair_mean_active_ratio,
        "pair_fb_valid_ratio": 1.0,
        "pair_geometry_quality": pair_geometry_quality,
        "pair_geometry_inlier_ratio": 0.0,
        "pair_geometry_reprojection_error": 999.0,
        "status": "used" if decision is not None else "discarded",
        "reason": "" if decision is not None else "insufficient_layout_data",
        "flow": None,
        "region_bounds": [],
        "region_info": {},
    }


class MotionClassificationPolicyTests(unittest.TestCase):
    def test_forces_classification_with_consistent_evidence(self):
        side_effect = [
            _pair_result(decision=True, pair_geometry_quality=0.55),  # short quick pair 1
            _pair_result(decision=True, pair_geometry_quality=0.55),  # short quick pair 2
            _pair_result(decision=True, pair_geometry_quality=0.55),  # short quick pair 3
            _pair_result(decision=True, pair_geometry_quality=0.55),  # long gap 1
            _pair_result(decision=True, pair_geometry_quality=0.55),  # long gap 2
            _pair_result(decision=True, pair_geometry_quality=0.55),  # long gap 3
        ]

        with patch("detector.pipeline._analyze_motion_pair_detailed", side_effect=side_effect):
            with patch(
                "detector.pipeline.np.clip",
                side_effect=lambda value, low, high: max(low, min(high, value)),
                create=True,
            ):
                result = _classify_non_equirectangular(
                    secuencias=[[object(), object(), object(), object()]],
                    min_valid_pairs=10,  # keep reliability gate strict
                    min_motion_confidence=0.70,
                    seam_support_ratio=0.95,
                    confidence_rollout_profile="high_accuracy",
                )

        self.assertEqual(result["classification"], "eac")
        self.assertTrue(result["reliable"])
        self.assertIn("forced_by_consistent_evidence", result["reliability_reason"])

    def test_keeps_unknown_when_evidence_is_contradictory(self):
        side_effect = [
            _pair_result(decision=True, score_eac_fused=0.52, score_cubic_fused=0.48, pair_geometry_quality=0.05),
            _pair_result(decision=False, score_eac_fused=0.48, score_cubic_fused=0.52, pair_geometry_quality=0.05),
            _pair_result(decision=True, score_eac_fused=0.52, score_cubic_fused=0.48, pair_geometry_quality=0.05),
            _pair_result(decision=None),  # long gap discarded
            _pair_result(decision=None),  # long gap discarded
            _pair_result(decision=None),  # long gap discarded
        ]

        with patch("detector.pipeline._analyze_motion_pair_detailed", side_effect=side_effect):
            with patch(
                "detector.pipeline.np.clip",
                side_effect=lambda value, low, high: max(low, min(high, value)),
                create=True,
            ):
                result = _classify_non_equirectangular(
                    secuencias=[[object(), object(), object(), object()]],
                    ambiguity_gap=0.40,
                    seam_support_ratio=0.55,
                    confidence_rollout_profile="high_accuracy",
                )

        self.assertEqual(result["classification"], "unknown")
        self.assertFalse(result["reliable"])


if __name__ == "__main__":
    unittest.main()
