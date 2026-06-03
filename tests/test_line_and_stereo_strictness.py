"""Focused tests for stricter seam detection and seam-aware stereo splitting."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from detector.line_detection import detect_horizontal_line
from detector.stereo_detection import detect_stereo


class HorizontalLineStrictnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.gray = np.zeros((100, 200), dtype=np.uint8)

    @patch("detector.line_detection._verify_line_with_fft", return_value={"confirmed": False, "confidence": 0.2, "dominance": -0.6})
    @patch("detector.line_detection.cv2.HoughLinesP")
    @patch("detector.line_detection.cv2.morphologyEx")
    @patch("detector.line_detection.cv2.Canny")
    @patch("detector.line_detection.cv2.threshold")
    @patch("detector.line_detection.prepare_frame_for_line_detection")
    def test_rejects_fragmented_center_candidates_even_with_high_coverage(
        self,
        mock_prepare,
        mock_threshold,
        mock_canny,
        mock_morph,
        mock_hough,
        _mock_fft,
    ) -> None:
        mock_prepare.return_value = self.gray
        mock_threshold.return_value = (90.0, self.gray)
        roi = np.zeros((4, 200), dtype=np.uint8)
        mock_canny.return_value = roi
        mock_morph.return_value = roi
        mock_hough.return_value = np.array(
            [
                [[0, 1, 40, 1]],
                [[120, 1, 160, 1]],
            ],
            dtype=np.int32,
        )

        result = detect_horizontal_line(
            self.frame,
            center_tolerance_ratio=0.02,
            search_band_ratio=0.02,
            max_slope=0.05,
            min_coverage_ratio=0.20,
        )

        self.assertFalse(result["has_horizontal_line"])
        quality_gate = result["debug_line_info"]["quality_gate"]
        self.assertFalse(quality_gate["continuity_ok"])

    @patch("detector.line_detection._verify_line_with_fft", return_value={"confirmed": True, "confidence": 0.95, "dominance": 0.80})
    @patch("detector.line_detection.cv2.HoughLinesP")
    @patch("detector.line_detection.cv2.morphologyEx")
    @patch("detector.line_detection.cv2.Canny")
    @patch("detector.line_detection.cv2.threshold")
    @patch("detector.line_detection.prepare_frame_for_line_detection")
    def test_rejects_candidate_that_is_centered_but_not_strictly_centered(
        self,
        mock_prepare,
        mock_threshold,
        mock_canny,
        mock_morph,
        mock_hough,
        _mock_fft,
    ) -> None:
        frame = np.zeros((101, 200, 3), dtype=np.uint8)
        gray = np.zeros((101, 200), dtype=np.uint8)
        mock_prepare.return_value = gray
        mock_threshold.return_value = (90.0, gray)
        roi = np.zeros((4, 200), dtype=np.uint8)
        mock_canny.return_value = roi
        mock_morph.return_value = roi
        mock_hough.return_value = np.array([[[0, 3, 199, 3]]], dtype=np.int32)

        result = detect_horizontal_line(
            frame,
            center_tolerance_ratio=0.02,
            search_band_ratio=0.02,
            max_slope=0.05,
            min_coverage_ratio=0.20,
        )

        self.assertFalse(result["has_horizontal_line"])
        quality_gate = result["debug_line_info"]["quality_gate"]
        self.assertFalse(quality_gate["strict_centered"])


class StereoSeamAwareSplitTests(unittest.TestCase):
    def test_uses_detected_seam_and_guard_band_for_up_down(self) -> None:
        height, width = 101, 40
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        seam_center = 40

        pattern = np.tile(np.arange(width, dtype=np.uint8), (38, 1))
        pattern_rgb = np.stack([pattern, pattern, pattern], axis=2)
        frame[:38, :, :] = pattern_rgb
        frame[42:80, :, :] = pattern_rgb
        frame[80:, :, :] = 255

        seam_result = detect_stereo(
            frames=[frame],
            indices_con_linea=[0],
            similarity_threshold=0.70,
            arrangement="up-down",
            line_centers={0: seam_center},
            seam_guard_ratio=0.02,
            min_valid_half_ratio=0.20,
            edge_similarity_threshold=0.0,
            min_frames_required=1,
            min_stability_ratio=0.0,
        )

        center_split_result = detect_stereo(
            frames=[frame],
            indices_con_linea=[0],
            similarity_threshold=0.70,
            arrangement="up-down",
            seam_guard_ratio=0.02,
            min_valid_half_ratio=0.20,
            edge_similarity_threshold=0.0,
            min_frames_required=1,
            min_stability_ratio=0.0,
        )

        self.assertTrue(seam_result["is_stereo"])
        self.assertFalse(center_split_result["is_stereo"])

    def test_requires_minimum_number_of_seam_frames(self) -> None:
        frame = np.zeros((80, 40, 3), dtype=np.uint8)
        frame[:30, :, :] = 120
        frame[40:70, :, :] = 120

        result = detect_stereo(
            frames=[frame],
            indices_con_linea=[0],
            similarity_threshold=0.70,
            arrangement="up-down",
            line_centers={0: 35},
            seam_guard_ratio=0.02,
            min_valid_half_ratio=0.20,
            edge_similarity_threshold=0.0,
            min_frames_required=2,
            min_stability_ratio=0.0,
        )

        self.assertFalse(result["is_stereo"])
        self.assertEqual(result["frames_evaluados"], 1)
        self.assertEqual(result["min_frames_required"], 2)


if __name__ == "__main__":
    unittest.main()
