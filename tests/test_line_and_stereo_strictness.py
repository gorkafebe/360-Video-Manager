"""Focused tests for stricter seam detection and seam-aware stereo policy."""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import patch


class _FakeImage:
    def __init__(self, height: int, width: int):
        self.shape = (height, width)
        self.size = height * width

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self
        y_sel, x_sel = key
        h = self.shape[0] if isinstance(y_sel, slice) else 1
        w = self.shape[1] if isinstance(x_sel, slice) else 1
        if isinstance(y_sel, slice):
            h = len(range(*y_sel.indices(self.shape[0])))
        if isinstance(x_sel, slice):
            w = len(range(*x_sel.indices(self.shape[1])))
        return _FakeImage(max(1, h), max(1, w))


def _fake_clip(value, low, high):
    return max(low, min(high, float(value)))


fake_cv2 = types.ModuleType("cv2")
fake_cv2.THRESH_BINARY = 0
fake_cv2.THRESH_OTSU = 0
fake_cv2.MORPH_RECT = 0
fake_cv2.MORPH_CLOSE = 0
fake_cv2.DIST_L12 = 0
fake_cv2.LSD_REFINE_STD = 0
fake_cv2.COLOR_BGR2GRAY = 0
fake_cv2.HISTCMP_CORREL = 0
fake_cv2.HISTCMP_BHATTACHARYYA = 0
fake_cv2.threshold = lambda *_args, **_kwargs: (80.0, _FakeImage(4, 200))
fake_cv2.Canny = lambda *_args, **_kwargs: _FakeImage(4, 200)
fake_cv2.getStructuringElement = lambda *_args, **_kwargs: object()
fake_cv2.morphologyEx = lambda img, *_args, **_kwargs: img
fake_cv2.HoughLinesP = lambda *_args, **_kwargs: None
fake_cv2.createLineSegmentDetector = lambda *_args, **_kwargs: types.SimpleNamespace(
    detect=lambda *_a, **_k: (None,)
)
fake_cv2.fitLine = lambda *_args, **_kwargs: (1.0, 0.0, 0.0, 0.0)
fake_cv2.cvtColor = lambda image, *_args, **_kwargs: image
fake_cv2.calcHist = lambda *_args, **_kwargs: 0.0
fake_cv2.normalize = lambda *_args, **_kwargs: None
fake_cv2.compareHist = lambda *_args, **_kwargs: 0.0
fake_cv2.VideoCapture = lambda *_args, **_kwargs: None

fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
fake_numpy.pi = math.pi
fake_numpy.float32 = float
fake_numpy.sqrt = lambda v: math.sqrt(float(v))
fake_numpy.degrees = lambda v: math.degrees(float(v))
fake_numpy.arctan2 = lambda y, x: math.atan2(float(y), float(x))
fake_numpy.clip = _fake_clip
fake_numpy.mean = lambda values: (sum(values) / len(values)) if values else 0.0
fake_numpy.min = min
fake_numpy.max = max
fake_numpy.sum = sum
fake_numpy.sin = math.sin
fake_numpy.cos = math.cos
fake_numpy.radians = math.radians
fake_numpy.array = lambda values, dtype=None: list(values)
fake_numpy.where = lambda *_args, **_kwargs: ([], [])
fake_numpy.column_stack = lambda *_args, **_kwargs: []
fake_numpy.fft = types.SimpleNamespace(fft2=lambda x: x, fftshift=lambda x: x)
fake_numpy.log1p = lambda x: x
fake_numpy.abs = abs
fake_numpy.count_nonzero = lambda *_args, **_kwargs: 0

sys.modules.setdefault("cv2", fake_cv2)
sys.modules.setdefault("numpy", fake_numpy)

from detector.line_detection import detect_horizontal_line
from detector.stereo_detection import detect_stereo


class HorizontalLineStrictnessTests(unittest.TestCase):
    @patch("detector.line_detection._verify_line_with_fft", return_value={"confirmed": False, "confidence": 0.2, "dominance": -0.6})
    @patch("detector.line_detection.cv2.HoughLinesP", return_value=[[[0, 1, 40, 1]], [[120, 1, 160, 1]]])
    @patch("detector.line_detection.prepare_frame_for_line_detection", return_value=_FakeImage(100, 200))
    def test_rejects_fragmented_center_candidates_even_with_high_coverage(
        self,
        _mock_prepare,
        _mock_hough,
        _mock_fft,
    ) -> None:
        result = detect_horizontal_line(
            _FakeImage(100, 200),
            center_tolerance_ratio=0.02,
            search_band_ratio=0.02,
            max_slope=0.05,
            min_coverage_ratio=0.20,
        )
        self.assertFalse(result["has_horizontal_line"])
        self.assertFalse(result["debug_line_info"]["quality_gate"]["continuity_ok"])

    @patch("detector.line_detection._verify_line_with_fft", return_value={"confirmed": True, "confidence": 0.95, "dominance": 0.8})
    @patch("detector.line_detection.cv2.HoughLinesP", return_value=[[[0, 4, 199, 4]]])
    @patch("detector.line_detection.prepare_frame_for_line_detection", return_value=_FakeImage(101, 200))
    def test_rejects_candidate_that_is_centered_but_not_strictly_centered(
        self,
        _mock_prepare,
        _mock_hough,
        _mock_fft,
    ) -> None:
        result = detect_horizontal_line(
            _FakeImage(101, 200),
            center_tolerance_ratio=0.02,
            search_band_ratio=0.02,
            max_slope=0.05,
            min_coverage_ratio=0.20,
        )
        self.assertFalse(result["has_horizontal_line"])
        self.assertFalse(result["debug_line_info"]["quality_gate"]["strict_centered"])


class StereoPolicyTests(unittest.TestCase):
    @patch("detector.stereo_detection._extract_seam_regions")
    @patch("detector.stereo_detection._compute_edge_similarity")
    @patch("detector.stereo_detection.compare_histograms")
    @patch("detector.stereo_detection.compute_histogram")
    def test_detected_seam_regions_enable_match_while_center_split_fails(
        self,
        _mock_hist,
        mock_compare_hist,
        mock_edge,
        mock_extract,
    ) -> None:
        frame = types.SimpleNamespace(shape=(80, 40, 3))
        _mock_hist.side_effect = [0.0, 0.0, 0.0, 0.0]
        mock_compare_hist.side_effect = [
            {"correlation": 0.95, "bhattacharyya": 0.08},  # seam-aware
            {"correlation": 0.45, "bhattacharyya": 0.45},  # center-split
        ]
        mock_edge.side_effect = [0.7, 0.15]
        mock_extract.side_effect = [("A", "B"), ("C", "D")]

        seam_result = detect_stereo(
            frames=[frame],
            indices_con_linea=[0],
            similarity_threshold=0.70,
            arrangement="up-down",
            line_centers={0: 40},
            edge_similarity_threshold=0.08,
            min_frames_required=1,
            min_stability_ratio=0.0,
        )
        center_result = detect_stereo(
            frames=[frame],
            indices_con_linea=[0],
            similarity_threshold=0.70,
            arrangement="up-down",
            edge_similarity_threshold=0.08,
            min_frames_required=1,
            min_stability_ratio=0.0,
        )

        self.assertTrue(seam_result["is_stereo"])
        self.assertFalse(center_result["is_stereo"])

    @patch("detector.stereo_detection._extract_seam_regions", return_value=("A", "B"))
    @patch("detector.stereo_detection._compute_edge_similarity", return_value=0.8)
    @patch("detector.stereo_detection.compare_histograms", return_value={"correlation": 0.95, "bhattacharyya": 0.05})
    @patch("detector.stereo_detection.compute_histogram", side_effect=[0.0, 0.0])
    def test_requires_minimum_number_of_seam_frames(
        self,
        _mock_hist,
        _mock_compare,
        _mock_edge,
        _mock_extract,
    ) -> None:
        frame = types.SimpleNamespace(shape=(80, 40, 3))
        result = detect_stereo(
            frames=[frame],
            indices_con_linea=[0],
            similarity_threshold=0.70,
            arrangement="up-down",
            line_centers={0: 35},
            min_frames_required=2,
            min_stability_ratio=0.0,
        )
        self.assertFalse(result["is_stereo"])
        self.assertEqual(result["frames_evaluados"], 1)
        self.assertEqual(result["min_frames_required"], 2)


if __name__ == "__main__":
    unittest.main()
