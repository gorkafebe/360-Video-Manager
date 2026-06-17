"""Unit tests for detector/line_detection.py quality-gate logic.

Uses the same mocking pattern as test_line_and_stereo_strictness.py:
  - prepare_frame_for_line_detection → _FakeImage
  - cv2.HoughLinesP → synthetic segment data
  - _verify_line_with_fft / _verify_vertical_line_with_fft → controlled FFT result

Coordinate conventions (mirror the production code):
  Horizontal ROI: full-width, band around frame centre-y.
    HoughLinesP returns (x_frame, y_roi) — x is already in frame space; y has y_top added.
    So y=band_half in HoughLinesP output corresponds to the frame vertical centre.
  Vertical ROI: full-height, band around frame centre-x.
    HoughLinesP returns (x_roi, y_frame) — y is already in frame space; x has x_left added.
    So x=band_half in HoughLinesP output corresponds to the frame horizontal centre.

For the default parameters (_H=200, _W=600, search_band_ratio=0.02):
  band_half = max(2, round(200*0.02*0.5)) = 2  →  ROI centre at y=2

For the default vertical parameters (_VH=600, _VW=200, search_band_ratio=0.02):
  band_half = max(2, round(200*0.02*0.5)) = 2  →  ROI centre at x=2
"""

import unittest
from unittest.mock import patch

from detector.line_detection import detect_horizontal_line, detect_vertical_line


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


# --- Horizontal frame geometry (200px tall × 600px wide) ---
# band_half = max(2, round(200*0.02*0.5)) = 2  →  y_top = 98, ROI centre at y_roi=2
_H, _W = 200, 600
_FRAME = _FakeImage(_H, _W)
_H_CENTER_ROI_Y = 2   # ROI y that maps to frame y=100 (centre)

# --- Vertical frame geometry (600px tall × 200px wide) ---
# band_half = max(2, round(200*0.02*0.5)) = 2  →  x_left = 98, ROI centre at x_roi=2
_VH, _VW = 600, 200
_VFRAME = _FakeImage(_VH, _VW)
_V_CENTER_ROI_X = 2  # ROI x that maps to frame x=100 (centre)

# FFT stub results
_FFT_OK = {"confirmed": True, "confidence": 0.90, "dominance": 0.50}
_FFT_NO = {"confirmed": False, "confidence": 0.10, "dominance": -0.50}

# Profile verifier stub results
_PROFILE_OK = {"confirmed": True, "coverage_ratio": 0.90, "peak_prominence": 8.5, "peak_index": 2}
_PROFILE_FAIL = {"confirmed": False, "coverage_ratio": 0.05, "peak_prominence": 1.5, "peak_index": 1}


def _hline(roi_y, x1=0, x2=None):
    """Single HoughLinesP segment. x coords are frame-absolute; roi_y is ROI-relative."""
    x2 = x2 if x2 is not None else _W - 1
    return [[[x1, roi_y, x2, roi_y]]]


def _vline(roi_x, y1=0, y2=None):
    """Single HoughLinesP segment. y coords are frame-absolute; roi_x is ROI-relative."""
    y2 = y2 if y2 is not None else _VH - 1
    return [[[roi_x, y1, roi_x, y2]]]


# ---------------------------------------------------------------------------
# Horizontal detection tests
# ---------------------------------------------------------------------------

class HorizontalDetectionTests(unittest.TestCase):

    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_NO)
    @patch("detector.line_detection.cv2.HoughLinesP", return_value=None)
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_no_hough_lines_no_detection(self, *_):
        """No Hough lines found → not detected."""
        result = detect_horizontal_line(_FRAME)
        self.assertFalse(result["has_horizontal_line"])

    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_full_width_center_line_detected(self, *_):
        """Full-width Hough line at frame centre, FFT confirmed → detected (true positive)."""
        result = detect_horizontal_line(_FRAME)
        self.assertTrue(
            result["has_horizontal_line"],
            msg=f"Expected detection; gate={result.get('debug_line_info', {}).get('quality_gate')}",
        )

    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y, x1=285, x2=315))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_short_fragment_no_detection(self, *_):
        """30px segment < hough_min_length (15% × 600 = 90px) → rejected before candidate."""
        result = detect_horizontal_line(_FRAME)
        self.assertFalse(result["has_horizontal_line"])

    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=[[[0, _H_CENTER_ROI_Y, _W - 1, _H_CENTER_ROI_Y + 35]]])
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_diagonal_edge_no_detection(self, *_):
        """Slope 35/599 ≈ 0.058 > max_slope=0.05 → rejected by is_horizontal check."""
        result = detect_horizontal_line(_FRAME)
        self.assertFalse(result["has_horizontal_line"])

    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(50))  # roi_y=50 → frame_y=148, 48px from centre (tolerance=4px)
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_line_far_from_center_no_detection(self, *_):
        """Hough line far off-centre (48px from centre, tolerance=4px) → not detected."""
        result = detect_horizontal_line(_FRAME)
        self.assertFalse(result["has_horizontal_line"])

    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_NO)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_fft_bypass_removed(self, *_):
        """Fix 1: full-width high-coverage Hough line rejected when FFT=False.

        Before the fix the 'high_coverage' flag (line_length >= 40% of width) bypassed
        FFT confirmation entirely. After the fix, fft_confirmed is required on ALL paths.
        """
        result = detect_horizontal_line(_FRAME)
        gate = result.get("debug_line_info", {}).get("quality_gate", {})
        self.assertTrue(
            gate.get("high_coverage"),
            "test precondition: full-width line must trigger high_coverage",
        )
        self.assertFalse(
            gate.get("fft_confirmed"),
            "test precondition: FFT stub must return False",
        )
        self.assertFalse(
            result["has_horizontal_line"],
            "High-coverage Hough line must be rejected when FFT=False (Fix 1)",
        )

    @patch("detector.line_detection._verify_line_with_projection_profile",
           return_value=_PROFILE_FAIL)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_fft_min_dominance_configurable(self, *_):
        """Fix 2: fft_min_dominance is forwarded to _verify_line_with_fft.

        dominance=0.15 passes the default threshold (0.10) but fails a strict one (0.90).
        """
        dom = 0.15

        def _mock_fft(roi, min_dominance=0.10):
            return {"confirmed": dom >= min_dominance, "confidence": dom, "dominance": dom}

        with patch("detector.line_detection._verify_line_with_fft", side_effect=_mock_fft):
            result_default = detect_horizontal_line(_FRAME, fft_min_dominance=0.10)
            result_strict = detect_horizontal_line(_FRAME, fft_min_dominance=0.90)

        self.assertTrue(
            result_default["has_horizontal_line"],
            "dominance=0.15 should pass fft_min_dominance=0.10",
        )
        self.assertFalse(
            result_strict["has_horizontal_line"],
            "dominance=0.15 should fail fft_min_dominance=0.90 (Fix 2)",
        )


# ---------------------------------------------------------------------------
# Profile gate tests (horizontal)
# ---------------------------------------------------------------------------

class ProfileGateTests(unittest.TestCase):

    @patch("detector.line_detection._verify_line_with_projection_profile",
           return_value=_PROFILE_FAIL)
    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_profile_gate_off_by_default_preserves_behaviour(self, *_):
        """Gate off by default: low-profile FFT-confirmed line is still detected.

        Even when the profile verifier reports failure, enable_profile_gate=False
        (the default) means detection_confirmed is computed exactly as before.
        The advisory key profile_confirmed is recorded in quality_gate but does
        not alter the outcome.
        """
        result = detect_horizontal_line(_FRAME)
        self.assertTrue(
            result["has_horizontal_line"],
            msg="Profile gate is OFF by default — line must be detected regardless of profile",
        )
        gate = result.get("debug_line_info", {}).get("quality_gate", {})
        self.assertIn("profile_confirmed", gate, "Advisory key must always be recorded")
        self.assertFalse(gate["profile_confirmed"], "Advisory must reflect the mocked failure")

    @patch("detector.line_detection._verify_line_with_projection_profile",
           return_value=_PROFILE_FAIL)
    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_profile_gate_rejects_horizon_when_enabled(self, *_):
        """Fix for IMPROVEMENT_PLAN 'Remaining limitation': horizon case.

        A horizon passes FFT (narrow ROI genuinely has horizontal energy) but fails
        the spatial profile (gradient is spread across many rows, not localised).
        With enable_profile_gate=True the detection must be rejected.
        """
        result = detect_horizontal_line(_FRAME, enable_profile_gate=True)
        self.assertFalse(
            result["has_horizontal_line"],
            msg="FFT-confirmed but profile-failed line must be rejected when gate is enabled",
        )

    @patch("detector.line_detection._verify_line_with_projection_profile",
           return_value=_PROFILE_OK)
    @patch("detector.line_detection._verify_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_hline(_H_CENTER_ROI_Y))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_H, _W))
    def test_profile_gate_confirms_true_seam_when_enabled(self, *_):
        """True seam passes both FFT and spatial profile → detected with gate enabled."""
        result = detect_horizontal_line(_FRAME, enable_profile_gate=True)
        self.assertTrue(
            result["has_horizontal_line"],
            msg="FFT-confirmed + profile-confirmed line must be detected with gate enabled",
        )


# ---------------------------------------------------------------------------
# Vertical detection tests
# ---------------------------------------------------------------------------

class VerticalDetectionTests(unittest.TestCase):

    @patch("detector.line_detection._verify_vertical_line_with_fft", return_value=_FFT_NO)
    @patch("detector.line_detection.cv2.HoughLinesP", return_value=None)
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_VH, _VW))
    def test_no_hough_lines_no_detection(self, *_):
        """No Hough lines found → not detected (vertical)."""
        result = detect_vertical_line(_VFRAME)
        self.assertFalse(result["has_vertical_line"])

    @patch("detector.line_detection._verify_vertical_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_vline(_V_CENTER_ROI_X))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_VH, _VW))
    def test_full_height_center_line_detected(self, *_):
        """Full-height vertical Hough line at frame centre, FFT confirmed → detected."""
        result = detect_vertical_line(_VFRAME)
        self.assertTrue(
            result["has_vertical_line"],
            msg=f"Expected detection; gate={result.get('debug_line_info', {}).get('quality_gate')}",
        )

    @patch("detector.line_detection._verify_vertical_line_with_fft", return_value=_FFT_NO)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_vline(_V_CENTER_ROI_X))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_VH, _VW))
    def test_fft_bypass_removed_vertical(self, *_):
        """Fix 1 on vertical path: high-coverage line rejected when FFT=False."""
        result = detect_vertical_line(_VFRAME)
        gate = result.get("debug_line_info", {}).get("quality_gate", {})
        self.assertTrue(
            gate.get("high_coverage"),
            "test precondition: full-height line must trigger high_coverage",
        )
        self.assertFalse(
            result["has_vertical_line"],
            "High-coverage vertical Hough line must be rejected when FFT=False (Fix 1)",
        )

    @patch("detector.line_detection._verify_line_with_projection_profile",
           return_value=_PROFILE_FAIL)
    @patch("detector.line_detection._verify_vertical_line_with_fft", return_value=_FFT_OK)
    @patch("detector.line_detection.cv2.HoughLinesP",
           return_value=_vline(_V_CENTER_ROI_X))
    @patch("detector.line_detection.prepare_frame_for_line_detection",
           return_value=_FakeImage(_VH, _VW))
    def test_profile_gate_rejects_horizon_when_enabled_vertical(self, *_):
        """Mirror of horizon-rejection test on the vertical path.

        FFT confirmed but profile failed → rejected when enable_profile_gate=True.
        """
        result = detect_vertical_line(_VFRAME, enable_profile_gate=True)
        self.assertFalse(
            result["has_vertical_line"],
            msg="FFT-confirmed but profile-failed vertical line must be rejected when gate is on",
        )


if __name__ == "__main__":
    unittest.main()
