"""Tests for unknown-projection EAC fallback and adaptive wraplength helper."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from workflows.unified_pipeline import _stage_convert_to_equirectangular
from app.gui.gui_app import _THUMB_CARD


class UnknownProjectionFallbackTests(unittest.TestCase):
    """_stage_convert_to_equirectangular must treat 'unknown' as 'eac'."""

    def _run_stage(self, projection_type, confidence=0.9, *, converted_return="/out/conv.mp4"):
        """Call the stage with a mocked conversion back-end."""
        with patch(
            "detector.projection_conversion.convert_detected_projection_to_equirectangular",
            return_value=converted_return,
        ) as mock_conv:
            result = _stage_convert_to_equirectangular(
                video_path="/input/video.mp4",
                projection_type=projection_type,
                confidence=confidence,
                output_dir="/out",
                confidence_threshold=0.5,
            )
        return result, mock_conv

    def test_unknown_triggers_conversion_not_skip(self):
        """unknown detection must not be skipped; conversion must be attempted."""
        result, mock_conv = self._run_stage("unknown")
        mock_conv.assert_called_once()
        self.assertIsNotNone(result)

    def test_unknown_calls_conversion_with_eac(self):
        """When projection is unknown, conversion must be called with 'eac'."""
        _result, mock_conv = self._run_stage("unknown")
        _call_kwargs = mock_conv.call_args
        # projection_type is the second positional argument
        called_projection = (
            _call_kwargs.kwargs.get("projection_type")
            or _call_kwargs.args[1]
        )
        self.assertEqual(called_projection, "eac")

    def test_unknown_returns_converted_path(self):
        """unknown → eac fallback should return the converted file path."""
        result, _ = self._run_stage("unknown", converted_return="/out/converted.mp4")
        self.assertEqual(result, "/out/converted.mp4")

    def test_equirectangular_still_skips(self):
        """equirectangular projection should still be skipped (no conversion)."""
        result, mock_conv = self._run_stage("equirectangular")
        mock_conv.assert_not_called()
        self.assertIsNone(result)

    def test_stereo_equi_still_skips(self):
        """stereo_equi projection should still be skipped (no conversion)."""
        result, mock_conv = self._run_stage("stereo_equi")
        mock_conv.assert_not_called()
        self.assertIsNone(result)

    def test_eac_converts_directly(self):
        """eac projection (without unknown) must go through conversion normally."""
        result, mock_conv = self._run_stage("eac")
        mock_conv.assert_called_once()
        self.assertIsNotNone(result)

    def test_low_confidence_skips_even_for_unknown(self):
        """Confidence below threshold must suppress conversion even after fallback."""
        result, mock_conv = self._run_stage("unknown", confidence=0.3)
        mock_conv.assert_not_called()
        self.assertIsNone(result)


class AdaptiveWrapLengthTests(unittest.TestCase):
    """Pure helper logic for card-title wraplength calculation used in _on_root_resize."""

    @staticmethod
    def _calc_card_wrap(window_width: int) -> int:
        """Mirror the formula used in VR360ManagerApp._on_root_resize."""
        return max(150, window_width - 32 - _THUMB_CARD[0] - 68)

    def test_large_window_gives_reasonable_wrap(self):
        wrap = self._calc_card_wrap(1000)
        # 1000 - 32 - 120 - 68 = 780
        self.assertEqual(wrap, 780)

    def test_small_window_clamps_to_minimum(self):
        wrap = self._calc_card_wrap(100)
        self.assertEqual(wrap, 150)

    def test_exactly_at_minimum_boundary(self):
        # 370 - 32 - 120 - 68 = 150
        wrap = self._calc_card_wrap(370)
        self.assertEqual(wrap, 150)

    def test_wrap_increases_with_window_width(self):
        self.assertLess(
            self._calc_card_wrap(800),
            self._calc_card_wrap(1200),
        )


if __name__ == "__main__":
    unittest.main()
