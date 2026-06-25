"""Tests for the unified EAC/Cubic projection scoring in projection_logic.py."""

from __future__ import annotations

import math

import detector.projection_logic as _projection_logic


def _np_is_functional(candidate) -> bool:
    try:
        return abs(float(candidate.sin(math.pi / 2)) - 1.0) < 1e-9
    except Exception:
        return False


if not _np_is_functional(_projection_logic.np):
    # Earlier-collected test modules (e.g. test_detection_retry.py) install an
    # incomplete fake `numpy` into sys.modules before this module is first
    # imported, leaving detector.projection_logic's `np` permanently bound to
    # that fake for the rest of the pytest session. The fake defines a
    # catch-all `__getattr__` so `hasattr(np, "sin")` is True even though
    # calling it returns None; rebind to the genuine package so
    # angle-correction math is real.
    import importlib
    import sys

    sys.modules.pop("numpy", None)
    _projection_logic.np = importlib.import_module("numpy")

from detector.projection_logic import EAC_LAYOUT, evaluate_cubemap, evaluate_eac

PI = math.pi


def _region(angle: float, *, concentration: float = 0.8, active_ratio: float = 0.3,
            magnitude: float = 5.0, valid: bool = True) -> dict:
    return {
        "valid": valid,
        "concentration": concentration,
        "active_ratio": active_ratio,
        "magnitude": magnitude,
        "angle": angle,
    }


def test_zero_flow_is_neutral_between_eac_and_cubic():
    region_info = {key: _region(0.0) for key in EAC_LAYOUT}

    score_eac = evaluate_eac(region_info)["score"]
    score_cubic = evaluate_cubemap(region_info)["score"]

    for score in (score_eac, score_cubic):
        assert score is None or score <= 0.05


def test_synthetic_eac_geometry_favours_eac():
    region_info = {
        (0, 0): _region(-PI / 2),  # LEFT
        (0, 1): _region(PI),       # FRONT
        (0, 2): _region(PI / 2),   # RIGHT
        (1, 0): _region(PI / 2),   # BOTTOM
        (1, 1): _region(-PI / 2),  # BACK
        (1, 2): _region(-PI / 2),  # TOP
    }

    score_eac = evaluate_eac(region_info)["score"]
    score_cubic = evaluate_cubemap(region_info)["score"]

    assert score_eac is not None and score_cubic is not None
    assert score_eac - score_cubic >= 0.3


def test_synthetic_cubic_geometry_favours_cubic():
    region_info = {
        (0, 0): _region(PI / 2),   # RIGHT
        (0, 1): _region(-PI / 2),  # LEFT
        (0, 2): _region(0.0),      # TOP
        (1, 0): _region(0.0),      # BOTTOM
        (1, 1): _region(PI),       # FRONT
        (1, 2): _region(0.0),      # BACK
    }

    score_eac = evaluate_eac(region_info)["score"]
    score_cubic = evaluate_cubemap(region_info)["score"]

    assert score_eac is not None and score_cubic is not None
    assert score_cubic - score_eac >= 0.3
