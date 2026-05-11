from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .region_validation import is_region_valid


_PI = float(np.pi)
EAC_LAYOUT: Dict[tuple, tuple] = {
    (0, 0): ("LEFT", 0.0),
    (0, 1): ("FRONT", 0.0),
    (0, 2): ("RIGHT", 0.0),
    (1, 0): ("BOTTOM", -_PI / 2.0),
    (1, 1): ("BACK", +_PI / 2.0),
    (1, 2): ("TOP", +_PI / 2.0),
}
CUBEMAP_LAYOUT: Dict[tuple, tuple] = {
    (0, 0): ("RIGHT", 0.0),
    (0, 1): ("LEFT", 0.0),
    (0, 2): ("TOP", 0.0),
    (1, 0): ("BOTTOM", 0.0),
    (1, 1): ("FRONT", 0.0),
    (1, 2): ("BACK", 0.0),
}


def _angle_diff_signed(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))

    
def _angle_diff_magnitude(a: float, b: float) -> float:
    return abs(_angle_diff_signed(a, b))


def _score_layout_eac(face_angles: Dict[str, float], tolerance_rad: float) -> Tuple[List[float], List[str]]:
    """Score EAC by relative symmetry, avoiding fixed absolute angular targets."""
    scores: List[float] = []
    details: List[str] = []

    # LEFT and RIGHT should be symmetric around FRONT: diff_left ~= -diff_right
    if "LEFT" in face_angles and "RIGHT" in face_angles and "FRONT" in face_angles:
        front = face_angles["FRONT"]
        diff_l = _angle_diff_signed(face_angles["LEFT"], front)
        diff_r = _angle_diff_signed(face_angles["RIGHT"], front)
        error = abs(diff_l + diff_r)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"LR_sym={np.degrees(error):.0f}°/{s:.2f}")
    elif "LEFT" in face_angles and "RIGHT" in face_angles:
        # No FRONT available: use LEFT/RIGHT opposite-direction consistency.
        diff_lr = _angle_diff_signed(face_angles["LEFT"], face_angles["RIGHT"])
        error = abs(abs(diff_lr) - _PI)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"LR_nof={np.degrees(diff_lr):.0f}°/{s:.2f}")

    # BACK and FRONT are opposite cube faces, so |diff| should be near pi.
    if "BACK" in face_angles and "FRONT" in face_angles:
        diff = _angle_diff_signed(face_angles["BACK"], face_angles["FRONT"])
        mag_diff = _angle_diff_magnitude(diff, 0.0)
        error = abs(mag_diff - _PI)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"B=|{np.degrees(mag_diff):.0f}°|/{s:.2f}")

    return scores, details


def _score_layout_cubemap(face_angles: Dict[str, float], tolerance_rad: float) -> Tuple[List[float], List[str]]:
    """Score cubemap by perpendicular side faces relative to FRONT."""
    scores: List[float] = []
    details: List[str] = []
    half_pi = _PI / 2.0

    if "LEFT" in face_angles and "FRONT" in face_angles:
        diff = _angle_diff_signed(face_angles["LEFT"], face_angles["FRONT"])
        error = abs(abs(diff) - half_pi)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"L={np.degrees(diff):.0f}°/{s:.2f}")

    if "RIGHT" in face_angles and "FRONT" in face_angles:
        diff = _angle_diff_signed(face_angles["RIGHT"], face_angles["FRONT"])
        error = abs(abs(diff) - half_pi)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"R={np.degrees(diff):.0f}°/{s:.2f}")

    if "LEFT" in face_angles and "RIGHT" in face_angles and "FRONT" not in face_angles:
        diff_lr = _angle_diff_signed(face_angles["LEFT"], face_angles["RIGHT"])
        error = abs(abs(diff_lr) - _PI)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"LR_nof={np.degrees(diff_lr):.0f}°/{s:.2f}")

    if "BACK" in face_angles and "FRONT" in face_angles:
        diff = _angle_diff_signed(face_angles["BACK"], face_angles["FRONT"])
        mag_diff = _angle_diff_magnitude(diff, 0.0)
        error = abs(mag_diff - _PI)
        s = max(0.0, 1.0 - error / tolerance_rad)
        scores.append(s)
        details.append(f"B=|{np.degrees(mag_diff):.0f}°|/{s:.2f}")

    return scores, details


def _score_layout(
    layout: Dict[tuple, tuple],
    region_info: Dict[tuple, Dict[str, Any]],
    min_concentration: float,
    min_active_ratio: float,
    tolerancia_45_deg: float,
) -> Tuple[Optional[float], List[str]]:
    face_angles: Dict[str, float] = {}
    for key, (face_name, correction) in layout.items():
        info = region_info.get(key)
        if not is_region_valid(info, min_concentration=min_concentration, min_active_ratio=min_active_ratio):
            continue
        raw = float(info["angle"])
        corrected = float(np.arctan2(np.sin(raw + correction), np.cos(raw + correction)))
        face_angles[face_name] = corrected

    tolerance_rad = float(np.radians(tolerancia_45_deg))
    if layout is EAC_LAYOUT:
        scores, details = _score_layout_eac(face_angles, tolerance_rad)
    else:
        scores, details = _score_layout_cubemap(face_angles, tolerance_rad)

    if len(scores) < 1:
        return None, details

    return float(np.mean(scores)), details


def evaluate_eac(
    regions: Dict[tuple, Dict[str, Any]],
    min_concentration: float = 0.25,
    min_active_ratio: float = 0.06,
    tolerancia_45_deg: float = 20.0,
) -> Dict[str, Any]:
    score, details = _score_layout(
        EAC_LAYOUT,
        regions,
        min_concentration=min_concentration,
        min_active_ratio=min_active_ratio,
        tolerancia_45_deg=tolerancia_45_deg,
    )
    return {"score": score, "details": details}


def evaluate_cubemap(
    regions: Dict[tuple, Dict[str, Any]],
    min_concentration: float = 0.25,
    min_active_ratio: float = 0.06,
    tolerancia_45_deg: float = 20.0,
) -> Dict[str, Any]:
    score, details = _score_layout(
        CUBEMAP_LAYOUT,
        regions,
        min_concentration=min_concentration,
        min_active_ratio=min_active_ratio,
        tolerancia_45_deg=tolerancia_45_deg,
    )
    return {"score": score, "details": details}


def decide_projection(
    score_eac: Optional[float],
    score_cubemap: Optional[float],
    min_margin: float = 0.0,
) -> Optional[bool]:
    """Devuelve True=EAC, False=CUBIC, None=insuficiente."""
    if score_eac is None or score_cubemap is None:
        return None
    if abs(float(score_eac) - float(score_cubemap)) < float(min_margin):
        return None
    return bool(score_eac >= score_cubemap)
