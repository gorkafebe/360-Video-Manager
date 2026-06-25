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

# Expected angular difference (radians) between each pair of cube faces
# derived from 3D cube geometry: adjacent faces = π/2, opposite = π.
FACE_PAIR_EXPECTED_DIFF: Dict[Tuple[str, str], float] = {
    ("FRONT", "LEFT"): _PI / 2,
    ("FRONT", "RIGHT"): _PI / 2,
    ("FRONT", "TOP"): _PI / 2,
    ("FRONT", "BOTTOM"): _PI / 2,
    ("FRONT", "BACK"): _PI,
    ("LEFT", "RIGHT"): _PI,
    ("LEFT", "TOP"): _PI / 2,
    ("LEFT", "BOTTOM"): _PI / 2,
    ("LEFT", "BACK"): _PI / 2,
    ("RIGHT", "TOP"): _PI / 2,
    ("RIGHT", "BOTTOM"): _PI / 2,
    ("RIGHT", "BACK"): _PI / 2,
    ("TOP", "BOTTOM"): _PI,
    ("TOP", "BACK"): _PI / 2,
    ("BOTTOM", "BACK"): _PI / 2,
}


def _angle_diff_signed(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _angle_diff_magnitude(a: float, b: float) -> float:
    return abs(_angle_diff_signed(a, b))


def _is_cross_row_pair(face_a: str, face_b: str, layout: Dict[tuple, tuple]) -> bool:
    """Return True if face_a and face_b are in different rows of layout."""
    row_map = {name: key[0] for key, (name, _) in layout.items()}
    row_a = row_map.get(face_a)
    row_b = row_map.get(face_b)
    if row_a is None or row_b is None:
        return False
    return row_a != row_b


def _score_layout_unified(
    face_angles: Dict[str, float],
    tolerance_rad: float,
    layout: Dict[tuple, tuple],
) -> Tuple[List[float], List[float], List[str]]:
    """Score a layout by testing pairwise angular differences against
    3D cube-geometry expectations. Both EAC and Cubic call this same
    function; only the layout dict (and which corrections were applied
    upstream) differs. Cross-row pairs carry weight 2.0, same-row 1.0."""
    scores: List[float] = []
    weights: List[float] = []
    details: List[str] = []

    available_faces = set(face_angles.keys())

    for (fa, fb), expected in FACE_PAIR_EXPECTED_DIFF.items():
        if fa not in available_faces or fb not in available_faces:
            continue

        observed = _angle_diff_magnitude(face_angles[fa], face_angles[fb])
        error = abs(observed - expected)
        s = max(0.0, 1.0 - error / tolerance_rad)

        cross = _is_cross_row_pair(fa, fb, layout)
        w = 2.0 if cross else 1.0
        scores.append(s)
        weights.append(w)
        prefix = "cross" if cross else "same"
        details.append(
            f"{prefix}:{fa[0]}/{fb[0]}"
            f"={np.degrees(observed):.0f}°"
            f"(exp={np.degrees(expected):.0f}°)"
            f"/{s:.2f}w{w:.0f}"
        )

    return scores, weights, details


def _raw_signal_is_degenerate(raw_angles: List[float], tolerance_rad: float) -> bool:
    """Return True when raw region angles carry no discriminating signal
    (every pairwise difference is smaller than the scoring tolerance)."""
    if len(raw_angles) < 2:
        return False
    for i in range(len(raw_angles)):
        for j in range(i + 1, len(raw_angles)):
            if _angle_diff_magnitude(raw_angles[i], raw_angles[j]) >= tolerance_rad:
                return False
    return True


def _score_layout(
    layout: Dict[tuple, tuple],
    region_info: Dict[tuple, Dict[str, Any]],
    min_concentration: float,
    min_active_ratio: float,
    tolerancia_45_deg: float,
) -> Tuple[Optional[float], List[str]]:
    face_angles: Dict[str, float] = {}
    raw_angles: List[float] = []
    for key, (face_name, correction) in layout.items():
        info = region_info.get(key)
        if not is_region_valid(info, min_concentration=min_concentration, min_active_ratio=min_active_ratio):
            continue
        raw = float(info["angle"])
        raw_angles.append(raw)
        corrected = float(np.arctan2(np.sin(raw + correction), np.cos(raw + correction)))
        face_angles[face_name] = corrected

    tolerance_rad = float(np.radians(tolerancia_45_deg))

    if _raw_signal_is_degenerate(raw_angles, tolerance_rad):
        return None, ["insufficient:degenerate_raw_signal"]

    scores, weights, details = _score_layout_unified(face_angles, tolerance_rad, layout)

    if not scores:
        return None, details

    has_cross_row = any(d.startswith("cross:") for d in details)
    if not has_cross_row:
        return None, details + ["insufficient:no_cross_row_pair"]

    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return float(weighted_score), details


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
