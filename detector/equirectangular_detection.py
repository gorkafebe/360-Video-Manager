"""Positive equirectangular evidence module.

Decides whether a frame set *positively* looks like mono equirectangular
footage, so that "no seam found" is not mistaken for "equirectangular"
(see ``_resolve_no_seam_classification`` in :mod:`detector.pipeline`). It looks
for three geometric signatures, each a property of the projection rather than
of the content:

A. **The 360° join.** Longitude wraps: the frame's last column and its first
   column are neighbours on the sphere, so they must be about as alike as any
   two adjacent interior columns. Measured on exactly those two columns
   (correlation of their brightness profiles, and their difference relative to
   the median adjacent-column difference elsewhere in the frame). A packed cube
   atlas puts two unrelated face edges there.
B. **Polar stretching.** Each equirectangular row is oversampled horizontally by
   1/cos(latitude), so near the top and bottom rows horizontal detail all but
   vanishes while vertical detail remains. Cube-atlas faces are not stretched.
C. **Mono, not stereo.** (1) The middle rows of a mono frame are the equator and
   show no stretching; in a top-bottom stereo frame they are the poles of both
   eyes and do. (2) The two eyes of a stereo frame are near-copies, so a
   side-by-side or top-bottom frame matches its own other half; the two
   halves of a mono frame are different views.

A frame passes only if A, B and C all pass. A video is strong evidence only with
enough usable frames and a large enough fraction of them passing. All thresholds
are parameters of :func:`aggregate_equirectangular_evidence` (exposed as
``VPD_EQUI_*`` in ``config/settings.py``); :func:`compute_frame_equirectangular_evidence`
only measures. The defaults were calibrated on real footage with
leave-one-source-out validation.

Signal A compares the actual join (the last and first columns), not border strips
that may lie tens of degrees of longitude apart, and no aspect-ratio prior is used,
so 16:9 equirectangular footage is not penalised. Every computation here is
``nan``-free by construction.
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision thresholds (defaults). Mirrored by VPD_EQUI_* in config/settings.py.
# Calibrated on 480 real frames (8 sources x 6 cases: original, repack, EAC,
# cubic, top-bottom and side-by-side stereo) with leave-one-source-out
# validation.
# Measured ranges are noted per threshold (per frame, all 8 sources).
# ---------------------------------------------------------------------------

# Signal A: correlation of the last and first column's brightness profiles.
# Equirectangular 0.858..1.000; EAC/cubic at most 0.825.
DEFAULT_JUNCTION_MIN_CORR = 0.84
# Signal A: join difference / median interior adjacent-column difference.
# Equirectangular 0.61..5.43; EAC/cubic frames that come close on the correlation
# (>= 0.74) are >= 22.9. A second guard behind the correlation, not a primary cut.
DEFAULT_JUNCTION_MAX_RATIO = 11.0
# Signal A: minimum std (grey levels) of each edge column; below it the frame
# carries no measurable join and is unusable rather than scored. Real frames: >= 4.5.
DEFAULT_JUNCTION_MIN_TEXTURE = 1.0
# Signal B: maximum horizontal/vertical gradient ratio in the top and bottom bands
# (the larger of the two). Equirectangular 0.083..0.227; EAC >= 0.538, cubic >= 1.017.
DEFAULT_POLAR_MAX_ANISOTROPY = 0.41
# Signals B and C: minimum mean gradient (grey levels) for a band to count at all;
# a constant (e.g. black) band fails instead of passing trivially. Faintest real
# equirectangular pole band 0.026; a black EAC band measured 0.000.
DEFAULT_BAND_MIN_TEXTURE = 0.01
# Signal C(1): minimum horizontal/vertical gradient ratio in the equator band.
# Equirectangular >= 0.382; top-bottom stereo (middle rows = poles) <= 0.162.
DEFAULT_EQUATOR_MIN_ANISOTROPY = 0.27
# Signal C(2): maximum similarity between the frame's two halves (either axis).
# Equirectangular <= 0.331; zero-disparity stereo 1.000 (real disparity lowers it).
DEFAULT_MAX_HALF_SIMILARITY = 0.66
# Video level: minimum fraction of usable frames passing A, B and C.
# Equirectangular cases 0.9..1.0; every other case 0.0.
DEFAULT_MIN_PASS_RATIO = 0.5
# Video level: minimum number of usable frames before any verdict.
DEFAULT_MIN_USABLE_FRAMES = 2

# ---------------------------------------------------------------------------
# Measurement constants (geometry, not tuning).
# ---------------------------------------------------------------------------

_JUNCTION_ROW_TOLERANCE = 2        # ± rows searched when aligning the two edge columns
_INTERIOR_SAMPLE_COLUMNS = 256     # adjacent-column pairs sampled for the baseline
_INTERIOR_DIFF_FLOOR = 0.25        # grey levels; keeps the ratio finite on flat frames
_ANISOTROPY_EPS = 0.01             # grey levels; numerical guard only
_POLAR_TOP_BAND = (0.00, 0.06)     # fractions of frame height
_POLAR_BOTTOM_BAND = (0.94, 1.00)
_EQUATOR_BAND = (0.47, 0.53)
_HALF_WORK_WIDTH = 256             # half-similarity is measured on a coarse copy
_HALF_MIN_RESIDUAL_STD = 0.5       # grey levels; below it a half carries no structure


# ---------------------------------------------------------------------------
# Per-frame measurements
# ---------------------------------------------------------------------------

def compute_aspect_ratio_score(
    frame: np.ndarray,
    target_ratio: float = 2.0,
    tolerance: float = 0.20,
) -> float:
    """INFORMATIONAL ONLY: closeness of width/height to 2:1, in [0.05, 1.0].

    Recorded in the per-frame result for debugging. It takes no part in any
    decision: real equirectangular footage is often 16:9, real EAC is sometimes
    2:1, and top-bottom stereo is not reliably square, so the ratio is neither a
    positive cue nor a stereo guard.
    """
    height, width = frame.shape[:2]
    if height == 0:
        return 0.05
    deviation = abs(width / float(height) - target_ratio)
    if deviation >= tolerance:
        far_penalty = max(0.0, 1.0 - (deviation - tolerance) / (target_ratio * 0.5))
        return max(0.05, float(far_penalty * 0.20))
    return float(np.clip(0.5 * (1.0 + np.cos(np.pi * deviation / tolerance)), 0.05, 1.0))


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 (never nan) when either input is constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _shifted_pairs(a: np.ndarray, b: np.ndarray, tol: int):
    """Yield (shift, a_slice, b_slice) for vertical shifts in [-tol, tol] along axis 0."""
    h = a.shape[0]
    for s in range(-tol, tol + 1):
        if s >= 0:
            yield s, a[s:], b[: h - s]
        else:
            yield s, a[: h + s], b[-s:]


def compute_junction_continuity(gray: np.ndarray) -> Dict[str, Any]:
    """Signal A: continuity across the 360° join (last column -> first column)."""
    h, w = gray.shape
    last, first = gray[:, w - 1], gray[:, 0]
    best_corr, best_shift, join_diff = -1.0, 0, float("inf")
    for s, a, b in _shifted_pairs(last, first, _JUNCTION_ROW_TOLERANCE):
        c = _pearson(a, b)
        if c > best_corr:
            best_corr, best_shift = c, s
        join_diff = min(join_diff, float(np.mean(np.abs(a - b))))

    xs = np.unique(np.linspace(1, w - 3, min(_INTERIOR_SAMPLE_COLUMNS, max(1, w - 3))).astype(int))
    left, right = gray[:, xs], gray[:, xs + 1]
    per_col = np.full(xs.shape[0], np.inf)
    for _s, a, b in _shifted_pairs(left, right, _JUNCTION_ROW_TOLERANCE):
        per_col = np.minimum(per_col, np.mean(np.abs(a - b), axis=0))
    interior = float(np.median(per_col)) if per_col.size else 0.0

    return {
        "junction_corr": float(best_corr),
        "junction_ratio": float(join_diff / max(interior, _INTERIOR_DIFF_FLOOR)),
        "junction_texture": float(min(last.std(), first.std())),
        "best_shift": int(best_shift),
    }


def compute_band_anisotropy(gray: np.ndarray, band: tuple) -> Dict[str, float]:
    """Mean |d/dx| and |d/dy| over a band of rows, and their ratio (B and C(1))."""
    h = gray.shape[0]
    r0 = int(band[0] * h)
    r1 = max(int(band[1] * h), r0 + 1)
    rows = gray[r0:r1]
    gx = float(np.mean(np.abs(np.diff(rows, axis=1)))) if rows.shape[1] > 1 else 0.0
    # Vertical differences for the band's rows, staying inside the frame.
    lo, hi = max(r0 - 1, 0), min(r1 + 1, h)
    gy = float(np.mean(np.abs(np.diff(gray[lo:hi], axis=0)))) if hi - lo > 1 else 0.0
    return {"gx": gx, "gy": gy, "texture": gx + gy, "anisotropy": gx / (gy + _ANISOTROPY_EPS)}


def _residual_similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Correlation of two equally sized halves after removing each half's row means."""
    ra = a - a.mean(axis=1, keepdims=True)
    rb = b - b.mean(axis=1, keepdims=True)
    texture = float(min(ra.std(), rb.std()))
    if texture < _HALF_MIN_RESIDUAL_STD:
        return {"similarity": 0.0, "texture": texture}
    return {"similarity": _pearson(ra.ravel(), rb.ravel()), "texture": texture}


def compute_half_similarity(gray: np.ndarray) -> Dict[str, float]:
    """Signal C(2): how well the frame matches its own other half, on each axis.

    Removing each half's row means first discards the generic sky-above-ground
    profile that any two parts of an equirectangular frame share, so only real
    structure is compared. Done on a coarse copy, which also makes the test
    tolerant of the small disparity between real stereo eyes.
    """
    h, w = gray.shape
    work_w = min(_HALF_WORK_WIDTH, w)
    work_h = max(8, int(round(h * work_w / float(w))))
    small = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)
    hw, hh = work_w // 2, work_h // 2
    lr = _residual_similarity(small[:, :hw], small[:, hw: 2 * hw])
    tb = _residual_similarity(small[:hh], small[hh: 2 * hh])
    return {
        "half_similarity_lr": lr["similarity"],
        "half_similarity_tb": tb["similarity"],
        "half_texture": float(min(lr["texture"], tb["texture"])),
    }


def compute_frame_equirectangular_evidence(frame: np.ndarray) -> Dict[str, Any]:
    """Measure all three signals on one frame. Applies no thresholds.

    Keys ``usable`` (the measurement succeeded) and ``error`` keep their meaning;
    ``score`` is the fraction of the three signals that pass under the module's
    default thresholds, for telemetry. The pass/fail decision used by the
    pipeline is made by :func:`aggregate_equirectangular_evidence`, with the
    configured thresholds, from the raw measurements returned here.
    ``wrap_score`` is the join correlation clipped to [0, 1]; ``best_shift`` is
    the row offset that aligned the two edge columns best.
    """
    base: Dict[str, Any] = {"score": 0.0, "usable": False, "error": None,
                            "aspect_ratio_score": 0.0, "wrap_score": 0.0, "best_shift": 0}
    try:
        if frame is None or frame.size == 0:
            base["error"] = "empty_frame"
            return base
        height, width = frame.shape[:2]
        if height < 16 or width < 32:
            base["error"] = "frame_too_small"
            return base
        gray = _to_gray(frame)
        top = compute_band_anisotropy(gray, _POLAR_TOP_BAND)
        bottom = compute_band_anisotropy(gray, _POLAR_BOTTOM_BAND)
        equator = compute_band_anisotropy(gray, _EQUATOR_BAND)
        junction = compute_junction_continuity(gray)
        rec: Dict[str, Any] = {
            "usable": True,
            "error": None,
            "aspect_ratio_score": compute_aspect_ratio_score(frame),  # informational only
            "wrap_score": float(np.clip(junction["junction_corr"], 0.0, 1.0)),
            **junction,
            "polar_top_anisotropy": top["anisotropy"],
            "polar_top_texture": top["texture"],
            "polar_bottom_anisotropy": bottom["anisotropy"],
            "polar_bottom_texture": bottom["texture"],
            "equator_anisotropy": equator["anisotropy"],
            "equator_texture": equator["texture"],
            **compute_half_similarity(gray),
        }
        rec["score"] = evaluate_frame(rec)["score"]
        return rec
    except Exception as exc:  # pragma: no cover - defensive: a frame must never raise
        logger.debug("equi evidence error on frame: %s", exc)
        base["error"] = str(exc)
        return base


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

_MEASUREMENT_KEYS = (
    "junction_corr", "junction_ratio", "junction_texture",
    "polar_top_anisotropy", "polar_top_texture", "polar_bottom_anisotropy", "polar_bottom_texture",
    "equator_anisotropy", "equator_texture", "half_similarity_lr", "half_similarity_tb",
)


def evaluate_frame(
    rec: Dict[str, Any],
    junction_min_corr: float = DEFAULT_JUNCTION_MIN_CORR,
    junction_max_ratio: float = DEFAULT_JUNCTION_MAX_RATIO,
    junction_min_texture: float = DEFAULT_JUNCTION_MIN_TEXTURE,
    polar_max_anisotropy: float = DEFAULT_POLAR_MAX_ANISOTROPY,
    band_min_texture: float = DEFAULT_BAND_MIN_TEXTURE,
    equator_min_anisotropy: float = DEFAULT_EQUATOR_MIN_ANISOTROPY,
    max_half_similarity: float = DEFAULT_MAX_HALF_SIMILARITY,
) -> Dict[str, Any]:
    """Apply the per-frame thresholds to one frame's measurements.

    Returns ``usable`` (measured, with a textured join), the three signal
    verdicts, ``passes`` (all three) and ``score`` (fraction passing).
    """
    if not rec.get("usable") or any(k not in rec for k in _MEASUREMENT_KEYS):
        return {"usable": False, "signal_a": False, "signal_b": False, "signal_c": False,
                "passes": False, "score": 0.0}
    usable = rec["junction_texture"] >= junction_min_texture
    signal_a = usable and rec["junction_corr"] >= junction_min_corr and rec["junction_ratio"] <= junction_max_ratio
    signal_b = all(
        rec[f"polar_{band}_texture"] >= band_min_texture
        and rec[f"polar_{band}_anisotropy"] <= polar_max_anisotropy
        for band in ("top", "bottom")
    )
    signal_c = (
        rec["equator_texture"] >= band_min_texture
        and rec["equator_anisotropy"] >= equator_min_anisotropy
        and max(rec["half_similarity_lr"], rec["half_similarity_tb"]) <= max_half_similarity
    )
    signals = (bool(signal_a), bool(signal_b), bool(signal_c))
    return {"usable": bool(usable), "signal_a": signals[0], "signal_b": signals[1],
            "signal_c": signals[2], "passes": all(signals), "score": sum(signals) / 3.0}


def aggregate_equirectangular_evidence(
    frame_results: List[Dict[str, Any]],
    junction_min_corr: float = DEFAULT_JUNCTION_MIN_CORR,
    junction_max_ratio: float = DEFAULT_JUNCTION_MAX_RATIO,
    junction_min_texture: float = DEFAULT_JUNCTION_MIN_TEXTURE,
    polar_max_anisotropy: float = DEFAULT_POLAR_MAX_ANISOTROPY,
    band_min_texture: float = DEFAULT_BAND_MIN_TEXTURE,
    equator_min_anisotropy: float = DEFAULT_EQUATOR_MIN_ANISOTROPY,
    max_half_similarity: float = DEFAULT_MAX_HALF_SIMILARITY,
    min_pass_ratio: float = DEFAULT_MIN_PASS_RATIO,
    min_usable_frames: int = DEFAULT_MIN_USABLE_FRAMES,
) -> Dict[str, Any]:
    """Aggregate per-frame measurements into a video-level verdict.

    A frame passes only if signals A, B and C all pass. The video is strong
    evidence only with at least ``min_usable_frames`` usable frames of which at
    least ``min_pass_ratio`` pass.

    Returns a dict with:
        usable_frames, strong_frames (frames passing), strong_ratio,
        mean_score, median_score (of per-frame signal fractions),
        final_score (= strong_ratio), is_strong_evidence, confidence, reason,
        plus per-signal pass counts for diagnosis.
    """
    thresholds = dict(
        junction_min_corr=junction_min_corr, junction_max_ratio=junction_max_ratio,
        junction_min_texture=junction_min_texture, polar_max_anisotropy=polar_max_anisotropy,
        band_min_texture=band_min_texture, equator_min_anisotropy=equator_min_anisotropy,
        max_half_similarity=max_half_similarity,
    )
    verdicts = [evaluate_frame(r, **thresholds) for r in frame_results]
    usable = [v for v in verdicts if v["usable"]]
    n = len(usable)
    counts = {
        "signal_a_passes": sum(v["signal_a"] for v in usable),
        "signal_b_passes": sum(v["signal_b"] for v in usable),
        "signal_c_passes": sum(v["signal_c"] for v in usable),
        "unusable_frames": len(verdicts) - n,
    }
    if n == 0:
        return {
            "usable_frames": 0, "strong_frames": 0, "strong_ratio": 0.0,
            "mean_score": 0.0, "median_score": 0.0, "final_score": 0.0,
            "is_strong_evidence": False, "confidence": 0.0, "reason": "no_usable_frames",
            **counts,
        }

    scores = np.array([v["score"] for v in usable], dtype=np.float64)
    passing = sum(v["passes"] for v in usable)
    ratio = passing / float(n)
    is_strong = bool(n >= int(min_usable_frames) and ratio >= float(min_pass_ratio))
    detail = (f"pass={passing}/{n},A={counts['signal_a_passes']},B={counts['signal_b_passes']},"
              f"C={counts['signal_c_passes']},unusable={counts['unusable_frames']}")
    return {
        "usable_frames": n,
        "strong_frames": int(passing),
        "strong_ratio": float(ratio),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "final_score": float(ratio),
        "is_strong_evidence": is_strong,
        "confidence": float(ratio) if is_strong else 0.0,
        "reason": (f"equirectangular_geometry_confirmed({detail})" if is_strong
                   else f"weak_equi_evidence({detail})"),
        **counts,
    }
