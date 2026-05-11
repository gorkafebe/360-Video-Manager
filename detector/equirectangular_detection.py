"""Positive equirectangular evidence module.

Detects mono equirectangular projection by looking for positive wrap-around
boundary continuity and a 2:1 aspect-ratio prior.  Designed to work even when:

- centered seam detection is imperfect,
- some false positive horizontal/vertical lines have been detected,
- frames_with_line > 0 does not imply non-equirectangular content.

No new dependencies beyond OpenCV and NumPy (already present in the repo).
All public functions are safe to call; errors on individual frames are caught
and result in that frame being flagged as unusable rather than raising.
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def compute_aspect_ratio_score(
    frame: np.ndarray,
    target_ratio: float = 2.0,
    tolerance: float = 0.20,
) -> float:
    """Return a soft [0, 1] score for how close the frame's aspect ratio is to *target_ratio*.

    Score is 1.0 at exactly the target and decays as a raised-cosine bell
    within ±tolerance, reaching ~0 at the edges. Outside that range the score
    is clipped to a small floor (0.05) so the contribution never becomes
    completely zero from ratio alone.

    Args:
        frame: BGR or grayscale image.
        target_ratio: Expected width/height.  Equirectangular is usually 2.0.
        tolerance: Half-width of the high-score region.

    Returns:
        Float in [0.05, 1.0].
    """
    height, width = frame.shape[:2]
    if height == 0:
        return 0.05
    ratio = width / float(height)
    deviation = abs(ratio - target_ratio)
    if deviation >= tolerance:
        # Outside comfortable zone — linear decay to floor
        far_penalty = max(0.0, 1.0 - (deviation - tolerance) / (target_ratio * 0.5))
        return max(0.05, float(far_penalty * 0.20))
    # Inside tolerance: raised-cosine bell
    phase = deviation / tolerance  # 0→0, tolerance→1
    score = 0.5 * (1.0 + np.cos(np.pi * phase))
    return float(np.clip(score, 0.05, 1.0))


def extract_border_strips(
    frame: np.ndarray,
    strip_ratio: float = 0.04,
    min_strip_px: int = 12,
) -> tuple:
    """Return (left_strip, right_strip) extracted symmetrically from the frame.

    Strip width is max(strip_ratio * width, min_strip_px).

    Args:
        frame: BGR or grayscale image.
        strip_ratio: Fraction of frame width for each strip.
        min_strip_px: Minimum strip width in pixels.

    Returns:
        Tuple (left_strip, right_strip) as numpy arrays (same dtype as input).
    """
    height, width = frame.shape[:2]
    strip_w = max(min_strip_px, int(width * strip_ratio))
    strip_w = min(strip_w, width // 4)  # never more than 25 % of frame width
    left_strip = frame[:, :strip_w]
    right_strip = frame[:, width - strip_w:]
    return left_strip, right_strip


def compute_edge_map(image: np.ndarray) -> np.ndarray:
    """Return a binary-ish Canny edge map of *image* (grayscale or BGR).

    Uses Otsu thresholding to derive adaptive Canny thresholds, consistent
    with the approach in line_detection.py.

    Args:
        image: Any image array.

    Returns:
        uint8 edge map with values 0 / 255.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = float(max(50.0, otsu_val))
    low = high * 0.4
    return cv2.Canny(gray, low, high, apertureSize=3)


def _strip_to_gray(strip: np.ndarray) -> np.ndarray:
    """Convert a strip to uint8 grayscale if needed."""
    if len(strip.shape) == 3:
        return cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    return strip.copy()


def compare_strips_with_vertical_shift(
    left: np.ndarray,
    right: np.ndarray,
    max_shift_ratio: float = 0.03,
) -> Dict[str, Any]:
    """Compare left and right border strips searching over a vertical shift range.

    For each integer vertical shift in [-max_shift_px, +max_shift_px]:
        - Align the overlapping region.
        - Compute:
            hist_corr     : grayscale histogram correlation (high = good).
            hist_bhatt    : Bhattacharyya distance (low = good).
            nmad          : normalised mean absolute difference (low = good).
            edge_sim      : edge-map overlap similarity (high = good).
        - Combine into a single wrap_score in [0, 1].

    Returns the best shift and best combined metrics dict with keys:
        best_shift, hist_corr, hist_bhatt, nmad, edge_sim, wrap_score,
        tested_shifts.

    Args:
        left:  Left border strip (BGR or gray).
        right: Right border strip (BGR or gray).
        max_shift_ratio: Fractional vertical search range relative to strip height.

    Returns:
        Dict with the best-shift metrics and a combined ``wrap_score`` in [0, 1].
    """
    lg = _strip_to_gray(left)
    rg = _strip_to_gray(right)

    height = lg.shape[0]
    max_shift_px = max(1, int(height * max_shift_ratio))

    best_wrap_score = -1.0
    best_metrics: Dict[str, Any] = {
        "best_shift": 0,
        "hist_corr": 0.0,
        "hist_bhatt": 1.0,
        "nmad": 1.0,
        "edge_sim": 0.0,
        "wrap_score": 0.0,
        "tested_shifts": 0,
    }

    le = compute_edge_map(lg)
    re = compute_edge_map(rg)

    shifts_tested = 0
    for shift in range(-max_shift_px, max_shift_px + 1):
        # Compute overlapping slice for each strip under this shift
        if shift >= 0:
            l_slice = lg[shift:, :]
            r_slice = rg[: height - shift, :] if shift > 0 else rg
            le_slice = le[shift:, :]
            re_slice = re[: height - shift, :] if shift > 0 else re
        else:
            l_slice = lg[: height + shift, :]
            r_slice = rg[-shift:, :]
            le_slice = le[: height + shift, :]
            re_slice = re[-shift:, :]

        overlap_h = l_slice.shape[0]
        if overlap_h < 8:
            continue

        # Resize right to match left width (they should be same, but guard)
        if r_slice.shape[1] != l_slice.shape[1]:
            r_slice = cv2.resize(r_slice, (l_slice.shape[1], overlap_h))
        if re_slice.shape[1] != le_slice.shape[1]:
            re_slice = cv2.resize(re_slice, (le_slice.shape[1], le_slice.shape[0]))

        # Histogram metrics
        hist_l = cv2.calcHist([l_slice], [0], None, [64], [0, 256])
        hist_r = cv2.calcHist([r_slice], [0], None, [64], [0, 256])
        cv2.normalize(hist_l, hist_l, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_corr = float(cv2.compareHist(hist_l, hist_r, cv2.HISTCMP_CORREL))
        hist_bhatt = float(cv2.compareHist(hist_l, hist_r, cv2.HISTCMP_BHATTACHARYYA))

        # Normalised mean absolute difference on intensity
        l_f = l_slice.astype(np.float32)
        r_f = r_slice.astype(np.float32)
        nmad = float(np.mean(np.abs(l_f - r_f))) / 255.0

        # Edge-map similarity: intersection over union of non-zero pixels
        le_b = (le_slice > 0)
        re_b = (re_slice > 0)
        # Trim re_b/le_b to same overlap_h
        le_b_crop = le_b[:overlap_h, :]
        re_b_crop = re_b[:overlap_h, :]
        union = np.count_nonzero(le_b_crop | re_b_crop)
        inter = np.count_nonzero(le_b_crop & re_b_crop)
        edge_sim = float(inter / union) if union > 0 else 0.0

        # Combine into wrap_score
        # Weights: hist_corr 0.35, hist_bhatt (inverted) 0.25, nmad (inverted) 0.20, edge_sim 0.20
        score = (
            0.35 * float(np.clip(hist_corr, 0.0, 1.0))
            + 0.25 * float(np.clip(1.0 - hist_bhatt, 0.0, 1.0))
            + 0.20 * float(np.clip(1.0 - nmad * 4.0, 0.0, 1.0))  # nmad > 0.25 → 0
            + 0.20 * float(np.clip(edge_sim, 0.0, 1.0))
        )

        shifts_tested += 1
        if score > best_wrap_score:
            best_wrap_score = score
            best_metrics = {
                "best_shift": int(shift),
                "hist_corr": float(hist_corr),
                "hist_bhatt": float(hist_bhatt),
                "nmad": float(nmad),
                "edge_sim": float(edge_sim),
                "wrap_score": float(score),
                "tested_shifts": shifts_tested,
            }

    best_metrics["tested_shifts"] = shifts_tested
    return best_metrics


def compute_frame_equirectangular_evidence(
    frame: np.ndarray,
    strip_ratio: float = 0.04,
    min_strip_px: int = 12,
    max_shift_ratio: float = 0.03,
    target_aspect_ratio: float = 2.0,
    aspect_tolerance: float = 0.20,
    wrap_weight: float = 0.80,
    aspect_weight: float = 0.20,
) -> Dict[str, Any]:
    """Compute positive equirectangular evidence for a single frame.

    Returns a dict with:
        score           : combined evidence in [0, 1]
        aspect_ratio_score : [0, 1]
        wrap_score      : [0, 1]
        best_shift      : int, best vertical shift found
        hist_corr       : float
        hist_bhatt      : float
        nmad            : float
        edge_sim        : float
        usable          : bool  (False if computation failed or frame too small)
        error           : optional error message

    Args:
        frame: BGR or grayscale image.
        strip_ratio: Border strip width as fraction of frame width.
        min_strip_px: Minimum strip width in pixels.
        max_shift_ratio: Max vertical shift search as fraction of strip height.
        target_aspect_ratio: Expected w/h for equirectangular (2.0).
        aspect_tolerance: Half-width of high-score zone around target ratio.
        wrap_weight: Weight of wrap_score in combined score.
        aspect_weight: Weight of aspect_ratio_score in combined score.

    Returns:
        Dict with evidence metrics and a ``usable`` flag.
    """
    base: Dict[str, Any] = {
        "score": 0.0,
        "aspect_ratio_score": 0.0,
        "wrap_score": 0.0,
        "best_shift": 0,
        "hist_corr": 0.0,
        "hist_bhatt": 1.0,
        "nmad": 1.0,
        "edge_sim": 0.0,
        "usable": False,
        "error": None,
    }
    try:
        if frame is None or frame.size == 0:
            base["error"] = "empty_frame"
            return base
        height, width = frame.shape[:2]
        if height < 16 or width < 32:
            base["error"] = "frame_too_small"
            return base

        aspect_score = compute_aspect_ratio_score(
            frame, target_ratio=target_aspect_ratio, tolerance=aspect_tolerance
        )

        left_strip, right_strip = extract_border_strips(frame, strip_ratio=strip_ratio, min_strip_px=min_strip_px)
        if left_strip.shape[0] < 8 or right_strip.shape[0] < 8:
            base["error"] = "strips_too_small"
            return base

        shift_result = compare_strips_with_vertical_shift(left_strip, right_strip, max_shift_ratio=max_shift_ratio)
        wrap_sc = float(shift_result["wrap_score"])

        combined = float(wrap_weight) * wrap_sc + float(aspect_weight) * aspect_score
        combined = float(np.clip(combined, 0.0, 1.0))

        return {
            "score": combined,
            "aspect_ratio_score": aspect_score,
            "wrap_score": wrap_sc,
            "best_shift": shift_result["best_shift"],
            "hist_corr": shift_result["hist_corr"],
            "hist_bhatt": shift_result["hist_bhatt"],
            "nmad": shift_result["nmad"],
            "edge_sim": shift_result["edge_sim"],
            "usable": True,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        logger.debug("equi evidence error on frame: %s", exc)
        base["error"] = str(exc)
        return base


def aggregate_equirectangular_evidence(
    frame_results: List[Dict[str, Any]],
    strong_frame_threshold: float = 0.65,
    min_strong_ratio: float = 0.30,
) -> Dict[str, Any]:
    """Aggregate per-frame equirectangular evidence into a video-level summary.

    Only frames with ``usable == True`` are considered.  Uses both median and
    mean so that a single lucky frame cannot dominate.

    Args:
        frame_results: List of dicts returned by compute_frame_equirectangular_evidence.
        strong_frame_threshold: Score above which a frame is considered "strong evidence".
        min_strong_ratio: Minimum fraction of usable frames that must be strong for
            ``is_strong_evidence`` to be True.

    Returns:
        Dict with:
            usable_frames, strong_frames, strong_ratio,
            mean_score, median_score, final_score,
            is_strong_evidence, confidence, reason.
    """
    usable = [r for r in frame_results if r.get("usable")]
    n = len(usable)

    if n == 0:
        return {
            "usable_frames": 0,
            "strong_frames": 0,
            "strong_ratio": 0.0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "final_score": 0.0,
            "is_strong_evidence": False,
            "confidence": 0.0,
            "reason": "no_usable_frames",
        }

    scores = np.array([float(r["score"]) for r in usable], dtype=np.float64)
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    strong_count = int(np.sum(scores >= strong_frame_threshold))
    strong_ratio = strong_count / float(n)

    # Primary: median; blended slightly with mean for stability
    final_score = float(0.70 * median_score + 0.30 * mean_score)

    # Strong evidence requires both a high final score and enough strong frames
    is_strong_evidence = bool(final_score >= strong_frame_threshold and strong_ratio >= min_strong_ratio)

    # Confidence mirrors the final score when evidence is strong, moderate otherwise
    if is_strong_evidence:
        confidence = float(np.clip(final_score, 0.0, 1.0))
        reason = "positive_equirectangular_wraparound_evidence"
    else:
        confidence = float(np.clip(final_score * 0.75, 0.0, 1.0))
        reason = f"weak_equi_evidence(score={final_score:.3f},strong_ratio={strong_ratio:.2f})"

    return {
        "usable_frames": n,
        "strong_frames": strong_count,
        "strong_ratio": float(strong_ratio),
        "mean_score": mean_score,
        "median_score": median_score,
        "final_score": float(final_score),
        "is_strong_evidence": is_strong_evidence,
        "confidence": confidence,
        "reason": reason,
    }
