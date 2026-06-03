"""Stereo-equirectangular detection via seam-aware similarity checks."""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def compute_histogram(frame_half: np.ndarray) -> np.ndarray:
    """Compute a normalised grayscale histogram (256 bins) for *frame_half*."""
    gray = cv2.cvtColor(frame_half, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> Dict[str, float]:
    """Compare two histograms using correlation and Bhattacharyya distance."""
    return {
        "correlation": float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)),
        "bhattacharyya": float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)),
    }


def _compute_edge_similarity(region_a: np.ndarray, region_b: np.ndarray) -> float:
    """Compute edge-map IoU similarity in [0, 1] between two regions."""
    if region_a.size == 0 or region_b.size == 0:
        return 0.0
    gray_a = cv2.cvtColor(region_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(region_b, cv2.COLOR_BGR2GRAY)

    edges_a = cv2.Canny(gray_a, 80, 160, apertureSize=3)
    edges_b = cv2.Canny(gray_b, 80, 160, apertureSize=3)

    edge_mask_a = edges_a > 0
    edge_mask_b = edges_b > 0
    union = int(np.count_nonzero(edge_mask_a | edge_mask_b))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(edge_mask_a & edge_mask_b))
    return float(inter / union)


def _longest_false_streak(values: List[bool]) -> int:
    """Return the longest consecutive False streak."""
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            if current > longest:
                longest = current
    return longest


def _extract_seam_regions(
    frame: np.ndarray,
    arrangement: str,
    seam_center: Optional[int],
    seam_guard_ratio: float,
    min_valid_half_ratio: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extract comparable regions around a seam while excluding a guard band."""
    height, width = frame.shape[:2]
    if arrangement == "left-right":
        if width < 4:
            return None
        seam_x = int(seam_center) if seam_center is not None else (width // 2)
        seam_x = int(np.clip(seam_x, 1, width - 1))
        guard = max(1, int(round(width * float(seam_guard_ratio))))
        left_end = max(1, seam_x - guard)
        right_start = min(width - 1, seam_x + guard)
        if right_start <= left_end:
            return None

        left = frame[:, :left_end]
        right = frame[:, right_start:]
        min_width = min(left.shape[1], right.shape[1])
        min_required_width = max(4, int(width * float(min_valid_half_ratio)))
        if min_width < min_required_width:
            return None

        left_region = left[:, -min_width:]
        right_region = right[:, :min_width]
        return left_region, right_region

    if height < 4:
        return None
    seam_y = int(seam_center) if seam_center is not None else (height // 2)
    seam_y = int(np.clip(seam_y, 1, height - 1))
    guard = max(1, int(round(height * float(seam_guard_ratio))))
    top_end = max(1, seam_y - guard)
    bottom_start = min(height - 1, seam_y + guard)
    if bottom_start <= top_end:
        return None

    top = frame[:top_end, :]
    bottom = frame[bottom_start:, :]
    min_height = min(top.shape[0], bottom.shape[0])
    min_required_height = max(4, int(height * float(min_valid_half_ratio)))
    if min_height < min_required_height:
        return None

    top_region = top[-min_height:, :]
    bottom_region = bottom[:min_height, :]
    return top_region, bottom_region


def detect_stereo(
    frames: List[np.ndarray],
    indices_con_linea: List[int],
    similarity_threshold: float,
    bhattacharyya_threshold: float = 0.30,
    min_match_ratio: float = 0.60,
    arrangement: str = "up-down",
    line_centers: Optional[Dict[int, int]] = None,
    seam_guard_ratio: float = 0.02,
    min_valid_half_ratio: float = 0.22,
    edge_similarity_threshold: float = 0.08,
    min_frames_required: int = 2,
    min_stability_ratio: float = 0.55,
) -> Dict[str, Any]:
    """Classify whether *frames* show stereo-equirectangular content.

    Args:
        frames: List of decoded BGR frames.
        indices_con_linea: Indices in *frames* that contain a detected seam line
            (horizontal for ``"up-down"``, vertical for ``"left-right"``).
        similarity_threshold: Minimum histogram correlation to count as a match.
        bhattacharyya_threshold: Maximum Bhattacharyya distance to count as a match.
        min_match_ratio: Fraction of evaluated frames that must match.
        arrangement: ``"up-down"`` (top vs bottom half) or ``"left-right"`` (left vs right half).

    Returns:
        Dict with keys: ``is_stereo``, ``frames_evaluados``, ``frames_match``,
        ``frames_no_match``, ``avg_similarity``, ``avg_bhattacharyya``,
        ``match_ratio``, ``min_match_ratio``, ``bhattacharyya_threshold``,
        ``frame_details``.
    """
    correlations: List[float] = []
    bhattacharyyas: List[float] = []
    edge_similarities: List[float] = []
    combined_scores: List[float] = []
    per_frame_matches: List[bool] = []
    frame_details: List[Dict[str, Any]] = []
    frames_evaluados = 0
    frames_match = 0

    line_centers = line_centers or {}

    for idx in indices_con_linea:
        if idx < 0 or idx >= len(frames):
            continue
        frame = frames[idx]
        seam_regions = _extract_seam_regions(
            frame=frame,
            arrangement=arrangement,
            seam_center=line_centers.get(idx),
            seam_guard_ratio=seam_guard_ratio,
            min_valid_half_ratio=min_valid_half_ratio,
        )
        if seam_regions is None:
            continue
        half_a, half_b = seam_regions

        hist_a = compute_histogram(half_a)
        hist_b = compute_histogram(half_b)
        metrics = compare_histograms(hist_a, hist_b)

        corr = metrics["correlation"]
        bhatt = metrics["bhattacharyya"]
        edge_similarity = _compute_edge_similarity(half_a, half_b)
        combined_score = (
            0.55 * float(np.clip(corr, 0.0, 1.0))
            + 0.25 * float(np.clip(1.0 - bhatt, 0.0, 1.0))
            + 0.20 * float(np.clip(edge_similarity, 0.0, 1.0))
        )
        frame_match = (
            (corr >= similarity_threshold)
            and (bhatt <= bhattacharyya_threshold)
            and (edge_similarity >= edge_similarity_threshold)
        )

        correlations.append(corr)
        bhattacharyyas.append(bhatt)
        edge_similarities.append(edge_similarity)
        combined_scores.append(combined_score)
        per_frame_matches.append(frame_match)
        frames_evaluados += 1
        if frame_match:
            frames_match += 1

        if arrangement == "left-right":
            frame_details.append({
                "frame_idx": idx,
                "corr": corr,
                "bhatt": bhatt,
                "edge_similarity": edge_similarity,
                "combined_score": combined_score,
                "match": frame_match,
                "seam_center": int(line_centers.get(idx, frame.shape[1] // 2)),
                "left_half": half_a,
                "right_half": half_b,
            })
        else:
            frame_details.append({
                "frame_idx": idx,
                "corr": corr,
                "bhatt": bhatt,
                "edge_similarity": edge_similarity,
                "combined_score": combined_score,
                "match": frame_match,
                "seam_center": int(line_centers.get(idx, frame.shape[0] // 2)),
                "top_half": half_a,
                "bottom_half": half_b,
            })

    avg_similarity = float(np.mean(correlations)) if correlations else 0.0
    avg_bhattacharyya = float(np.mean(bhattacharyyas)) if bhattacharyyas else 1.0
    avg_edge_similarity = float(np.mean(edge_similarities)) if edge_similarities else 0.0
    avg_combined_score = float(np.mean(combined_scores)) if combined_scores else 0.0
    match_ratio = frames_match / frames_evaluados if frames_evaluados > 0 else 0.0
    longest_mismatch_streak = _longest_false_streak(per_frame_matches)
    stability_ratio = (
        float(1.0 - (longest_mismatch_streak / float(frames_evaluados)))
        if frames_evaluados > 0
        else 0.0
    )
    is_stereo = (
        (frames_evaluados >= int(min_frames_required))
        and (match_ratio >= min_match_ratio)
        and (stability_ratio >= float(min_stability_ratio))
    )

    return {
        "is_stereo": is_stereo,
        "frames_evaluados": frames_evaluados,
        "frames_match": frames_match,
        "frames_no_match": frames_evaluados - frames_match,
        "avg_similarity": avg_similarity,
        "avg_bhattacharyya": avg_bhattacharyya,
        "avg_edge_similarity": avg_edge_similarity,
        "avg_combined_score": avg_combined_score,
        "match_ratio": match_ratio,
        "min_match_ratio": min_match_ratio,
        "longest_mismatch_streak": int(longest_mismatch_streak),
        "stability_ratio": float(stability_ratio),
        "min_frames_required": int(min_frames_required),
        "min_stability_ratio": float(min_stability_ratio),
        "edge_similarity_threshold": float(edge_similarity_threshold),
        "bhattacharyya_threshold": bhattacharyya_threshold,
        "frame_details": frame_details,
    }
