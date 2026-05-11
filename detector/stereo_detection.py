"""Stereo-equirectangular detection via histogram comparison."""

from typing import Any, Dict, List

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


def detect_stereo(
    frames: List[np.ndarray],
    indices_con_linea: List[int],
    similarity_threshold: float,
    bhattacharyya_threshold: float = 0.30,
    min_match_ratio: float = 0.60,
    arrangement: str = "up-down",
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
    frame_details: List[Dict[str, Any]] = []
    frames_evaluados = 0
    frames_match = 0

    for idx in indices_con_linea:
        if idx < 0 or idx >= len(frames):
            continue
        frame = frames[idx]

        if arrangement == "left-right":
            width = frame.shape[1]
            if width < 2:
                continue
            half_a = frame[:, : width // 2]
            half_b = frame[:, width // 2 :]
        else:
            height = frame.shape[0]
            if height < 2:
                continue
            half_a = frame[: height // 2, :]
            half_b = frame[height // 2 :, :]

        hist_a = compute_histogram(half_a)
        hist_b = compute_histogram(half_b)
        metrics = compare_histograms(hist_a, hist_b)

        corr = metrics["correlation"]
        bhatt = metrics["bhattacharyya"]
        frame_match = (corr >= similarity_threshold) and (bhatt <= bhattacharyya_threshold)

        correlations.append(corr)
        bhattacharyyas.append(bhatt)
        frames_evaluados += 1
        if frame_match:
            frames_match += 1

        if arrangement == "left-right":
            frame_details.append({
                "frame_idx": idx,
                "corr": corr,
                "bhatt": bhatt,
                "match": frame_match,
                "left_half": half_a,
                "right_half": half_b,
            })
        else:
            frame_details.append({
                "frame_idx": idx,
                "corr": corr,
                "bhatt": bhatt,
                "match": frame_match,
                "top_half": half_a,
                "bottom_half": half_b,
            })

    avg_similarity = float(np.mean(correlations)) if correlations else 0.0
    avg_bhattacharyya = float(np.mean(bhattacharyyas)) if bhattacharyyas else 1.0
    match_ratio = frames_match / frames_evaluados if frames_evaluados > 0 else 0.0
    is_stereo = (frames_evaluados > 0) and (match_ratio >= min_match_ratio)

    return {
        "is_stereo": is_stereo,
        "frames_evaluados": frames_evaluados,
        "frames_match": frames_match,
        "frames_no_match": frames_evaluados - frames_match,
        "avg_similarity": avg_similarity,
        "avg_bhattacharyya": avg_bhattacharyya,
        "match_ratio": match_ratio,
        "min_match_ratio": min_match_ratio,
        "bhattacharyya_threshold": bhattacharyya_threshold,
        "frame_details": frame_details,
    }
