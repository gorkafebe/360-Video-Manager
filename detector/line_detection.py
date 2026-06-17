import logging
from collections import defaultdict
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .preprocessing import prepare_frame_for_line_detection

logger = logging.getLogger(__name__)


MIN_SEARCH_BAND_HALF = 2


def _compute_candidate_quality(
    *,
    line_length: float,
    total_span: float,
    min_coverage_px: float,
    strong_coverage_px: float,
    distance_from_center: float,
    center_tolerance: float,
    slope: float,
    max_slope: float,
    continuity_ratio: float,
) -> Dict[str, float]:
    """Return quality components and final score for a seam candidate."""
    coverage_norm = float(np.clip(line_length / max(min_coverage_px, 1.0), 0.0, 2.0)) / 2.0
    strong_coverage_norm = float(np.clip(line_length / max(strong_coverage_px, 1.0), 0.0, 1.0))
    center_score = 1.0 - float(
        np.clip(distance_from_center / max(center_tolerance, 1.0), 0.0, 1.0)
    )
    slope_score = 1.0 - float(np.clip(slope / max(max_slope, 1e-6), 0.0, 1.0))
    continuity_score = float(np.clip(continuity_ratio, 0.0, 1.0))
    span_score = float(np.clip(line_length / max(total_span, 1.0), 0.0, 1.0))

    quality_score = (
        0.30 * coverage_norm
        + 0.20 * strong_coverage_norm
        + 0.20 * center_score
        + 0.15 * slope_score
        + 0.10 * continuity_score
        + 0.05 * span_score
    )
    return {
        "coverage_norm": float(coverage_norm),
        "strong_coverage_norm": float(strong_coverage_norm),
        "center_score": float(center_score),
        "slope_score": float(slope_score),
        "continuity_score": float(continuity_score),
        "span_score": float(span_score),
        "quality_score": float(np.clip(quality_score, 0.0, 1.0)),
    }


def _verify_line_with_fft(gray_roi: np.ndarray, min_dominance: float = 0.10) -> Dict[str, Any]:
    """Confirms a detected horizontal line via spectral analysis of the band ROI.

    A genuine horizontal seam produces strong energy along the DC-column region
    (u≈0, all v) in the 2-D power spectrum. Returns 'confirmed' (bool),
    'confidence' (float in [0, 1]), and 'dominance' (signed float in [-1, 1]).
    """
    if gray_roi.size == 0 or gray_roi.shape[0] < 4 or gray_roi.shape[1] < 4:
        return {"confirmed": False, "confidence": 0.0, "dominance": 0.0}

    gray_f = gray_roi.astype(np.float32) / 255.0
    _h, _w = gray_f.shape
    gray_f = gray_f * np.outer(np.hanning(_h), np.hanning(_w))
    fft2 = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    magnitude[cy, cx] = 0.0  # suppress DC bin

    # Column strip around cx (u≈0 in frequency domain): captures energy from horizontal
    # edges, which vary in y but are constant in x — exactly what a face seam looks like.
    col_band = max(1, int(w * 0.10))
    horiz_edge_energy = float(
        np.sum(magnitude[:, max(0, cx - col_band) : min(w, cx + col_band + 1)])
    )

    # Row strip around cy (v≈0 in frequency domain): captures energy from vertical
    # edges, which vary in x but are constant in y.
    row_band = max(1, int(h * 0.10))
    vert_edge_energy = float(
        np.sum(magnitude[max(0, cy - row_band) : min(h, cy + row_band + 1), :])
    )

    dominance = (horiz_edge_energy - vert_edge_energy) / (
        horiz_edge_energy + vert_edge_energy + 1e-8
    )
    confidence = float(np.clip((dominance + 1.0) / 2.0, 0.0, 1.0))
    confirmed = dominance >= min_dominance

    return {"confirmed": bool(confirmed), "confidence": confidence, "dominance": float(dominance)}


def detect_horizontal_line(
    frame: np.ndarray,
    center_tolerance_ratio: float = 0.02,
    search_band_ratio: float = 0.02,
    max_slope: float = 0.05,
    min_coverage_ratio: float = 0.20,
    min_quality_score: float = 0.62,
    fallback_min_quality_score: float = 0.78,
    fft_min_dominance: float = 0.10,
    strong_coverage_ratio: float = 0.40,
    morph_length_ratio: float = 0.02,
    enable_profile_gate: bool = False,
    profile_min_coverage_ratio: float = 0.20,
    profile_min_prominence: float = 3.0,
) -> Dict[str, Any]:
    """Detect a centred horizontal seam line in *frame*.

    Applies Hough → LSD → fitLine candidate detection on a narrow centre strip,
    followed by a quality gate that requires FFT spectral confirmation on all paths.
    All numeric thresholds are configurable via function parameters (backed by env vars
    in config/settings.py using the VPD_LINE_* prefix).
    """
    try:
        height, width = frame.shape[:2]
        gray = prepare_frame_for_line_detection(frame)

        center_y = height / 2.0
        center_tolerance = height * center_tolerance_ratio
        hough_threshold = max(40, int(width * 0.04))
        hough_min_length = max(30, int(width * 0.15))  # raised from 5% to 15% of width
        hough_max_gap = max(10, int(width * 0.02))
        min_line_length_required = float(hough_min_length)
        min_coverage_px = float(width) * min_coverage_ratio  # minimum merged span
        strong_coverage_px = float(width) * strong_coverage_ratio
        band_half = max(MIN_SEARCH_BAND_HALF, int(round(height * float(search_band_ratio) * 0.5)))
        y_top = max(0, int(center_y) - band_half)
        y_bottom = min(height, int(center_y) + band_half)
        roi = gray[y_top:y_bottom, :]

        max_debug_samples = 50

        debug_line_info: Dict[str, Any] = {
            "center_y": center_y,
            "tolerance": center_tolerance,
            "search_band_top": int(y_top),
            "search_band_bottom": int(y_bottom),
            "search_band_ratio": float(search_band_ratio),
            "total_lines": 0,
            "candidate_lines": [],
            "horizontal_lines": [],
            "rejected_not_horizontal": [],
            "rejected_not_centered": [],
            "rejected_dx_zero": [],
            "rejected_too_short": [],
            "distance_from_center_horizontal_stats": {
                "min": None,
                "mean": None,
                "max": None,
            },
            "slope_ratio_centered_stats": {
                "min": None,
                "mean": None,
                "max": None,
            },
            "selected_line": None,
        }

        horizontal_distances: list[float] = []
        centered_slope_ratios: list[float] = []

        otsu_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_high = float(max(50.0, otsu_thresh))
        canny_low = canny_high * 0.4
        dst = cv2.Canny(roi, canny_low, canny_high, apertureSize=3)
        morph_len = max(15, int(width * morph_length_ratio))
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_len, 1))
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, morph_kernel)

        lines = cv2.HoughLinesP(
            dst,
            1,
            np.pi / 180,
            threshold=hough_threshold,
            minLineLength=hough_min_length,
            maxLineGap=hough_max_gap,
        )

        best_candidate: Optional[Dict[str, Any]] = None

        if lines is not None:
            debug_line_info["total_lines"] = int(len(lines))
            candidates_raw = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                y1 += y_top
                y2 += y_top
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dx == 0:
                    if len(debug_line_info["rejected_dx_zero"]) < max_debug_samples:
                        debug_line_info["rejected_dx_zero"].append(
                            {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                            }
                        )
                    continue

                is_horizontal = (dy / dx) < max_slope

                line_center_y = (y1 + y2) / 2.0
                distance_from_center = abs(line_center_y - center_y)
                is_centered = distance_from_center <= float(center_tolerance)

                line_length = float(np.sqrt(dx ** 2 + dy ** 2))
                is_long_enough = line_length >= float(min_line_length_required)
                slope = (dy / dx) if dx > 0 else float("inf")

                if is_horizontal:
                    horizontal_distances.append(float(distance_from_center))
                    if len(debug_line_info["horizontal_lines"]) < max_debug_samples:
                        debug_line_info["horizontal_lines"].append(
                            {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "line_center_y": float(line_center_y),
                                "distance_from_center": float(distance_from_center),
                                "slope_ratio": float(slope),
                            }
                        )

                if is_centered:
                    centered_slope_ratios.append(float(slope))

                if not is_horizontal and len(debug_line_info["rejected_not_horizontal"]) < max_debug_samples:
                    debug_line_info["rejected_not_horizontal"].append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "dx": int(dx),
                            "dy": int(dy),
                            "slope_ratio": float(slope),
                        }
                    )

                if is_horizontal and not is_centered and len(debug_line_info["rejected_not_centered"]) < max_debug_samples:
                    debug_line_info["rejected_not_centered"].append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "line_center_y": float(line_center_y),
                            "distance_from_center": float(distance_from_center),
                        }
                    )

                if is_horizontal and is_centered:
                    debug_line_info["candidate_lines"].append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "line_length": line_length,
                        }
                    )

                if is_horizontal and is_centered and not is_long_enough and len(debug_line_info["rejected_too_short"]) < max_debug_samples:
                    debug_line_info["rejected_too_short"].append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "line_length": float(line_length),
                            "min_required": float(min_line_length_required),
                        }
                    )

                if is_horizontal and is_centered and is_long_enough:
                    candidates_raw.append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "line_center_y": float(line_center_y),
                            "slope": float(slope),
                        }
                    )

            buckets: Dict[int, list] = defaultdict(list)
            for seg in candidates_raw:
                key = int(round(seg["line_center_y"] / max(1.0, center_tolerance)))
                buckets[key].append(seg)

            best_coverage = 0.0
            for segs in buckets.values():
                intervals = sorted(
                    [(min(s["x1"], s["x2"]), max(s["x1"], s["x2"])) for s in segs]
                )
                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])

                if not merged:
                    continue

                total_coverage = float(sum(max(0, e - s) for s, e in merged))
                mean_y = float(np.mean([s["line_center_y"] for s in segs]))
                largest_merged = float(max((e - s) for s, e in merged)) if merged else 0.0
                continuity_ratio = (
                    float(largest_merged / total_coverage) if total_coverage > 0 else 0.0
                )
                if total_coverage > best_coverage:
                    best_coverage = total_coverage
                    best_candidate = {
                        "x1": int(merged[0][0]),
                        "y1": int(round(mean_y)),
                        "x2": int(merged[-1][1]),
                        "y2": int(round(mean_y)),
                        "line_center_y": mean_y,
                        "distance_from_center": abs(mean_y - center_y),
                        "line_length": total_coverage,
                        "slope": float(np.mean([s["slope"] for s in segs])),
                        "continuity_ratio": float(continuity_ratio),
                    }

            # Coverage gate: Hough candidate must span min_coverage_ratio of frame width
            if best_candidate is not None and best_candidate["line_length"] < min_coverage_px:
                best_candidate = None

        if best_candidate is None:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lsd_detect = lsd.detect(roi)
            lsd_lines = lsd_detect[0] if isinstance(lsd_detect, tuple) else lsd_detect
            if lsd_lines is not None:
                for lsd_line in lsd_lines:
                    x1, y1, x2, y2 = (int(v) for v in lsd_line[0])
                    y1 += y_top
                    y2 += y_top
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dx == 0:
                        continue
                    slope = dy / dx
                    is_horizontal = slope < max_slope
                    line_center_y = (y1 + y2) / 2.0
                    is_centered = abs(line_center_y - center_y) <= float(center_tolerance)
                    line_length = float(np.sqrt(dx ** 2 + dy ** 2))
                    is_long_enough = line_length >= float(min_line_length_required)
                    if is_horizontal and is_centered and is_long_enough:
                        candidate = {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "line_center_y": float(line_center_y),
                            "distance_from_center": float(abs(line_center_y - center_y)),
                            "line_length": float(line_length),
                            "slope": float(slope),
                            "continuity_ratio": 1.0,
                            "from_lsd": True,
                        }
                        if best_candidate is None or candidate["line_length"] > best_candidate["line_length"]:
                            best_candidate = candidate

            # Coverage gate: LSD single-segment must also meet coverage threshold
            if best_candidate is not None and best_candidate.get("from_lsd") and best_candidate["line_length"] < min_coverage_px:
                best_candidate = None

        if best_candidate is None:
            edge_ys, edge_xs = np.where(dst > 0)
            if len(edge_xs) >= 100:  # raised from 20; fitLine on sparse edges is unreliable
                pts = np.column_stack([edge_xs, edge_ys]).astype(np.float32)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L12, 0, 0.01, 0.01)
                vx = float(vx)
                vy = float(vy)
                x0 = float(x0)
                y0 = float(y0)
                if abs(vx) > 1e-6 and abs(vy / vx) < max_slope:
                    fitted_y_abs = y0 + y_top
                    if abs(fitted_y_abs - center_y) <= float(center_tolerance):
                        t_left = (0.0 - x0) / vx if abs(vx) > 1e-9 else 0.0
                        t_right = ((width - 1.0) - x0) / vx if abs(vx) > 1e-9 else 0.0
                        fx1 = 0
                        fy1 = int(round(y0 + (t_left * vy))) + y_top
                        fx2 = width - 1
                        fy2 = int(round(y0 + (t_right * vy))) + y_top
                        best_candidate = {
                            "x1": int(fx1),
                            "y1": int(fy1),
                            "x2": int(fx2),
                            "y2": int(fy2),
                            "line_center_y": float(fitted_y_abs),
                            "distance_from_center": float(abs(fitted_y_abs - center_y)),
                            "line_length": float(width),
                            "slope": float(abs(vy / vx)) if abs(vx) > 1e-9 else 0.0,
                            "continuity_ratio": 1.0,
                            "from_fitline": True,
                        }

            if horizontal_distances:
                distances = np.array(horizontal_distances, dtype=float)
                debug_line_info["distance_from_center_horizontal_stats"] = {
                    "min": float(np.min(distances)),
                    "mean": float(np.mean(distances)),
                    "max": float(np.max(distances)),
                }

            if centered_slope_ratios:
                slopes = np.array(centered_slope_ratios, dtype=float)
                debug_line_info["slope_ratio_centered_stats"] = {
                    "min": float(np.min(slopes)),
                    "mean": float(np.mean(slopes)),
                    "max": float(np.max(slopes)),
                }

        if best_candidate is not None:
            debug_line_info["selected_line"] = {
                "x1": int(best_candidate["x1"]),
                "y1": int(best_candidate["y1"]),
                "x2": int(best_candidate["x2"]),
                "y2": int(best_candidate["y2"]),
                "line_center_y": float(best_candidate["line_center_y"]),
            }
            dx_best = abs(int(best_candidate["x2"]) - int(best_candidate["x1"]))
            dy_best = abs(int(best_candidate["y2"]) - int(best_candidate["y1"]))
            angle_deg = float(np.degrees(np.arctan2(dy_best, dx_best)))

            fft_check = _verify_line_with_fft(roi, min_dominance=fft_min_dominance)
            fft_confirmed = bool(fft_check["confirmed"])

            profile_check = _verify_line_with_projection_profile(
                roi,
                orientation="horizontal",
                min_coverage_ratio=profile_min_coverage_ratio,
                min_prominence=profile_min_prominence,
            )
            profile_confirmed = bool(profile_check["confirmed"])

            continuity_ratio = float(best_candidate.get("continuity_ratio", 1.0))
            quality = _compute_candidate_quality(
                line_length=float(best_candidate["line_length"]),
                total_span=float(width),
                min_coverage_px=float(min_coverage_px),
                strong_coverage_px=float(strong_coverage_px),
                distance_from_center=float(best_candidate["distance_from_center"]),
                center_tolerance=float(center_tolerance),
                slope=float(best_candidate.get("slope", 0.0)),
                max_slope=float(max_slope),
                continuity_ratio=continuity_ratio,
            )

            # Quality gate: FFT spectral confirmation is required on ALL paths.
            # Fallback candidates (LSD / fitLine) also need FFT but apply a stricter quality floor.
            # high_coverage relaxes slope_tight for non-fallback Hough lines only.
            from_fallback = bool(
                best_candidate.get("from_lsd", False) or best_candidate.get("from_fitline", False)
            )
            strict_centered = bool(
                best_candidate["distance_from_center"] <= (float(center_tolerance) * 0.65)
            )
            slope_tight = bool(best_candidate.get("slope", 0.0) <= (float(max_slope) * 0.70))
            continuity_ok = bool(continuity_ratio >= 0.70)
            quality_ok = bool(quality["quality_score"] >= float(min_quality_score))
            fallback_quality_ok = bool(
                quality["quality_score"] >= float(fallback_min_quality_score)
            )
            high_coverage = bool(
                best_candidate["line_length"] >= strong_coverage_px and not from_fallback
            )
            if from_fallback:
                detection_confirmed = bool(
                    fft_confirmed
                    and fallback_quality_ok
                    and strict_centered
                    and slope_tight
                    and continuity_ok
                )
            else:
                detection_confirmed = bool(
                    fft_confirmed
                    and quality_ok
                    and strict_centered
                    and continuity_ok
                    and (slope_tight or high_coverage)
                )
            if enable_profile_gate and not profile_confirmed:
                detection_confirmed = False
            logger.debug(
                "[LINE_H] fft_confirmed=%s dominance=%.3f quality=%.3f "
                "strict_centered=%s slope_tight=%s high_coverage=%s continuity_ok=%s -> %s",
                fft_confirmed,
                fft_check.get("dominance", float("nan")),
                quality["quality_score"],
                strict_centered, slope_tight, high_coverage, continuity_ok,
                "CONFIRMED" if detection_confirmed else "REJECTED",
            )

            debug_line_info["quality_gate"] = {
                "fft_confirmed": fft_confirmed,
                "from_fallback": from_fallback,
                "high_coverage": high_coverage,
                "strict_centered": strict_centered,
                "slope_tight": slope_tight,
                "continuity_ok": continuity_ok,
                "quality_ok": quality_ok,
                "fallback_quality_ok": fallback_quality_ok,
                "detection_confirmed": detection_confirmed,
                "line_length": float(best_candidate["line_length"]),
                "min_coverage_px": float(min_coverage_px),
                "strong_coverage_px": float(strong_coverage_px),
                "quality_score": float(quality["quality_score"]),
                "continuity_ratio": float(continuity_ratio),
                "profile_confirmed": profile_confirmed,
                "profile_coverage_ratio": profile_check["coverage_ratio"],
                "profile_prominence": profile_check["peak_prominence"],
            }
            debug_line_info["quality_metrics"] = quality

            if not detection_confirmed:
                # Candidate found but failed quality gate — preserve debug info, skip detection
                return {
                    "has_horizontal_line": False,
                    "projection_type": "unknown",
                    "fft_confirmed": fft_confirmed,
                    "fft_confidence": fft_check["confidence"],
                    "fft_dominance": fft_check["dominance"],
                    "debug_line_info": debug_line_info,
                    "line_data": None,
                }

            return {
                "has_horizontal_line": True,
                "projection_type": "non_equirectangular",
                "fft_confirmed": fft_confirmed,
                "fft_confidence": fft_check["confidence"],
                "fft_dominance": fft_check["dominance"],
                "debug_line_info": debug_line_info,
                "line_data": {
                    "x1": int(best_candidate["x1"]),
                    "y1": int(best_candidate["y1"]),
                    "x2": int(best_candidate["x2"]),
                    "y2": int(best_candidate["y2"]),
                    "center_y": int(round(float(best_candidate["line_center_y"]))),
                    "distance_from_center": float(best_candidate["distance_from_center"]),
                    "angle_deg": angle_deg,
                    "length": float(best_candidate["line_length"]),
                    "quality_score": float(quality["quality_score"]),
                },
            }

        return {
            "has_horizontal_line": False,
            "projection_type": "unknown",
            "debug_line_info": debug_line_info,
            "line_data": None,
        }

    except Exception as exc:
        return {
            "has_horizontal_line": False,
            "projection_type": "unknown",
            "debug_line_info": {
                "center_y": 0.0,
                "tolerance": 0.0,
                "total_lines": 0,
                "candidate_lines": [],
                "selected_line": None,
            },
            "line_data": None,
            "error": str(exc),
        }


def _verify_vertical_line_with_fft(gray_roi: np.ndarray, min_dominance: float = 0.10) -> Dict[str, Any]:
    """Confirms a detected vertical line via spectral analysis of the vertical band ROI.

    A genuine vertical seam produces strong energy along the DC-row region
    (v≈0, all u) in the 2-D power spectrum — the orientation transpose of the
    horizontal detector.  Returns 'confirmed', 'confidence', 'dominance'.
    """
    if gray_roi.size == 0 or gray_roi.shape[0] < 4 or gray_roi.shape[1] < 4:
        return {"confirmed": False, "confidence": 0.0, "dominance": 0.0}

    gray_f = gray_roi.astype(np.float32) / 255.0
    _h, _w = gray_f.shape
    gray_f = gray_f * np.outer(np.hanning(_h), np.hanning(_w))
    fft2 = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    magnitude[cy, cx] = 0.0  # suppress DC bin

    # Row strip around cy (v≈0): captures energy from vertical edges,
    # which vary in x but are constant in y — exactly what a left-right seam looks like.
    row_band = max(1, int(h * 0.10))
    vert_seam_energy = float(
        np.sum(magnitude[max(0, cy - row_band) : min(h, cy + row_band + 1), :])
    )

    # Column strip around cx (u≈0): captures energy from horizontal edges.
    col_band = max(1, int(w * 0.10))
    horiz_seam_energy = float(
        np.sum(magnitude[:, max(0, cx - col_band) : min(w, cx + col_band + 1)])
    )

    dominance = (vert_seam_energy - horiz_seam_energy) / (
        vert_seam_energy + horiz_seam_energy + 1e-8
    )
    confidence = float(np.clip((dominance + 1.0) / 2.0, 0.0, 1.0))
    confirmed = dominance >= min_dominance

    return {"confirmed": bool(confirmed), "confidence": confidence, "dominance": float(dominance)}


def _verify_line_with_projection_profile(
    gray_roi: np.ndarray,
    orientation: str = "horizontal",
    min_coverage_ratio: float = 0.20,
    min_prominence: float = 3.0,
) -> Dict[str, Any]:
    """Spatial confirmer: checks that gradient spike is localised at one row/column.

    For 'horizontal': Sobel_y, reduce mean-abs across columns → per-row profile;
    peak prominence = peak / (median + eps); coverage = fraction of columns whose
    per-column gradient peak falls within ±1 row of the global peak row.
    For 'vertical': transpose roi before applying the same logic.
    Returns {confirmed, coverage_ratio, peak_prominence, peak_index}.
    """
    _FAIL: Dict[str, Any] = {
        "confirmed": False, "coverage_ratio": 0.0, "peak_prominence": 0.0, "peak_index": -1
    }
    if gray_roi.size == 0 or gray_roi.shape[0] < 4 or gray_roi.shape[1] < 4:
        return _FAIL
    try:
        work = gray_roi if orientation == "horizontal" else gray_roi.T
        h, w = work.shape[:2]
        if h < 4 or w < 4:
            return _FAIL
        sobel = cv2.Sobel(work.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        abs_sobel = np.abs(sobel)                      # (h, w)
        profile = np.mean(abs_sobel, axis=1)           # (h,) per-row mean gradient
        peak_idx = int(np.argmax(profile))
        peak_val = float(profile[peak_idx])
        prominence = peak_val / (float(np.median(profile)) + 1e-8)
        col_peaks = np.argmax(abs_sobel, axis=0)       # (w,) per-col peak row
        near_peak = np.abs(col_peaks.astype(np.int32) - peak_idx) <= 1
        coverage_ratio = float(np.sum(near_peak)) / max(w, 1)
        confirmed = bool(coverage_ratio >= min_coverage_ratio and prominence >= min_prominence)
        return {
            "confirmed": confirmed,
            "coverage_ratio": coverage_ratio,
            "peak_prominence": float(prominence),
            "peak_index": peak_idx,
        }
    except Exception:
        return _FAIL


def detect_vertical_line(
    frame: np.ndarray,
    center_tolerance_ratio: float = 0.02,
    search_band_ratio: float = 0.02,
    max_slope: float = 0.05,
    min_coverage_ratio: float = 0.20,
    min_quality_score: float = 0.62,
    fallback_min_quality_score: float = 0.78,
    fft_min_dominance: float = 0.10,
    strong_coverage_ratio: float = 0.40,
    morph_length_ratio: float = 0.02,
    enable_profile_gate: bool = False,
    profile_min_coverage_ratio: float = 0.20,
    profile_min_prominence: float = 3.0,
) -> Dict[str, Any]:
    """Detect a centred vertical seam line in *frame*.

    Mirrors detect_horizontal_line exactly, with width/height axes swapped.
    The search band is a vertical strip centred at x = width/2.
    Coverage is measured along the frame height.
    All numeric thresholds are configurable via function parameters (backed by env vars
    in config/settings.py using the VPD_LINE_* prefix).
    """
    try:
        height, width = frame.shape[:2]
        gray = prepare_frame_for_line_detection(frame)

        center_x = width / 2.0
        center_tolerance = width * center_tolerance_ratio
        hough_threshold = max(40, int(height * 0.04))
        hough_min_length = max(30, int(height * 0.15))
        hough_max_gap = max(10, int(height * 0.02))
        min_line_length_required = float(hough_min_length)
        min_coverage_px = float(height) * min_coverage_ratio
        strong_coverage_px = float(height) * strong_coverage_ratio
        band_half = max(MIN_SEARCH_BAND_HALF, int(round(width * float(search_band_ratio) * 0.5)))
        x_left = max(0, int(center_x) - band_half)
        x_right = min(width, int(center_x) + band_half)
        roi = gray[:, x_left:x_right]

        max_debug_samples = 50

        debug_line_info: Dict[str, Any] = {
            "center_x": center_x,
            "tolerance": center_tolerance,
            "search_band_left": int(x_left),
            "search_band_right": int(x_right),
            "search_band_ratio": float(search_band_ratio),
            "total_lines": 0,
            "candidate_lines": [],
            "vertical_lines": [],
            "rejected_not_vertical": [],
            "rejected_not_centered": [],
            "rejected_dy_zero": [],
            "rejected_too_short": [],
            "distance_from_center_vertical_stats": {
                "min": None,
                "mean": None,
                "max": None,
            },
            "slope_ratio_centered_stats": {
                "min": None,
                "mean": None,
                "max": None,
            },
            "selected_line": None,
        }

        vertical_distances: list[float] = []
        centered_slope_ratios: list[float] = []

        otsu_thresh, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_high = float(max(50.0, otsu_thresh))
        canny_low = canny_high * 0.4
        dst = cv2.Canny(roi, canny_low, canny_high, apertureSize=3)
        morph_len = max(15, int(height * morph_length_ratio))
        # Vertical morphology kernel (1 × morph_len) to connect vertical edge fragments
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_len))
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, morph_kernel)

        lines = cv2.HoughLinesP(
            dst,
            1,
            np.pi / 180,
            threshold=hough_threshold,
            minLineLength=hough_min_length,
            maxLineGap=hough_max_gap,
        )

        best_candidate: Optional[Dict[str, Any]] = None

        if lines is not None:
            debug_line_info["total_lines"] = int(len(lines))
            candidates_raw = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1 += x_left
                x2 += x_left
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dy == 0:
                    if len(debug_line_info["rejected_dy_zero"]) < max_debug_samples:
                        debug_line_info["rejected_dy_zero"].append(
                            {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                        )
                    continue

                # A vertical line has a very small dx/dy ratio (analogous to dy/dx for horizontal)
                is_vertical = (dx / dy) < max_slope

                line_center_x = (x1 + x2) / 2.0
                distance_from_center = abs(line_center_x - center_x)
                is_centered = distance_from_center <= float(center_tolerance)

                line_length = float(np.sqrt(dx ** 2 + dy ** 2))
                is_long_enough = line_length >= float(min_line_length_required)
                slope = (dx / dy) if dy > 0 else float("inf")  # dx/dy for vertical lines

                if is_vertical:
                    vertical_distances.append(float(distance_from_center))
                    if len(debug_line_info["vertical_lines"]) < max_debug_samples:
                        debug_line_info["vertical_lines"].append(
                            {
                                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                                "line_center_x": float(line_center_x),
                                "distance_from_center": float(distance_from_center),
                                "slope_ratio": float(slope),
                            }
                        )

                if is_centered:
                    centered_slope_ratios.append(float(slope))

                if not is_vertical and len(debug_line_info["rejected_not_vertical"]) < max_debug_samples:
                    debug_line_info["rejected_not_vertical"].append(
                        {
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "dx": int(dx), "dy": int(dy), "slope_ratio": float(slope),
                        }
                    )

                if is_vertical and not is_centered and len(debug_line_info["rejected_not_centered"]) < max_debug_samples:
                    debug_line_info["rejected_not_centered"].append(
                        {
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "line_center_x": float(line_center_x),
                            "distance_from_center": float(distance_from_center),
                        }
                    )

                if is_vertical and is_centered:
                    debug_line_info["candidate_lines"].append(
                        {
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "line_length": line_length,
                        }
                    )

                if is_vertical and is_centered and not is_long_enough and len(debug_line_info["rejected_too_short"]) < max_debug_samples:
                    debug_line_info["rejected_too_short"].append(
                        {
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "line_length": float(line_length),
                            "min_required": float(min_line_length_required),
                        }
                    )

                if is_vertical and is_centered and is_long_enough:
                    candidates_raw.append(
                        {
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "line_center_x": float(line_center_x),
                            "slope": float(slope),
                        }
                    )

            # Bucket segments by their center_x row (analogous to center_y bucketing)
            buckets: Dict[int, list] = defaultdict(list)
            for seg in candidates_raw:
                key = int(round(seg["line_center_x"] / max(1.0, center_tolerance)))
                buckets[key].append(seg)

            best_coverage = 0.0
            for segs in buckets.values():
                # Coverage measured along the Y axis (height) for vertical lines
                intervals = sorted(
                    [(min(s["y1"], s["y2"]), max(s["y1"], s["y2"])) for s in segs]
                )
                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])

                if not merged:
                    continue

                total_coverage = float(sum(max(0, e - s) for s, e in merged))
                mean_x = float(np.mean([s["line_center_x"] for s in segs]))
                largest_merged = float(max((e - s) for s, e in merged)) if merged else 0.0
                continuity_ratio = (
                    float(largest_merged / total_coverage) if total_coverage > 0 else 0.0
                )
                if total_coverage > best_coverage:
                    best_coverage = total_coverage
                    best_candidate = {
                        "x1": int(round(mean_x)),
                        "y1": int(merged[0][0]),
                        "x2": int(round(mean_x)),
                        "y2": int(merged[-1][1]),
                        "line_center_x": mean_x,
                        "distance_from_center": abs(mean_x - center_x),
                        "line_length": total_coverage,
                        "slope": float(np.mean([s["slope"] for s in segs])),
                        "continuity_ratio": float(continuity_ratio),
                    }

            # Coverage gate: Hough candidate must span min_coverage_ratio of frame height
            if best_candidate is not None and best_candidate["line_length"] < min_coverage_px:
                best_candidate = None

        if best_candidate is None:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lsd_detect = lsd.detect(roi)
            lsd_lines = lsd_detect[0] if isinstance(lsd_detect, tuple) else lsd_detect
            if lsd_lines is not None:
                for lsd_line in lsd_lines:
                    x1, y1, x2, y2 = (int(v) for v in lsd_line[0])
                    x1 += x_left
                    x2 += x_left
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dy == 0:
                        continue
                    slope = dx / dy
                    is_vertical = slope < max_slope
                    line_center_x = (x1 + x2) / 2.0
                    is_centered = abs(line_center_x - center_x) <= float(center_tolerance)
                    line_length = float(np.sqrt(dx ** 2 + dy ** 2))
                    is_long_enough = line_length >= float(min_line_length_required)
                    if is_vertical and is_centered and is_long_enough:
                        candidate = {
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "line_center_x": float(line_center_x),
                            "distance_from_center": float(abs(line_center_x - center_x)),
                            "line_length": float(line_length),
                            "slope": float(slope),
                            "continuity_ratio": 1.0,
                            "from_lsd": True,
                        }
                        if best_candidate is None or candidate["line_length"] > best_candidate["line_length"]:
                            best_candidate = candidate

            # Coverage gate for LSD single-segment
            if best_candidate is not None and best_candidate.get("from_lsd") and best_candidate["line_length"] < min_coverage_px:
                best_candidate = None

        if best_candidate is None:
            edge_ys, edge_xs = np.where(dst > 0)
            if len(edge_ys) >= 100:
                pts = np.column_stack([edge_xs, edge_ys]).astype(np.float32)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L12, 0, 0.01, 0.01)
                vx = float(vx)
                vy = float(vy)
                x0 = float(x0)
                y0 = float(y0)
                # For a vertical line the dominant direction is Y; check vy>>vx
                if abs(vy) > 1e-6 and abs(vx / vy) < max_slope:
                    fitted_x_abs = x0 + x_left
                    if abs(fitted_x_abs - center_x) <= float(center_tolerance):
                        t_top = (0.0 - y0) / vy if abs(vy) > 1e-9 else 0.0
                        t_bottom = ((height - 1.0) - y0) / vy if abs(vy) > 1e-9 else 0.0
                        fy1 = 0
                        fx1 = int(round(x0 + (t_top * vx))) + x_left
                        fy2 = height - 1
                        fx2 = int(round(x0 + (t_bottom * vx))) + x_left
                        best_candidate = {
                            "x1": int(fx1),
                            "y1": int(fy1),
                            "x2": int(fx2),
                            "y2": int(fy2),
                            "line_center_x": float(fitted_x_abs),
                            "distance_from_center": float(abs(fitted_x_abs - center_x)),
                            "line_length": float(height),
                            "slope": float(abs(vx / vy)) if abs(vy) > 1e-9 else 0.0,
                            "continuity_ratio": 1.0,
                            "from_fitline": True,
                        }

            if vertical_distances:
                distances = np.array(vertical_distances, dtype=float)
                debug_line_info["distance_from_center_vertical_stats"] = {
                    "min": float(np.min(distances)),
                    "mean": float(np.mean(distances)),
                    "max": float(np.max(distances)),
                }

            if centered_slope_ratios:
                slopes = np.array(centered_slope_ratios, dtype=float)
                debug_line_info["slope_ratio_centered_stats"] = {
                    "min": float(np.min(slopes)),
                    "mean": float(np.mean(slopes)),
                    "max": float(np.max(slopes)),
                }

        if best_candidate is not None:
            debug_line_info["selected_line"] = {
                "x1": int(best_candidate["x1"]),
                "y1": int(best_candidate["y1"]),
                "x2": int(best_candidate["x2"]),
                "y2": int(best_candidate["y2"]),
                "line_center_x": float(best_candidate["line_center_x"]),
            }
            dx_best = abs(int(best_candidate["x2"]) - int(best_candidate["x1"]))
            dy_best = abs(int(best_candidate["y2"]) - int(best_candidate["y1"]))
            angle_deg = float(np.degrees(np.arctan2(dx_best, dy_best)))  # angle from vertical

            fft_check = _verify_vertical_line_with_fft(roi, min_dominance=fft_min_dominance)
            fft_confirmed = bool(fft_check["confirmed"])

            profile_check = _verify_line_with_projection_profile(
                roi,
                orientation="vertical",
                min_coverage_ratio=profile_min_coverage_ratio,
                min_prominence=profile_min_prominence,
            )
            profile_confirmed = bool(profile_check["confirmed"])

            continuity_ratio = float(best_candidate.get("continuity_ratio", 1.0))
            quality = _compute_candidate_quality(
                line_length=float(best_candidate["line_length"]),
                total_span=float(height),
                min_coverage_px=float(min_coverage_px),
                strong_coverage_px=float(strong_coverage_px),
                distance_from_center=float(best_candidate["distance_from_center"]),
                center_tolerance=float(center_tolerance),
                slope=float(best_candidate.get("slope", 0.0)),
                max_slope=float(max_slope),
                continuity_ratio=continuity_ratio,
            )

            from_fallback = bool(
                best_candidate.get("from_lsd", False) or best_candidate.get("from_fitline", False)
            )
            strict_centered = bool(
                best_candidate["distance_from_center"] <= (float(center_tolerance) * 0.65)
            )
            slope_tight = bool(best_candidate.get("slope", 0.0) <= (float(max_slope) * 0.70))
            continuity_ok = bool(continuity_ratio >= 0.70)
            quality_ok = bool(quality["quality_score"] >= float(min_quality_score))
            fallback_quality_ok = bool(
                quality["quality_score"] >= float(fallback_min_quality_score)
            )
            high_coverage = bool(
                best_candidate["line_length"] >= strong_coverage_px and not from_fallback
            )
            if from_fallback:
                detection_confirmed = bool(
                    fft_confirmed
                    and fallback_quality_ok
                    and strict_centered
                    and slope_tight
                    and continuity_ok
                )
            else:
                detection_confirmed = bool(
                    fft_confirmed
                    and quality_ok
                    and strict_centered
                    and continuity_ok
                    and (slope_tight or high_coverage)
                )
            if enable_profile_gate and not profile_confirmed:
                detection_confirmed = False
            logger.debug(
                "[LINE_V] fft_confirmed=%s dominance=%.3f quality=%.3f "
                "strict_centered=%s slope_tight=%s high_coverage=%s continuity_ok=%s -> %s",
                fft_confirmed,
                fft_check.get("dominance", float("nan")),
                quality["quality_score"],
                strict_centered, slope_tight, high_coverage, continuity_ok,
                "CONFIRMED" if detection_confirmed else "REJECTED",
            )

            debug_line_info["quality_gate"] = {
                "fft_confirmed": fft_confirmed,
                "from_fallback": from_fallback,
                "high_coverage": high_coverage,
                "strict_centered": strict_centered,
                "slope_tight": slope_tight,
                "continuity_ok": continuity_ok,
                "quality_ok": quality_ok,
                "fallback_quality_ok": fallback_quality_ok,
                "detection_confirmed": detection_confirmed,
                "line_length": float(best_candidate["line_length"]),
                "min_coverage_px": float(min_coverage_px),
                "strong_coverage_px": float(strong_coverage_px),
                "quality_score": float(quality["quality_score"]),
                "continuity_ratio": float(continuity_ratio),
                "profile_confirmed": profile_confirmed,
                "profile_coverage_ratio": profile_check["coverage_ratio"],
                "profile_prominence": profile_check["peak_prominence"],
            }
            debug_line_info["quality_metrics"] = quality

            if not detection_confirmed:
                return {
                    "has_vertical_line": False,
                    "projection_type": "unknown",
                    "fft_confirmed": fft_confirmed,
                    "fft_confidence": fft_check["confidence"],
                    "fft_dominance": fft_check["dominance"],
                    "debug_line_info": debug_line_info,
                    "line_data": None,
                }

            return {
                "has_vertical_line": True,
                "projection_type": "non_equirectangular",
                "fft_confirmed": fft_confirmed,
                "fft_confidence": fft_check["confidence"],
                "fft_dominance": fft_check["dominance"],
                "debug_line_info": debug_line_info,
                "line_data": {
                    "x1": int(best_candidate["x1"]),
                    "y1": int(best_candidate["y1"]),
                    "x2": int(best_candidate["x2"]),
                    "y2": int(best_candidate["y2"]),
                    "center_x": int(round(float(best_candidate["line_center_x"]))),
                    "distance_from_center": float(best_candidate["distance_from_center"]),
                    "angle_deg": angle_deg,
                    "length": float(best_candidate["line_length"]),
                    "quality_score": float(quality["quality_score"]),
                },
            }

        return {
            "has_vertical_line": False,
            "projection_type": "unknown",
            "debug_line_info": debug_line_info,
            "line_data": None,
        }

    except Exception as exc:
        return {
            "has_vertical_line": False,
            "projection_type": "unknown",
            "debug_line_info": {
                "center_x": 0.0,
                "tolerance": 0.0,
                "total_lines": 0,
                "candidate_lines": [],
                "selected_line": None,
            },
            "line_data": None,
            "error": str(exc),
        }


def draw_line_debug(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Dibuja visual debug de búsqueda/candidatas/selección para inspección."""
    vis = frame.copy()
    height, width = vis.shape[:2]
    debug_line_info = result.get("debug_line_info", {})
    found = bool(result.get("has_horizontal_line", result.get("has_vertical_line", False)))
    is_vertical = bool(result.get("line_orientation") == "vertical" or "center_x" in debug_line_info)

    if is_vertical:
        center_x = float(debug_line_info.get("center_x", width / 2.0))
        tolerance = float(debug_line_info.get("tolerance", width * 0.02))
        left = int(debug_line_info.get("search_band_left", max(0, int(round(center_x - tolerance)))))
        right = int(debug_line_info.get("search_band_right", min(width - 1, int(round(center_x + tolerance)))))

        overlay = vis.copy()
        cv2.rectangle(overlay, (left, 0), (right, height - 1), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.16, vis, 0.84, 0.0, vis)
        cv2.line(vis, (left, 0), (left, height - 1), (255, 0, 0), 2)
        cv2.line(vis, (right, 0), (right, height - 1), (255, 0, 0), 2)

        for c in debug_line_info.get("candidate_lines", []):
            cv2.line(vis, (int(c["x1"]), int(c["y1"])), (int(c["x2"]), int(c["y2"])), (0, 255, 255), 1)

        selected = debug_line_info.get("selected_line")
        if selected is not None:
            x1 = int(selected["x1"])
            y1 = int(selected["y1"])
            x2 = int(selected["x2"])
            y2 = int(selected["y2"])
            line_center_x = int(round(float(selected.get("line_center_x", center_x))))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(vis, (line_center_x, height // 2), 8, (0, 255, 0), -1)

        text_lines = [
            f"center_x={center_x:.1f}",
            f"tolerance={tolerance:.1f}",
            f"search_band={left}:{right}",
            f"hough_lines={int(debug_line_info.get('total_lines', 0))}",
            f"candidates={len(debug_line_info.get('candidate_lines', []))}",
            f"result={'FOUND' if found else 'NOT FOUND'}",
        ]
    else:
        center_y = float(debug_line_info.get("center_y", height / 2.0))
        tolerance = float(debug_line_info.get("tolerance", height * 0.02))
        top = int(debug_line_info.get("search_band_top", max(0, int(round(center_y - tolerance)))))
        bottom = int(debug_line_info.get("search_band_bottom", min(height - 1, int(round(center_y + tolerance)))))

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, top), (width - 1, bottom), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.16, vis, 0.84, 0.0, vis)
        cv2.line(vis, (0, top), (width - 1, top), (255, 0, 0), 2)
        cv2.line(vis, (0, bottom), (width - 1, bottom), (255, 0, 0), 2)

        for c in debug_line_info.get("candidate_lines", []):
            cv2.line(vis, (int(c["x1"]), int(c["y1"])), (int(c["x2"]), int(c["y2"])), (0, 255, 255), 1)

        selected = debug_line_info.get("selected_line")
        if selected is not None:
            x1 = int(selected["x1"])
            y1 = int(selected["y1"])
            x2 = int(selected["x2"])
            y2 = int(selected["y2"])
            line_center_y = int(round(float(selected.get("line_center_y", center_y))))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(vis, (width // 2, line_center_y), 8, (0, 255, 0), -1)

        text_lines = [
            f"center_y={center_y:.1f}",
            f"tolerance={tolerance:.1f}",
            f"search_band={top}:{bottom}",
            f"hough_lines={int(debug_line_info.get('total_lines', 0))}",
            f"candidates={len(debug_line_info.get('candidate_lines', []))}",
            f"result={'FOUND' if found else 'NOT FOUND'}",
        ]

    y = 26
    for t in text_lines:
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24

    return vis
