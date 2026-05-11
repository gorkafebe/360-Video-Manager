from collections import defaultdict
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .preprocessing import prepare_frame_for_line_detection


def _verify_line_with_fft(gray_roi: np.ndarray, min_dominance: float = 0.10) -> Dict[str, Any]:
    """Confirms a detected horizontal line via spectral analysis of the band ROI.

    A genuine horizontal seam produces strong energy along the DC-column region
    (u≈0, all v) in the 2-D power spectrum. Returns 'confirmed' (bool),
    'confidence' (float in [0, 1]), and 'dominance' (signed float in [-1, 1]).
    """
    if gray_roi.size == 0 or gray_roi.shape[0] < 4 or gray_roi.shape[1] < 4:
        return {"confirmed": False, "confidence": 0.0, "dominance": 0.0}

    gray_f = gray_roi.astype(np.float32) / 255.0
    fft2 = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

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
    band_ratio: float = 0.08,
    max_slope: float = 0.05,
    min_coverage_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Detecta línea horizontal centrada sin efectos secundarios."""
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
        strong_coverage_px = float(width) * 0.40  # span that overrides FFT requirement
        band_half = int(height * band_ratio / 4.0)
        y_top = max(0, int(center_y) - band_half)
        y_bottom = min(height, int(center_y) + band_half)
        roi = gray[y_top:y_bottom, :]

        max_debug_samples = 50

        debug_line_info: Dict[str, Any] = {
            "center_y": center_y,
            "tolerance": center_tolerance,
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
        morph_len = max(15, int(width * 0.02))
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

            fft_check = _verify_line_with_fft(roi)
            fft_confirmed = bool(fft_check["confirmed"])

            # Quality gate: require FFT confirmation OR high non-fallback Hough coverage.
            # Fallback candidates (LSD / fitLine) always require FFT to be confirmed because
            # they can be constructed from noisy edge clusters without real segment evidence.
            from_fallback = bool(
                best_candidate.get("from_lsd", False) or best_candidate.get("from_fitline", False)
            )
            high_coverage = bool(
                best_candidate["line_length"] >= strong_coverage_px and not from_fallback
            )
            detection_confirmed = fft_confirmed or high_coverage

            debug_line_info["quality_gate"] = {
                "fft_confirmed": fft_confirmed,
                "from_fallback": from_fallback,
                "high_coverage": high_coverage,
                "detection_confirmed": detection_confirmed,
                "line_length": float(best_candidate["line_length"]),
                "min_coverage_px": float(min_coverage_px),
                "strong_coverage_px": float(strong_coverage_px),
            }

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
    fft2 = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

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


def detect_vertical_line(
    frame: np.ndarray,
    center_tolerance_ratio: float = 0.02,
    band_ratio: float = 0.08,
    max_slope: float = 0.05,
    min_coverage_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Detecta línea vertical centrada (seam izquierda-derecha) sin efectos secundarios.

    Mirrors detect_horizontal_line exactly, with width/height axes swapped.
    The search band is a vertical strip centred at x = width/2.
    Coverage is measured along the frame height.
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
        strong_coverage_px = float(height) * 0.40
        band_half = int(width * band_ratio / 4.0)
        x_left = max(0, int(center_x) - band_half)
        x_right = min(width, int(center_x) + band_half)
        roi = gray[:, x_left:x_right]

        max_debug_samples = 50

        debug_line_info: Dict[str, Any] = {
            "center_x": center_x,
            "tolerance": center_tolerance,
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
        morph_len = max(15, int(height * 0.02))
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

            fft_check = _verify_vertical_line_with_fft(roi)
            fft_confirmed = bool(fft_check["confirmed"])

            from_fallback = bool(
                best_candidate.get("from_lsd", False) or best_candidate.get("from_fitline", False)
            )
            high_coverage = bool(
                best_candidate["line_length"] >= strong_coverage_px and not from_fallback
            )
            detection_confirmed = fft_confirmed or high_coverage

            debug_line_info["quality_gate"] = {
                "fft_confirmed": fft_confirmed,
                "from_fallback": from_fallback,
                "high_coverage": high_coverage,
                "detection_confirmed": detection_confirmed,
                "line_length": float(best_candidate["line_length"]),
                "min_coverage_px": float(min_coverage_px),
                "strong_coverage_px": float(strong_coverage_px),
            }

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
        left = max(0, int(round(center_x - tolerance)))
        right = min(width - 1, int(round(center_x + tolerance)))

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
            f"hough_lines={int(debug_line_info.get('total_lines', 0))}",
            f"candidates={len(debug_line_info.get('candidate_lines', []))}",
            f"result={'FOUND' if found else 'NOT FOUND'}",
        ]
    else:
        center_y = float(debug_line_info.get("center_y", height / 2.0))
        tolerance = float(debug_line_info.get("tolerance", height * 0.02))
        top = max(0, int(round(center_y - tolerance)))
        bottom = min(height - 1, int(round(center_y + tolerance)))

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
            f"hough_lines={int(debug_line_info.get('total_lines', 0))}",
            f"candidates={len(debug_line_info.get('candidate_lines', []))}",
            f"result={'FOUND' if found else 'NOT FOUND'}",
        ]

    y = 26
    for t in text_lines:
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24

    return vis
