import logging
import os
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

from .debug_utils import (
    create_run_debug_dir,
    draw_motion_vectors,
    draw_regions,
    format_final_stats,
    log_discard,
    log_success,
    save_frame_debug,
    save_line_detected_frame,
    save_line_visual_debug,
    save_stereo_halves,
)
from .line_detection import detect_horizontal_line, detect_vertical_line
from .motion_analysis import (
    compute_forward_backward_consistency_mask,
    compute_global_geometry_evidence,
    compute_optical_flow,
    compute_region_affine_angles,
    compute_region_motion,
    split_into_regions,
)
from .projection_logic import decide_projection, evaluate_cubemap, evaluate_eac
from .region_validation import is_region_valid
from .equirectangular_detection import aggregate_equirectangular_evidence, compute_frame_equirectangular_evidence
from .projection_conversion import convert_detected_projection_to_equirectangular
from .stereo_detection import detect_stereo
from .video_io import FrameExtractorError, convert_video_codec, extract_main_frames, extract_secondary_frames


logger = logging.getLogger(__name__)


from config.settings import get_settings as _get_settings

def load_config():
    """Return the detector config dict from the central settings."""
    return _get_settings().as_detector_config()

CONFIG = load_config()


def _resolve_motion_feature_flags(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or CONFIG
    profile = str(cfg.get("motion_rollout_profile", "high_accuracy")).strip().lower()
    enable_refinement = bool(cfg.get("flow_enable_refinement", False))
    enable_fb_check = bool(cfg.get("flow_enable_fb_check", False))
    enable_geometry_evidence = bool(cfg.get("enable_geometry_evidence", False))

    if profile == "robust":
        enable_fb_check = True
        enable_geometry_evidence = True
    elif profile == "high_accuracy":
        enable_refinement = True
        enable_fb_check = True
        enable_geometry_evidence = True
    else:
        profile = "baseline"

    return {
        "profile": profile,
        "enable_refinement": enable_refinement,
        "enable_fb_check": enable_fb_check,
        "fb_threshold": float(cfg.get("flow_fb_threshold", 1.5)),
        "enable_geometry_evidence": enable_geometry_evidence,
        "geometry_evidence_weight": float(cfg.get("geometry_evidence_weight", 0.20)),
    }


def _fuse_optional_score(
    base_score: Optional[float],
    optional_score: Optional[float],
    optional_weight: float,
) -> Optional[float]:
    if base_score is None:
        return None
    if optional_score is None:
        return float(base_score)
    w = float(np.clip(optional_weight, 0.0, 0.45))
    return float((1.0 - w) * float(base_score) + w * float(optional_score))


def frame_esta_en_negro(frame: np.ndarray, umbral_intensidad: int = 16, proporcion_minima_oscura: float = 0.98) -> bool:
    if frame is None or frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixeles_oscuros = np.sum(gray <= umbral_intensidad)
    proporcion_oscura = pixeles_oscuros / gray.size
    return bool(proporcion_oscura >= proporcion_minima_oscura)


def evaluar_suficiencia_datos(
    total_frames_extraidos: int,
    frames_analizados: int,
    frames_descartados: int,
    min_frames_analyzed_required: int,
    max_discard_ratio: float,
) -> List[str]:
    reasons: List[str] = []
    if frames_analizados < min_frames_analyzed_required:
        reasons.append(f"few_analyzed_frames:{frames_analizados}<{min_frames_analyzed_required}")
    if total_frames_extraidos > 0:
        discard_ratio = frames_descartados / float(total_frames_extraidos)
        if discard_ratio > max_discard_ratio:
            reasons.append(f"high_discard_ratio:{discard_ratio:.2f}>{max_discard_ratio:.2f}")
    return reasons


def _analyze_motion_pair_detailed(
    frame1: np.ndarray,
    frame2: np.ndarray,
    umbral_magnitud: float = 1.0,
    proporcion_minima_pixeles: float = 0.01,
    tolerancia_45_deg: float = 20.0,
    umbral_concentracion: float = 0.25,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
    min_active_ratio: float = 0.06,
    min_layout_score_margin: float = 0.0,
    flow_algorithm: str = "farneback",
    enable_flow_refinement: bool = False,
    enable_forward_backward_check: bool = False,
    forward_backward_threshold: float = 1.5,
    enable_geometry_evidence: bool = False,
    geometry_evidence_weight: float = 0.20,
) -> Dict[str, Any]:
    try:
        flow = compute_optical_flow(
            frame1,
            frame2,
            gaussian_kernel_size,
            gaussian_sigma,
            flow_algorithm,
            enable_refinement=enable_flow_refinement,
        )
        fb_mask: Optional[np.ndarray] = None
        fb_valid_ratio = 1.0
        if enable_forward_backward_check:
            fb_mask = compute_forward_backward_consistency_mask(
                frame1,
                frame2,
                gaussian_kernel_size=gaussian_kernel_size,
                gaussian_sigma=gaussian_sigma,
                flow_algorithm=flow_algorithm,
                enable_refinement=enable_flow_refinement,
                fb_threshold=forward_backward_threshold,
            )
            fb_valid_ratio = float(np.mean(fb_mask)) if fb_mask.size else 0.0

        geometry_evidence = {
            "valid": False,
            "quality": 0.0,
            "eac_score": None,
            "cubic_score": None,
            "inlier_ratio": 0.0,
            "reprojection_error": 999.0,
        }
        if enable_geometry_evidence:
            geometry_evidence = compute_global_geometry_evidence(frame1, frame2)

        region_bounds = split_into_regions(frame1)

        region_info: Dict[tuple, Dict[str, Any]] = {}
        valid_regions = 0
        invalid_regions = 0
        for row, col, y0, y1, x0, x1 in region_bounds:
            info = compute_region_motion(
                flow,
                (row, col, y0, y1, x0, x1),
                umbral_magnitud=umbral_magnitud,
                proporcion_minima_pixeles=proporcion_minima_pixeles,
                valid_mask=fb_mask,
            )
            region_info[(row, col)] = info
            if is_region_valid(info, min_concentration=umbral_concentracion, min_active_ratio=min_active_ratio):
                valid_regions += 1
            else:
                invalid_regions += 1

        pair_mean_magnitude = 0.0
        pair_mean_active_ratio = 0.0
        pair_motion_strength = 0.0
        if region_info:
            magnitudes = [float(v.get("magnitude", 0.0)) for v in region_info.values() if bool(v.get("valid"))]
            active_ratios = [float(v.get("active_ratio", 0.0)) for v in region_info.values()]
            pair_mean_magnitude = float(np.mean(magnitudes)) if magnitudes else 0.0
            pair_mean_active_ratio = float(np.mean(active_ratios)) if active_ratios else 0.0
            pair_motion_strength = pair_mean_magnitude * pair_mean_active_ratio

        affine_angles = compute_region_affine_angles(frame1, frame2, region_bounds)
        for row, col, _y0, _y1, _x0, _x1 in region_bounds:
            affine_angle = affine_angles.get((row, col))
            if affine_angle is not None and region_info.get((row, col), {}).get("valid", False):
                region_info[(row, col)]["angle"] = affine_angle

        eac_eval = evaluate_eac(
            region_info,
            min_concentration=umbral_concentracion,
            min_active_ratio=min_active_ratio,
            tolerancia_45_deg=tolerancia_45_deg,
        )
        cubic_eval = evaluate_cubemap(
            region_info,
            min_concentration=umbral_concentracion,
            min_active_ratio=min_active_ratio,
            tolerancia_45_deg=tolerancia_45_deg,
        )
        fused_eac_score = _fuse_optional_score(
            eac_eval["score"],
            geometry_evidence.get("eac_score"),
            geometry_evidence_weight if bool(geometry_evidence.get("valid")) else 0.0,
        )
        fused_cubic_score = _fuse_optional_score(
            cubic_eval["score"],
            geometry_evidence.get("cubic_score"),
            geometry_evidence_weight if bool(geometry_evidence.get("valid")) else 0.0,
        )

        decision = decide_projection(
            fused_eac_score,
            fused_cubic_score,
            min_margin=min_layout_score_margin,
        )

        status = "used" if decision is not None else "discarded"
        reason = "" if decision is not None else "insufficient_layout_data"
        return {
            "decision": decision,
            "score_eac": eac_eval["score"],
            "score_cubic": cubic_eval["score"],
            "score_eac_fused": fused_eac_score,
            "score_cubic_fused": fused_cubic_score,
            "valid_regions": valid_regions,
            "invalid_regions": invalid_regions,
            "pair_mean_magnitude": pair_mean_magnitude,
            "pair_mean_active_ratio": pair_mean_active_ratio,
            "pair_motion_strength": pair_motion_strength,
            "pair_fb_valid_ratio": fb_valid_ratio,
            "pair_geometry_quality": float(geometry_evidence.get("quality", 0.0)),
            "pair_geometry_inlier_ratio": float(geometry_evidence.get("inlier_ratio", 0.0)),
            "pair_geometry_reprojection_error": float(geometry_evidence.get("reprojection_error", 999.0)),
            "status": status,
            "reason": reason,
            "flow": flow,
            "region_bounds": region_bounds,
            "region_info": region_info,
        }
    except Exception:
        logger.exception("Error en análisis de movimiento de par")
        return {
            "decision": None,
            "score_eac": None,
            "score_cubic": None,
            "score_eac_fused": None,
            "score_cubic_fused": None,
            "valid_regions": 0,
            "invalid_regions": 0,
            "pair_mean_magnitude": 0.0,
            "pair_mean_active_ratio": 0.0,
            "pair_motion_strength": 0.0,
            "pair_fb_valid_ratio": 0.0,
            "pair_geometry_quality": 0.0,
            "pair_geometry_inlier_ratio": 0.0,
            "pair_geometry_reprojection_error": 999.0,
            "status": "discarded",
            "reason": "exception",
            "flow": None,
            "region_bounds": [],
            "region_info": {},
        }


def _classify_non_equirectangular(
    secuencias: List[List[np.ndarray]],
    labels_secuencia: Optional[List[str]] = None,
    motion_visualizations_dir: Optional[str] = None,
    min_valid_pairs: int = 4,
    min_motion_confidence: float = 0.2,
    min_layout_score_margin: float = 0.10,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
    min_active_ratio: float = 0.06,
    dominant_ratio_threshold: float = 0.8,
    ambiguity_gap: float = 0.10,
    flow_algorithm: str = "farneback",
    enable_flow_refinement: bool = False,
    enable_forward_backward_check: bool = False,
    forward_backward_threshold: float = 1.5,
    enable_geometry_evidence: bool = False,
    geometry_evidence_weight: float = 0.20,
    confidence_rollout_profile: str = "high_accuracy",
    seam_support_ratio: float = 1.0,
) -> Dict[str, Any]:
    def _build_pair_plan(seq_len: int, prioritize_long: bool = False) -> List[tuple[int, int, int]]:
        if seq_len < 2:
            return []
        max_gap = max(1, seq_len - 1)
        candidate_gaps = [1, min(2, max_gap), min(3, max_gap), max_gap]
        dedup_gaps: List[int] = []
        seen_gaps: Set[int] = set()
        for gap in candidate_gaps:
            if gap not in seen_gaps:
                dedup_gaps.append(gap)
                seen_gaps.add(gap)
        if prioritize_long:
            dedup_gaps = sorted(dedup_gaps, reverse=True)
        else:
            dedup_gaps = sorted(dedup_gaps)

        pairs: List[tuple[int, int, int]] = []
        seen_pairs: Set[tuple[int, int]] = set()
        for gap in dedup_gaps:
            for i in range(0, seq_len - gap):
                j = i + gap
                key = (i, j)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append((i, j, gap))
        return pairs

    def _pair_weight(pair_result: Dict[str, Any]) -> float:
        score_eac = pair_result.get("score_eac_fused")
        score_cubic = pair_result.get("score_cubic_fused")
        layout_margin = 0.0
        if score_eac is not None and score_cubic is not None:
            layout_margin = abs(float(score_eac) - float(score_cubic))
        motion_strength = float(pair_result.get("pair_motion_strength", 0.0))
        geometry_quality = float(pair_result.get("pair_geometry_quality", 0.0))
        motion_component = min(1.0, motion_strength / 0.10)
        gap = max(1, int(pair_result.get("pair_gap", 1)))
        gap_boost = min(1.35, 1.0 + (0.08 * (gap - 1)))
        weight = (
            0.30
            + (0.35 * motion_component)
            + (0.20 * min(1.0, layout_margin))
            + (0.15 * np.clip(geometry_quality, 0.0, 1.0))
        )
        return max(0.1, weight * gap_boost)

    def _is_pair_low_motion(pair_result: Dict[str, Any]) -> bool:
        pair_mag = float(pair_result.get("pair_mean_magnitude", 0.0))
        pair_active = float(pair_result.get("pair_mean_active_ratio", 0.0))
        return pair_mag < 1.15 or pair_active < 0.05

    pares_totales = 0
    pares_validos = 0
    pares_eac = 0
    pares_cubic = 0
    total_regiones_validas = 0
    total_regiones_invalidas = 0
    eac_scores_list: List[float] = []
    cubic_scores_list: List[float] = []
    weighted_eac = 0.0
    weighted_cubic = 0.0
    valid_pair_motion_magnitudes: List[float] = []
    valid_pair_active_ratios: List[float] = []
    pair_geometry_qualities: List[float] = []
    pair_layout_margins: List[float] = []
    pair_decisions: List[bool] = []
    low_motion_pairs = 0
    pair_gaps_used: List[int] = []
    long_gap_weighted_eac = 0.0
    long_gap_weighted_cubic = 0.0
    geometry_vote_eac = 0.0
    geometry_vote_cubic = 0.0

    for seq_idx, secuencia in enumerate(secuencias):
        seq_label = f"seq_{seq_idx:03d}"
        if labels_secuencia and seq_idx < len(labels_secuencia):
            seq_label = labels_secuencia[seq_idx]
        baseline_pairs = _build_pair_plan(len(secuencia), prioritize_long=False)
        short_pairs = [p for p in baseline_pairs if p[2] == 1]
        extended_pairs = [p for p in baseline_pairs if p[2] != 1]

        prelim_weighted_eac = 0.0
        prelim_weighted_cubic = 0.0
        short_motion_signals: List[float] = []
        short_pair_cache: Dict[tuple[int, int], Dict[str, Any]] = {}

        candidate_pairs: List[tuple[int, int, int]] = []
        if short_pairs:
            candidate_pairs.extend(short_pairs)
            for i, j, _gap in short_pairs:
                quick_pair = _analyze_motion_pair_detailed(
                    secuencia[i],
                    secuencia[j],
                    gaussian_kernel_size=gaussian_kernel_size,
                    gaussian_sigma=gaussian_sigma,
                    min_active_ratio=min_active_ratio,
                    min_layout_score_margin=min_layout_score_margin,
                    flow_algorithm=flow_algorithm,
                    enable_flow_refinement=enable_flow_refinement,
                    enable_forward_backward_check=enable_forward_backward_check,
                    forward_backward_threshold=forward_backward_threshold,
                    enable_geometry_evidence=enable_geometry_evidence,
                    geometry_evidence_weight=geometry_evidence_weight,
                )
                short_pair_cache[(i, j)] = quick_pair
                if quick_pair.get("decision") is None:
                    continue
                short_motion_signals.append(float(quick_pair.get("pair_motion_strength", 0.0)))
                w = _pair_weight(quick_pair)
                if bool(quick_pair.get("decision")):
                    prelim_weighted_eac += w
                else:
                    prelim_weighted_cubic += w

        prelim_total_weight = prelim_weighted_eac + prelim_weighted_cubic
        prelim_confidence = (
            abs(prelim_weighted_eac - prelim_weighted_cubic) / prelim_total_weight
            if prelim_total_weight > 0
            else 0.0
        )
        prelim_mean_signal = float(np.mean(short_motion_signals)) if short_motion_signals else 0.0
        prioritize_long = prelim_confidence < 0.35 or prelim_mean_signal < 0.035
        if prioritize_long:
            extended_pairs = sorted(extended_pairs, key=lambda pair: pair[2], reverse=True)
        candidate_pairs.extend(extended_pairs)

        seen_candidates: Set[tuple[int, int]] = set()
        for i, j, gap in candidate_pairs:
            if (i, j) in seen_candidates:
                continue
            seen_candidates.add((i, j))
            pares_totales += 1
            pair_label = f"{seq_label}_pair_{i+1:02d}_{j+1:02d}_gap{gap:02d}"
            logger.info(
                f"[PAIR][{pair_label}] START "
                f"(secondary_frames={i+1:02d}->{j+1:02d}, gap={gap})"
            )

            pair_result = short_pair_cache.get((i, j))
            if pair_result is None:
                pair_result = _analyze_motion_pair_detailed(
                    secuencia[i],
                    secuencia[j],
                    gaussian_kernel_size=gaussian_kernel_size,
                    gaussian_sigma=gaussian_sigma,
                    min_active_ratio=min_active_ratio,
                    min_layout_score_margin=min_layout_score_margin,
                    flow_algorithm=flow_algorithm,
                    enable_flow_refinement=enable_flow_refinement,
                    enable_forward_backward_check=enable_forward_backward_check,
                    forward_backward_threshold=forward_backward_threshold,
                    enable_geometry_evidence=enable_geometry_evidence,
                    geometry_evidence_weight=geometry_evidence_weight,
                )

            total_regiones_validas += int(pair_result.get("valid_regions", 0))
            total_regiones_invalidas += int(pair_result.get("invalid_regions", 0))

            if motion_visualizations_dir:
                status_name: str
                if pair_result.get("decision") is None:
                    reason = str(pair_result.get("reason", "invalid"))
                    status_name = f"motion_{pair_label}_discarded_{reason}.jpg"
                else:
                    status_name = f"motion_{pair_label}_used_{'eac' if pair_result.get('decision') else 'cubic'}.jpg"
                vis_path = os.path.join(motion_visualizations_dir, status_name)
                vis = secuencia[i + 1].copy()
                flow = pair_result.get("flow")
                region_bounds = pair_result.get("region_bounds", [])
                region_info = pair_result.get("region_info", {})
                if flow is not None:
                    vis = draw_regions(vis, region_bounds)
                    vis = draw_motion_vectors(vis, flow, umbral_magnitud=1.0, step=28)

                    arrow_len = 45
                    for row, col, y0, y1, x0, x1 in region_bounds:
                        key = (row, col)
                        info = region_info.get(key)
                        cx = (x0 + x1) // 2
                        cy = (y0 + y1) // 2
                        if not info or not info.get("valid"):
                            cv2.circle(vis, (cx, cy), 6, (120, 120, 120), -1)
                            cv2.putText(
                                vis,
                                "NO MOTION",
                                (x0 + 4, y0 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (120, 120, 120),
                                1,
                            )
                            continue

                        valid = is_region_valid(info, min_concentration=0.25, min_active_ratio=0.06)
                        color = (255, 0, 0) if valid else (0, 80, 200)
                        a = float(info["angle"])
                        ex = int(cx + arrow_len * np.cos(a))
                        ey = int(cy + arrow_len * np.sin(a))
                        cv2.arrowedLine(vis, (cx, cy), (ex, ey), color, 2, tipLength=0.25)
                        label = f"{np.degrees(a):.0f} c={float(info['concentration']):.2f}"
                        if not valid:
                            label += " INV"
                        cv2.putText(
                            vis,
                            label,
                            (x0 + 4, y0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                        )

                score_eac = pair_result.get("score_eac_fused")
                score_cubic = pair_result.get("score_cubic_fused")
                eac_score_str = f"{score_eac:.3f}" if score_eac is not None else "N/A"
                cubic_score_str = f"{score_cubic:.3f}" if score_cubic is not None else "N/A"
                winner = "EAC" if pair_result.get("decision") else "CUBIC"
                cv2.putText(vis, f"EAC   score={eac_score_str}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f"CUBIC score={cubic_score_str}", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(vis, f"Pattern: {winner}", (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                save_frame_debug(vis_path, vis, "MOTION")
                logger.info(f"[PAIR][{pair_label}] output_file={vis_path}")

            if pair_result.get("decision") is None:
                continue

            pares_validos += 1
            pair_gaps_used.append(gap)
            pair_result["pair_gap"] = gap
            if bool(pair_result.get("decision")):
                pares_eac += 1
            else:
                pares_cubic += 1
            weight = _pair_weight(pair_result)
            if bool(pair_result.get("decision")):
                weighted_eac += weight
            else:
                weighted_cubic += weight
            if gap >= 2:
                if bool(pair_result.get("decision")):
                    long_gap_weighted_eac += weight
                else:
                    long_gap_weighted_cubic += weight
            pair_decisions.append(bool(pair_result.get("decision")))

            pair_mag = float(pair_result.get("pair_mean_magnitude", 0.0))
            pair_active = float(pair_result.get("pair_mean_active_ratio", 0.0))
            valid_pair_motion_magnitudes.append(pair_mag)
            valid_pair_active_ratios.append(pair_active)
            pair_geometry_qualities.append(float(pair_result.get("pair_geometry_quality", 0.0)))
            score_eac_fused = pair_result.get("score_eac_fused")
            score_cubic_fused = pair_result.get("score_cubic_fused")
            if score_eac_fused is not None and score_cubic_fused is not None:
                fused_margin = abs(float(score_eac_fused) - float(score_cubic_fused))
                pair_layout_margins.append(fused_margin)
                geometry_quality = float(pair_result.get("pair_geometry_quality", 0.0))
                geometry_vote = geometry_quality * min(1.0, fused_margin * 2.5)
                if float(score_eac_fused) >= float(score_cubic_fused):
                    geometry_vote_eac += geometry_vote
                else:
                    geometry_vote_cubic += geometry_vote
            if _is_pair_low_motion(pair_result):
                low_motion_pairs += 1
            if pair_result.get("score_eac") is not None:
                eac_scores_list.append(float(pair_result["score_eac"]))
            if pair_result.get("score_cubic") is not None:
                cubic_scores_list.append(float(pair_result["score_cubic"]))
            log_success(
                f"[PAIR][{pair_label}] -> USED "
                f"(valid_regions={pair_result.get('valid_regions', 0)}, "
                f"invalid_regions={pair_result.get('invalid_regions', 0)}, "
                f"decision={'EAC' if pair_result.get('decision') else 'CUBIC'}, "
                f"mean_mag={pair_mag:.3f}, active_ratio={pair_active:.3f}, weight={weight:.3f})"
            )

    avg_eac_score = float(np.mean(eac_scores_list)) if eac_scores_list else None
    avg_cubic_score = float(np.mean(cubic_scores_list)) if cubic_scores_list else None
    score_margin = 0.0
    weighted_total = weighted_eac + weighted_cubic
    ratio_eac = (weighted_eac / weighted_total) if weighted_total > 0 else 0.0
    ratio_cubic = (weighted_cubic / weighted_total) if weighted_total > 0 else 0.0
    long_gap_total = long_gap_weighted_eac + long_gap_weighted_cubic
    long_gap_ratio_eac = (long_gap_weighted_eac / long_gap_total) if long_gap_total > 0 else 0.0
    long_gap_ratio_cubic = (long_gap_weighted_cubic / long_gap_total) if long_gap_total > 0 else 0.0
    mean_pair_magnitude = float(np.mean(valid_pair_motion_magnitudes)) if valid_pair_motion_magnitudes else 0.0
    mean_pair_active_ratio = float(np.mean(valid_pair_active_ratios)) if valid_pair_active_ratios else 0.0
    low_motion_ratio = (low_motion_pairs / pares_validos) if pares_validos > 0 else 0.0
    low_motion_detected = bool(
        pares_validos > 0
        and (
            low_motion_ratio >= 0.60
            or mean_pair_active_ratio < 0.055
            or mean_pair_magnitude < 1.15
        )
    )

    if pares_validos == 0:
        log_discard("Análisis de movimiento: sin pares válidos. Clasificación conservadora => UNKNOWN.")
        return {
            "classification": "unknown",
            "ratio_eac": 0.0,
            "pares_totales": pares_totales,
            "pares_validos": 0,
            "pares_eac": 0,
            "pares_invalidos": pares_totales,
            "total_regiones_validas": total_regiones_validas,
            "total_regiones_invalidas": total_regiones_invalidas,
            "avg_eac_score": avg_eac_score,
            "avg_cubic_score": avg_cubic_score,
            "score_margin": score_margin,
            "motion_confidence": 0.0,
            "reliable": False,
            "reliability_reason": "no_valid_pairs",
            "motion_low_detected": False,
            "motion_low_ratio": 0.0,
            "motion_mean_pair_magnitude": 0.0,
            "motion_mean_pair_active_ratio": 0.0,
            "motion_pair_gaps_used": [],
            "weighted_ratio_eac": 0.0,
            "pair_geometry_mean_quality": 0.0,
        }

    effective_min_valid_pairs = int(min_valid_pairs)
    effective_min_motion_confidence = float(min_motion_confidence)
    effective_dominant_ratio_threshold = float(dominant_ratio_threshold)
    if low_motion_detected:
        effective_min_valid_pairs = max(2, int(min_valid_pairs) - 1)
        effective_min_motion_confidence = max(0.10, float(min_motion_confidence) * 0.60)
        effective_dominant_ratio_threshold = max(0.65, float(dominant_ratio_threshold) - 0.10)
        logger.info(
            "[LOW-MOTION] active (ratio=%.0f%% mean_mag=%.3f mean_active=%.3f gaps=%s)",
            low_motion_ratio * 100.0,
            mean_pair_magnitude,
            mean_pair_active_ratio,
            sorted(set(pair_gaps_used)),
        )

    def _decision_persistence(decisions: List[bool], target: bool) -> float:
        if not decisions:
            return 0.0
        best = 0
        current = 0
        for decision in decisions:
            if bool(decision) == bool(target):
                current += 1
                best = max(best, current)
            else:
                current = 0
        return float(best / max(1, len(decisions)))

    persistence_eac = _decision_persistence(pair_decisions, True)
    persistence_cubic = _decision_persistence(pair_decisions, False)
    seam_confidence = float(np.clip((float(seam_support_ratio) - 0.45) / 0.45, 0.0, 1.0))
    geometry_vote_total = geometry_vote_eac + geometry_vote_cubic

    if ratio_eac >= effective_dominant_ratio_threshold:
        clasificacion = "eac"
    elif ratio_cubic >= effective_dominant_ratio_threshold:
        clasificacion = "cubic"
    else:
        ratio_gap = abs(ratio_eac - ratio_cubic)
        if ratio_gap <= float(ambiguity_gap):
            geometry_gap = abs(geometry_vote_eac - geometry_vote_cubic)
            long_gap_ratio_gap = abs(long_gap_ratio_eac - long_gap_ratio_cubic)
            if geometry_vote_total >= 0.20 and geometry_gap >= 0.10:
                clasificacion = "eac" if geometry_vote_eac >= geometry_vote_cubic else "cubic"
            elif long_gap_total >= 0.20 and long_gap_ratio_gap >= 0.12:
                clasificacion = "eac" if long_gap_ratio_eac >= long_gap_ratio_cubic else "cubic"
            elif max(persistence_eac, persistence_cubic) >= 0.60:
                clasificacion = "eac" if persistence_eac >= persistence_cubic else "cubic"
            else:
                clasificacion = "unknown"
        else:
            clasificacion = "eac" if ratio_eac > ratio_cubic else "cubic"

    base_confidence = abs(ratio_eac - 0.5) * 2.0
    layout_confidence = float(np.clip(np.mean(pair_layout_margins) / 0.35, 0.0, 1.0)) if pair_layout_margins else 0.0
    motion_signal_confidence = float(np.clip((mean_pair_magnitude * mean_pair_active_ratio) / 0.10, 0.0, 1.0))
    geometry_confidence = float(np.clip(np.mean(pair_geometry_qualities), 0.0, 1.0)) if pair_geometry_qualities else 0.0
    profile_key = str(confidence_rollout_profile or "baseline").strip().lower()
    if profile_key == "high_accuracy":
        motion_confidence = (
            0.32 * base_confidence
            + 0.20 * layout_confidence
            + 0.15 * motion_signal_confidence
            + 0.23 * geometry_confidence
            + 0.10 * seam_confidence
        )
    elif profile_key == "robust":
        motion_confidence = (
            0.40 * base_confidence
            + 0.25 * layout_confidence
            + 0.20 * motion_signal_confidence
            + 0.10 * geometry_confidence
            + 0.05 * seam_confidence
        )
    else:
        motion_confidence = (
            0.60 * base_confidence
            + 0.20 * layout_confidence
            + 0.15 * motion_signal_confidence
            + 0.05 * seam_confidence
        )
    score_margin = abs(ratio_eac - ratio_cubic)

    reliability_reasons: List[str] = []
    if pares_validos < effective_min_valid_pairs:
        reliability_reasons.append(f"few_valid_pairs:{pares_validos}<{effective_min_valid_pairs}")
    if motion_confidence < effective_min_motion_confidence:
        reliability_reasons.append(
            f"low_motion_confidence:{motion_confidence:.2f}<{effective_min_motion_confidence:.2f}"
        )
    if score_margin < float(min_layout_score_margin):
        reliability_reasons.append(f"low_layout_margin:{score_margin:.3f}<{min_layout_score_margin:.3f}")
    dominant_ratio = max(ratio_eac, ratio_cubic)
    if dominant_ratio < effective_dominant_ratio_threshold:
        reliability_reasons.append(
            f"weak_dominance:{dominant_ratio:.2f}<{effective_dominant_ratio_threshold:.2f}"
        )
    if low_motion_detected:
        reliability_reasons.append(
            f"low_motion_detected:ratio={low_motion_ratio:.2f},mag={mean_pair_magnitude:.3f},active={mean_pair_active_ratio:.3f}"
        )

    reliable = len([r for r in reliability_reasons if not r.startswith("low_motion_detected:")]) == 0
    if not reliable and clasificacion != "unknown":
        contradictory_evidence = bool(
            score_margin <= max(0.03, float(ambiguity_gap) * 0.50)
            and abs(long_gap_ratio_eac - long_gap_ratio_cubic) < 0.10
            and abs(geometry_vote_eac - geometry_vote_cubic) < 0.08
            and max(persistence_eac, persistence_cubic) < 0.58
        )
        strong_consistency = bool(
            max(persistence_eac, persistence_cubic) >= 0.65
            or abs(long_gap_ratio_eac - long_gap_ratio_cubic) >= 0.18
            or abs(geometry_vote_eac - geometry_vote_cubic) >= 0.14
            or seam_confidence >= 0.80
        )
        if contradictory_evidence or not strong_consistency:
            clasificacion = "unknown"
        else:
            reliable = True
            reliability_reasons.append("forced_by_consistent_evidence")

    logger.info(f"[EVAL] pairs={pares_validos} eac={pares_eac} cubic={pares_cubic}")
    logger.info(
        f"[EVAL] ratios -> eac={ratio_eac:.1%} cubic={ratio_cubic:.1%} "
        f"decision={clasificacion.upper()} reliable={reliable}"
    )

    return {
        "classification": clasificacion,
        "ratio_eac": ratio_eac,
        "ratio_cubic": ratio_cubic,
        "pares_totales": pares_totales,
        "pares_validos": pares_validos,
        "pares_eac": pares_eac,
        "pares_cubic": pares_cubic,
        "pares_invalidos": pares_totales - pares_validos,
        "total_regiones_validas": total_regiones_validas,
        "total_regiones_invalidas": total_regiones_invalidas,
        "avg_eac_score": avg_eac_score,
        "avg_cubic_score": avg_cubic_score,
        "score_margin": score_margin,
        "motion_confidence": motion_confidence,
        "reliable": reliable,
        "reliability_reason": ";".join(reliability_reasons) if reliability_reasons else "",
        "motion_low_detected": low_motion_detected,
        "motion_low_ratio": low_motion_ratio,
        "motion_mean_pair_magnitude": mean_pair_magnitude,
        "motion_mean_pair_active_ratio": mean_pair_active_ratio,
        "motion_pair_gaps_used": sorted(set(pair_gaps_used)),
        "weighted_ratio_eac": ratio_eac,
        "pair_geometry_mean_quality": geometry_confidence,
        "pair_geometry_vote_gap": abs(geometry_vote_eac - geometry_vote_cubic),
        "pair_long_gap_ratio_gap": abs(long_gap_ratio_eac - long_gap_ratio_cubic),
        "pair_persistence_eac": persistence_eac,
        "pair_persistence_cubic": persistence_cubic,
        "seam_confidence": seam_confidence,
    }


def run_detection_pipeline(
    video_path: str,
    num_frames: int = 10,
    frames: Optional[List[np.ndarray]] = None,
    frames_metadata: Optional[List[Dict[str, Any]]] = None,
    video_name: Optional[str] = None,
    debug_context: Optional[Dict[str, str]] = None,
    paso_frames_secundarios: int = 5,
    min_frames_with_line_required: int = 7,
    min_valid_pairs: int = 4,
    min_motion_confidence: float = 0.2,
    min_frames_analyzed_required: Optional[int] = None,
    max_discard_ratio: Optional[float] = None,
    min_layout_score_margin: Optional[float] = None,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
    flow_algorithm: str = "farneback",
    enable_flow_refinement: bool = False,
    enable_forward_backward_check: bool = False,
    forward_backward_threshold: float = 1.5,
    enable_geometry_evidence: bool = False,
    geometry_evidence_weight: float = 0.20,
    confidence_rollout_profile: str = "high_accuracy",
) -> Dict[str, Any]:
    try:
        logger.info(f"Detectando tipo de proyección para: {video_path}")

        if min_frames_analyzed_required is None:
            min_frames_analyzed_required = int(CONFIG["min_frames_analyzed_required"])
        if max_discard_ratio is None:
            max_discard_ratio = float(CONFIG["max_discard_ratio"])
        if min_layout_score_margin is None:
            min_layout_score_margin = float(CONFIG["min_layout_score_margin"])

        stereo_hist_threshold = float(CONFIG["stereo_hist_similarity_threshold"])
        save_stereo_halves_enabled = bool(CONFIG["save_stereo_halves"])

        total_frames_video = 0
        video_path_procesado = video_path
        if frames is None:
            extraction_result = extract_main_frames(
                video_path,
                num_frames=num_frames,
                modo_extraccion="equiespaciados",
                guardar_frames=False,
                save_image_fn=save_frame_debug,
                log_success_fn=log_success,
                log_discard_fn=log_discard,
            )
            frames = extraction_result.get("frames", [])
            frames_metadata = extraction_result.get("frames_metadata", [])
            total_frames_video = extraction_result.get("total_frames", 0)
            video_path_procesado = extraction_result.get("video_path_procesado", video_path)
            video_name = extraction_result.get("video_name", "video")
        else:
            logger.info(f"Usando {len(frames)} frames pre-extraídos")
            if not video_name:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
            cap_meta = cv2.VideoCapture(video_path)
            if cap_meta.isOpened():
                total_frames_video = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_meta.release()

        if not frames:
            return {
                "projection_type": "unknown",
                "confidence": 0.0,
                "frames_analyzed": 0,
                "frames_with_line": 0,
                "stats": {
                    "total_frames_extracted": 0,
                    "black_frames": 0,
                    "valid_frames_used": 0,
                    "discarded_frames": 0,
                    "pairs_total": 0,
                    "pairs_valid": 0,
                    "pairs_invalid": 0,
                    "regions_valid": 0,
                    "regions_invalid": 0,
                    "avg_eac_score": "N/A",
                    "avg_cubic_score": "N/A",
                    "score_margin": "N/A",
                    "final_classification": "unknown",
                    "final_confidence": 0.0,
                },
                "error": "No se pudieron extraer frames",
            }

        horizontal_line_dir = CONFIG["frames_output_dir"]
        secondary_sequences_dir = None
        motion_visualizations_dir = None
        video_tag = video_name
        if debug_context:
            horizontal_line_dir = debug_context.get("horizontal_line_dir", horizontal_line_dir)
            secondary_sequences_dir = debug_context.get("secondary_sequences_dir")
            motion_visualizations_dir = debug_context.get("motion_visualizations_dir")
            video_tag = debug_context.get("video_tag", video_name)

        os.makedirs(horizontal_line_dir, exist_ok=True)

        positions_principales = [m["position"] for m in frames_metadata] if frames_metadata else []
        secuencias_secundarias: List[List[np.ndarray]] = []
        if positions_principales and total_frames_video > 0:
            secondary_raw = extract_secondary_frames(
                video_path_procesado,
                positions_principales,
                total_frames_video,
                paso_frames=paso_frames_secundarios,
            )
            for main_idx, seq_raw in enumerate(secondary_raw, start=1):
                seq: List[np.ndarray] = []
                seq_group_dir = None
                if secondary_sequences_dir:
                    seq_group_dir = os.path.join(secondary_sequences_dir, f"main_{main_idx:03d}")
                    os.makedirs(seq_group_dir, exist_ok=True)

                for seq_idx, item in enumerate(seq_raw, start=1):
                    p = int(item["position"])
                    frame_obj = item["frame"]
                    if item.get("valid") and frame_obj is not None:
                        seq.append(frame_obj)
                        log_success(
                            f"[SECONDARY][main={main_idx:03d}][seq={seq_idx:02d}][video_frame={p:06d}] -> USED "
                            f"(captured for motion analysis)"
                        )
                        if seq_group_dir:
                            filename = f"secondary_frame_main{main_idx:03d}_seq_{seq_idx:02d}_video_{p:06d}_used.jpg"
                            save_frame_debug(os.path.join(seq_group_dir, filename), frame_obj, "SECONDARY")
                    else:
                        log_discard(
                            f"[SECONDARY][main={main_idx:03d}][seq={seq_idx:02d}][video_frame={p:06d}] -> DISCARDED (read_failed)"
                        )
                secuencias_secundarias.append(seq)
        else:
            if not positions_principales:
                logger.warning("Sin metadatos de posición: análisis de movimiento secundario omitido.")

        total_frames_extraidos = len(frames)
        frames_analyzed = 0
        frames_with_line = 0
        black_frames = 0
        discarded_no_line = 0
        frames_guardados: List[str] = []
        indices_con_linea: List[int] = []
        indices_con_linea_vertical: List[int] = []  # frames with vertical line only (LR stereo candidate)
        equi_frame_results: List[Dict[str, Any]] = []

        logger.info(f"\nAnalizando {len(frames)} frames con detección de línea horizontal...\n")
        for idx, frame in enumerate(frames):
            global_pos = None
            if frames_metadata and idx < len(frames_metadata):
                global_pos = frames_metadata[idx].get("position")
            pos_tag = f"{int(global_pos):06d}" if global_pos is not None else "unknown"

            if frame_esta_en_negro(frame):
                black_frames += 1
                log_discard(f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] -> DISCARDED (black_frame)")
                discard_path = os.path.join(horizontal_line_dir, f"main_frame_{idx+1:03d}_video_{pos_tag}_discarded_black.jpg")
                save_frame_debug(discard_path, frame, "MAIN")
                continue

            equi_frame_results.append(compute_frame_equirectangular_evidence(frame))

            line_result = detect_horizontal_line(
                frame,
                center_tolerance_ratio=CONFIG["line_center_max_distance_ratio"],
                band_ratio=CONFIG["line_center_band_ratio"],
                max_slope=CONFIG["line_max_slope"],
                min_coverage_ratio=CONFIG["line_min_coverage_ratio"],
            )
            frames_analyzed += 1

            save_line_visual_debug(
                frame=frame,
                frame_idx=idx + 1,
                output_dir=horizontal_line_dir,
                debug_line_info=line_result.get("debug_line_info", {}),
                found=bool(line_result.get("has_horizontal_line")),
            )

            if line_result.get("has_horizontal_line"):
                fft_confirmed = bool(line_result.get("fft_confirmed", False))
                fft_confidence = float(line_result.get("fft_confidence", 0.0))
                logger.info(
                    f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] "
                    f"line_fft_confirmed={fft_confirmed} fft_confidence={fft_confidence:.3f}"
                )
                frames_with_line += 1
                indices_con_linea.append(idx)
                log_success(f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] -> USED (horizontal_line_detected)")
                line_data = line_result.get("line_data")
                if line_data:
                    if global_pos is not None:
                        line_data["video_position"] = int(global_pos)
                    fp = save_line_detected_frame(frame, line_data, idx + 1, horizontal_line_dir)
                    if fp:
                        frames_guardados.append(fp)
            else:
                # Horizontal line not found — try vertical line for left-right stereo candidate
                logger.debug(
                    f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] "
                    "horizontal line not detected, attempting vertical line detection"
                )
                vert_result = detect_vertical_line(
                    frame,
                    center_tolerance_ratio=CONFIG["line_center_max_distance_ratio"],
                    band_ratio=CONFIG["line_center_band_ratio"],
                    max_slope=CONFIG["line_max_slope"],
                    min_coverage_ratio=CONFIG["line_min_coverage_ratio"],
                )
                save_line_visual_debug(
                    frame=frame,
                    frame_idx=idx + 1,
                    output_dir=horizontal_line_dir,
                    debug_line_info=vert_result.get("debug_line_info", {}),
                    found=bool(vert_result.get("has_vertical_line")),
                    line_orientation="vertical",
                )
                if vert_result.get("has_vertical_line"):
                    fft_confirmed_v = bool(vert_result.get("fft_confirmed", False))
                    fft_confidence_v = float(vert_result.get("fft_confidence", 0.0))
                    logger.info(
                        f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] "
                        f"vertical_line_detected fft_confirmed={fft_confirmed_v} fft_confidence={fft_confidence_v:.3f}"
                    )
                    indices_con_linea_vertical.append(idx)
                    log_success(
                        f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] -> VERTICAL LINE DETECTED (lr_stereo_candidate)"
                    )
                    vertical_line_data = vert_result.get("line_data")
                    if vertical_line_data:
                        if global_pos is not None:
                            vertical_line_data["video_position"] = int(global_pos)
                        fp = save_line_detected_frame(frame, vertical_line_data, idx + 1, horizontal_line_dir)
                        if fp:
                            frames_guardados.append(fp)
                else:
                    discarded_no_line += 1
                    log_discard(f"[MAIN][idx={idx+1:02d}][video_frame={pos_tag}] -> DISCARDED (no_horizontal_line)")
                    discard_path = os.path.join(horizontal_line_dir, f"line_main_{idx+1:03d}_video_{pos_tag}_not_detected.jpg")
                    save_frame_debug(discard_path, frame, "LINE")

        equi_agg = aggregate_equirectangular_evidence(equi_frame_results)

        projection_type = "unknown"
        confidence = 0.0
        motion_reliable = False
        motion_reliability_reason = "motion_not_executed"
        motion_pairs_total = 0
        motion_pairs_valid = 0
        motion_pairs_invalid = 0
        motion_confidence = 0.0
        motion_avg_eac_score: Optional[float] = None
        motion_avg_cubic_score: Optional[float] = None
        motion_score_margin = 0.0
        motion_regions_valid = 0
        motion_regions_invalid = 0
        motion_low_detected = False
        motion_low_ratio = 0.0
        motion_mean_pair_magnitude = 0.0
        motion_mean_pair_active_ratio = 0.0
        motion_pair_gaps_used: List[int] = []
        motion_pair_geometry_quality = 0.0
        stereo_avg_similarity = 0.0

        if frames_analyzed > 0:
            ratio = frames_with_line / frames_analyzed
            if frames_with_line == 0:
                # No horizontal lines found — check for left-right stereo before falling back
                if indices_con_linea_vertical:
                    stereo_lr_result = detect_stereo(
                        frames=frames,
                        indices_con_linea=indices_con_linea_vertical,
                        similarity_threshold=stereo_hist_threshold,
                        arrangement="left-right",
                    )
                    stereo_avg_similarity = float(stereo_lr_result["avg_similarity"])
                    for detail in stereo_lr_result.get("frame_details", []):
                        match_label = "MATCH" if detail["match"] else "NO MATCH"
                        logger.info(
                            f"[STEREO-LR] Frame {int(detail['frame_idx'])+1:02d} -> "
                            f"corr={float(detail['corr']):.4f} bhatt={float(detail['bhatt']):.4f} -> {match_label}"
                        )
                        if save_stereo_halves_enabled:
                            save_stereo_halves(
                                int(detail["frame_idx"]),
                                detail["left_half"],
                                detail["right_half"],
                                horizontal_line_dir,
                            )
                    logger.info(
                        f"[STEREO-LR] frames_evaluados={stereo_lr_result['frames_evaluados']} "
                        f"matches={stereo_lr_result['frames_match']} no_matches={stereo_lr_result['frames_no_match']} "
                        f"match_ratio={float(stereo_lr_result['match_ratio']):.0%} required={float(stereo_lr_result['min_match_ratio']):.0%} "
                        f"avg_corr={float(stereo_lr_result['avg_similarity']):.4f} avg_bhatt={float(stereo_lr_result['avg_bhattacharyya']):.4f} "
                        f"thresholds=(corr>={stereo_hist_threshold} bhatt<={float(stereo_lr_result['bhattacharyya_threshold'])}) "
                        f"decision={'stereo_equi_lr' if stereo_lr_result['is_stereo'] else 'not_stereo_lr'}"
                    )
                    if stereo_lr_result["is_stereo"]:
                        projection_type = "stereo_equi"
                        confidence = max(
                            len(indices_con_linea_vertical) / frames_analyzed,
                            stereo_avg_similarity,
                        )
                        motion_reliable = True
                        motion_reliability_reason = "early_stereo_equi_lr_histogram_match"
                        log_success(
                            "Clasificación temprana: STEREO_EQUI (left-right) "
                            f"(hist_similarity={stereo_avg_similarity:.4f}, threshold={stereo_hist_threshold:.4f})."
                        )
                    else:
                        projection_type = "equirectangular"
                        confidence = equi_agg["confidence"] if equi_agg["is_strong_evidence"] else 0.75
                        motion_reliable = True
                        motion_reliability_reason = "early_equirectangular_lr_stereo_failed"
                        log_success(
                            "Clasificación temprana: EQUIRECTANGULAR "
                            "(no horizontal line; left-right stereo test did not pass)."
                        )
                else:
                    projection_type = "equirectangular"
                    confidence = equi_agg["confidence"] if equi_agg["is_strong_evidence"] else 0.75
                    motion_reliable = True
                    motion_reliability_reason = "early_equirectangular_no_horizontal_line"
                    log_success("Clasificación temprana: EQUIRECTANGULAR (no se detectó línea horizontal en frames analizados).")
            else:
                stereo_result = detect_stereo(
                    frames=frames,
                    indices_con_linea=indices_con_linea,
                    similarity_threshold=stereo_hist_threshold,
                )
                stereo_avg_similarity = float(stereo_result["avg_similarity"])
                for detail in stereo_result.get("frame_details", []):
                    match_label = "MATCH" if detail["match"] else "NO MATCH"
                    logger.info(
                        f"[STEREO] Frame {int(detail['frame_idx'])+1:02d} -> "
                        f"corr={float(detail['corr']):.4f} bhatt={float(detail['bhatt']):.4f} -> {match_label}"
                    )
                    if save_stereo_halves_enabled:
                        save_stereo_halves(
                            int(detail["frame_idx"]),
                            detail["top_half"],
                            detail["bottom_half"],
                            horizontal_line_dir,
                        )

                logger.info(
                    f"[STEREO] frames_evaluados={stereo_result['frames_evaluados']} "
                    f"matches={stereo_result['frames_match']} no_matches={stereo_result['frames_no_match']} "
                    f"match_ratio={float(stereo_result['match_ratio']):.0%} required={float(stereo_result['min_match_ratio']):.0%} "
                    f"avg_corr={float(stereo_result['avg_similarity']):.4f} avg_bhatt={float(stereo_result['avg_bhattacharyya']):.4f} "
                    f"thresholds=(corr>={stereo_hist_threshold} bhatt<={float(stereo_result['bhattacharyya_threshold'])}) "
                    f"decision={'stereo_equi' if stereo_result['is_stereo'] else 'not_stereo'}"
                )

                if stereo_result["is_stereo"]:
                    projection_type = "stereo_equi"
                    confidence = max(ratio, stereo_avg_similarity)
                    motion_reliable = True
                    motion_reliability_reason = "early_stereo_equi_histogram_match"
                    log_success(
                        "Clasificación temprana: STEREO_EQUI "
                        f"(hist_similarity={stereo_avg_similarity:.4f}, threshold={stereo_hist_threshold:.4f})."
                    )
                elif frames_with_line >= min_frames_with_line_required:
                    if secuencias_secundarias:
                        secuencias_con_linea = [
                            secuencias_secundarias[i]
                            for i in indices_con_linea
                            if i < len(secuencias_secundarias)
                        ]
                        labels = [
                            f"main{i+1:03d}"
                            for i in indices_con_linea
                            if i < len(secuencias_secundarias)
                        ]
                        motion_result = _classify_non_equirectangular(
                            secuencias_con_linea,
                            labels_secuencia=labels,
                            motion_visualizations_dir=motion_visualizations_dir,
                            min_valid_pairs=min_valid_pairs,
                            min_motion_confidence=min_motion_confidence,
                            min_layout_score_margin=min_layout_score_margin,
                            gaussian_kernel_size=gaussian_kernel_size,
                            gaussian_sigma=gaussian_sigma,
                            flow_algorithm=flow_algorithm,
                            enable_flow_refinement=enable_flow_refinement,
                            enable_forward_backward_check=enable_forward_backward_check,
                            forward_backward_threshold=forward_backward_threshold,
                            enable_geometry_evidence=enable_geometry_evidence,
                            geometry_evidence_weight=geometry_evidence_weight,
                            confidence_rollout_profile=confidence_rollout_profile,
                            seam_support_ratio=ratio,
                        )
                        projection_type = motion_result["classification"]
                        motion_reliable = bool(motion_result["reliable"])
                        motion_reliability_reason = str(motion_result["reliability_reason"])
                        motion_pairs_total = int(motion_result.get("pares_totales", 0))
                        motion_pairs_valid = int(motion_result.get("pares_validos", 0))
                        motion_pairs_invalid = int(motion_result.get("pares_invalidos", 0))
                        motion_confidence = float(motion_result.get("motion_confidence", 0.0))
                        motion_avg_eac_score = motion_result.get("avg_eac_score")
                        motion_avg_cubic_score = motion_result.get("avg_cubic_score")
                        motion_score_margin = float(motion_result.get("score_margin", 0.0))
                        motion_regions_valid = int(motion_result.get("total_regiones_validas", 0))
                        motion_regions_invalid = int(motion_result.get("total_regiones_invalidas", 0))
                        motion_low_detected = bool(motion_result.get("motion_low_detected", False))
                        motion_low_ratio = float(motion_result.get("motion_low_ratio", 0.0))
                        motion_mean_pair_magnitude = float(motion_result.get("motion_mean_pair_magnitude", 0.0))
                        motion_mean_pair_active_ratio = float(motion_result.get("motion_mean_pair_active_ratio", 0.0))
                        motion_pair_gaps_used = [int(x) for x in motion_result.get("motion_pair_gaps_used", [])]
                        motion_pair_geometry_quality = float(motion_result.get("pair_geometry_mean_quality", 0.0))
                        logger.info(
                            "[MOTION] low_detected=%s low_ratio=%.0f%% mean_mag=%.3f mean_active=%.3f gaps=%s",
                            motion_low_detected,
                            motion_low_ratio * 100.0,
                            motion_mean_pair_magnitude,
                            motion_mean_pair_active_ratio,
                            motion_pair_gaps_used,
                        )
                    else:
                        projection_type = "unknown"
                        motion_reliable = False
                        motion_reliability_reason = "no_secondary_sequences"
                    confidence = ratio
                else:
                    motion_reliable = False
                    motion_reliability_reason = (
                        f"insufficient_structural_frames:{frames_with_line}<{min_frames_with_line_required}"
                    )

        discarded_frames = black_frames + discarded_no_line
        insufficiency_reasons = evaluar_suficiencia_datos(
            total_frames_extraidos=total_frames_extraidos,
            frames_analizados=frames_analyzed,
            frames_descartados=discarded_frames,
            min_frames_analyzed_required=min_frames_analyzed_required,
            max_discard_ratio=max_discard_ratio,
        )

        if projection_type != "equirectangular" and insufficiency_reasons:
            projection_type = "unknown"
            motion_reliable = False
            motion_reliability_reason = ";".join(insufficiency_reasons)

        if projection_type in ("eac", "cubic") and not motion_reliable:
            projection_type = "unknown"

        # Rule B: positive equi evidence can rescue unreliable/unknown when stereo is false.
        # Only fires when frames_with_line > 0 (early equi path already handles the zero case).
        _stereo_is_false = (frames_with_line > 0 and not stereo_result["is_stereo"]) if frames_with_line > 0 else False
        if (
            projection_type not in ("stereo_equi", "eac", "cubic")
            and _stereo_is_false
            and (projection_type in ("unknown", "") or not motion_reliable)
            and equi_agg["is_strong_evidence"]
        ):
            projection_type = "equirectangular"
            confidence = equi_agg["confidence"]
            motion_reliable = True
            motion_reliability_reason = "positive_equirectangular_wraparound_evidence"

        logger.info("\n=== RESULTADO ===")
        logger.info(f"Frames analizados: {frames_analyzed}")
        logger.info(f"Frames con línea: {frames_with_line}")
        logger.info(f"Proyección: {projection_type.upper()}")
        logger.info(f"Confianza: {confidence:.1%}")
        if frames_with_line > 0:
            logger.info(
                f"Stereo hist similarity: {stereo_avg_similarity:.4f} "
                f"(threshold={stereo_hist_threshold:.4f})"
            )
        logger.info(f"Frames con línea guardados: {len(frames_guardados)}\n")

        stats = {
            "total_frames_extracted": total_frames_extraidos,
            "black_frames": black_frames,
            "valid_frames_used": frames_with_line,
            "discarded_frames": discarded_frames,
            "pairs_total": motion_pairs_total,
            "pairs_valid": motion_pairs_valid,
            "pairs_invalid": motion_pairs_invalid,
            "regions_valid": motion_regions_valid,
            "regions_invalid": motion_regions_invalid,
            "motion_low_detected": motion_low_detected,
            "motion_low_ratio": f"{motion_low_ratio:.3f}",
            "motion_mean_pair_magnitude": f"{motion_mean_pair_magnitude:.3f}",
            "motion_mean_pair_active_ratio": f"{motion_mean_pair_active_ratio:.3f}",
            "motion_pair_gaps_used": ",".join(str(g) for g in motion_pair_gaps_used),
            "motion_pair_geometry_quality": f"{motion_pair_geometry_quality:.3f}",
            "avg_eac_score": f"{motion_avg_eac_score:.3f}" if motion_avg_eac_score is not None else "N/A",
            "avg_cubic_score": f"{motion_avg_cubic_score:.3f}" if motion_avg_cubic_score is not None else "N/A",
            "score_margin": f"{motion_score_margin:.3f}",
            "final_classification": projection_type,
            "final_confidence": confidence,
            "equi_usable_frames": equi_agg["usable_frames"],
            "equi_strong_frames": equi_agg["strong_frames"],
            "equi_mean_score": f"{equi_agg['mean_score']:.3f}",
            "equi_median_score": f"{equi_agg['median_score']:.3f}",
        }

        return {
            "projection_type": projection_type,
            "confidence": confidence,
            "frames_analyzed": frames_analyzed,
            "frames_with_line": frames_with_line,
            "frames_with_line_saved": frames_guardados,
            "motion_reliable": motion_reliable,
            "motion_reliability_reason": motion_reliability_reason,
            "motion_pairs_total": motion_pairs_total,
            "motion_pairs_valid": motion_pairs_valid,
            "motion_confidence": motion_confidence,
            "motion_low_detected": motion_low_detected,
            "motion_low_ratio": motion_low_ratio,
            "motion_mean_pair_magnitude": motion_mean_pair_magnitude,
            "motion_mean_pair_active_ratio": motion_mean_pair_active_ratio,
            "motion_pair_gaps_used": motion_pair_gaps_used,
            "stats": stats,
            "video_path": video_path,
            "equi_evidence_score": equi_agg["final_score"],
            "equi_evidence_strong": equi_agg["is_strong_evidence"],
            "equi_evidence_reason": equi_agg["reason"],
        }
    except Exception as exc:
        logger.exception("Error detectando tipo de proyección")
        return {
            "projection_type": "unknown",
            "confidence": 0.0,
            "frames_analyzed": 0,
            "frames_with_line": 0,
            "frames_with_line_saved": [],
            "motion_reliable": False,
            "motion_reliability_reason": "exception",
            "motion_pairs_total": 0,
            "motion_pairs_valid": 0,
            "motion_confidence": 0.0,
            "motion_low_detected": False,
            "motion_low_ratio": 0.0,
            "motion_mean_pair_magnitude": 0.0,
            "motion_mean_pair_active_ratio": 0.0,
            "motion_pair_gaps_used": [],
            "stats": {
                "total_frames_extracted": 0,
                "black_frames": 0,
                "valid_frames_used": 0,
                "discarded_frames": 0,
                "pairs_total": 0,
                "pairs_valid": 0,
                "pairs_invalid": 0,
                "regions_valid": 0,
                "regions_invalid": 0,
                "avg_eac_score": "N/A",
                "avg_cubic_score": "N/A",
                "score_margin": "N/A",
                "final_classification": "unknown",
                "final_confidence": 0.0,
            },
            "error": str(exc),
        }


def _build_detection_retry_plan(num_frames: int, max_retries: int = 2) -> List[Dict[str, int]]:
    requested_frames = max(2, int(num_frames))
    frame_plan: List[int] = [requested_frames]

    for decrement in range(2, 2 * (max_retries + 1), 2):
        candidate = max(2, requested_frames - decrement)
        if candidate != frame_plan[-1]:
            frame_plan.append(candidate)
        if len(frame_plan) >= (max_retries + 1):
            break

    secondary_plan = [5, 9, 13]
    min_line_plan = [7, 6, 4]
    retry_plan: List[Dict[str, int]] = []
    for idx, frame_count in enumerate(frame_plan):
        retry_plan.append(
            {
                "num_frames": frame_count,
                "paso_frames_secundarios": secondary_plan[min(idx, len(secondary_plan) - 1)],
                "min_frames_with_line_required": max(
                    1,
                    min(frame_count, min_line_plan[min(idx, len(min_line_plan) - 1)]),
                ),
            }
        )
    return retry_plan


def run_detection_with_retries(
    video_path: str,
    num_frames: int = 10,
    debug_base_dir: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    gaussian_kernel_size = 5
    gaussian_sigma = 1.2
    flow_algorithm = str(CONFIG["flow_algorithm"])
    motion_flags = _resolve_motion_feature_flags()
    min_valid_pairs = 4
    min_motion_confidence = 0.2
    min_frames_analyzed_required = int(CONFIG["min_frames_analyzed_required"])
    max_discard_ratio = float(CONFIG["max_discard_ratio"])
    min_layout_score_margin = float(CONFIG["min_layout_score_margin"])

    final_detection: Dict[str, Any] = {}
    final_debug_context: Dict[str, str] = {}
    retry_plan = _build_detection_retry_plan(num_frames)
    previous_motion_signal: Optional[float] = None

    for attempt_idx, attempt_plan in enumerate(retry_plan):
        attempt = attempt_idx + 1
        current_num_frames = attempt_plan["num_frames"]
        paso_sec = attempt_plan["paso_frames_secundarios"]
        min_lines = attempt_plan["min_frames_with_line_required"]

        logger.info(
            f"Intento {attempt}/{len(retry_plan)}: "
            f"num_frames={current_num_frames}, paso_secundario={paso_sec}, min_lines={min_lines}"
        )

        debug_context: Dict[str, str] = {}
        if debug_base_dir:
            debug_context = create_run_debug_dir(video_path, debug_base_dir)
            logger.info(f"Directorio de salida para frames principales: {debug_context['main_frames_dir']}")

        frame_result = extract_main_frames(
            video_path,
            modo_extraccion="equiespaciados",
            num_frames=current_num_frames,
            output_dir=debug_context.get("main_frames_dir"),
            guardar_frames=bool(debug_context),
            frame_filename_prefix=debug_context.get("video_tag"),
            save_image_fn=save_frame_debug if debug_context else None,
            log_success_fn=log_success,
            log_discard_fn=log_discard,
        )

        detection_result = run_detection_pipeline(
            video_path,
            num_frames=current_num_frames,
            frames=frame_result.get("frames", []),
            frames_metadata=frame_result.get("frames_metadata", []),
            video_name=frame_result.get("video_name"),
            debug_context=debug_context or None,
            paso_frames_secundarios=paso_sec,
            min_frames_with_line_required=min_lines,
            min_valid_pairs=min_valid_pairs,
            min_motion_confidence=min_motion_confidence,
            min_frames_analyzed_required=min_frames_analyzed_required,
            max_discard_ratio=max_discard_ratio,
            min_layout_score_margin=min_layout_score_margin,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            flow_algorithm=flow_algorithm,
            enable_flow_refinement=bool(motion_flags["enable_refinement"]),
            enable_forward_backward_check=bool(motion_flags["enable_fb_check"]),
            forward_backward_threshold=float(motion_flags["fb_threshold"]),
            enable_geometry_evidence=bool(motion_flags["enable_geometry_evidence"]),
            geometry_evidence_weight=float(motion_flags["geometry_evidence_weight"]),
            confidence_rollout_profile=str(motion_flags["profile"]),
        )

        final_detection = detection_result
        final_debug_context = debug_context

        reliability_reason = str(detection_result.get("motion_reliability_reason", ""))
        low_motion_cause = bool(detection_result.get("motion_low_detected")) or ("low_motion" in reliability_reason)
        current_motion_signal = float(detection_result.get("motion_mean_pair_magnitude", 0.0)) * float(
            detection_result.get("motion_mean_pair_active_ratio", 0.0)
        )
        motion_signal_gain = (
            current_motion_signal - previous_motion_signal
            if previous_motion_signal is not None
            else 0.0
        )
        logger.info(
            "[RETRY][attempt=%d] low_motion_cause=%s motion_signal=%.4f gain=%.4f",
            attempt,
            low_motion_cause,
            current_motion_signal,
            motion_signal_gain,
        )

        if detection_result.get("motion_reliable"):
            logger.info(f"Intento {attempt}: clasificación confiable alcanzada. Fin de reintentos.")
            break
        if detection_result.get("projection_type") == "equirectangular":
            logger.info(f"Intento {attempt}: clasificación temprana EQUIRECTANGULAR alcanzada. Fin de reintentos.")
            break
        if low_motion_cause and previous_motion_signal is not None and motion_signal_gain < 0.005:
            logger.warning(
                f"Intento {attempt}: señal de movimiento sin mejora relevante "
                f"(actual={current_motion_signal:.4f}, ganancia={motion_signal_gain:.4f}). "
                "Cierre anticipado de reintentos."
            )
            break
        if attempt < len(retry_plan):
            if low_motion_cause:
                next_plan = retry_plan[attempt_idx + 1]
                next_plan["paso_frames_secundarios"] = max(
                    next_plan["paso_frames_secundarios"],
                    paso_sec + 6,
                )
                next_plan["num_frames"] = min(
                    next_plan["num_frames"],
                    max(2, current_num_frames - 2),
                )
                next_plan["min_frames_with_line_required"] = min(
                    next_plan["min_frames_with_line_required"],
                    max(1, next_plan["num_frames"] - 2),
                )
            logger.warning(
                f"Intento {attempt}: resultado no confiable "
                f"(motivo={reliability_reason}). "
                f"Reintentando con mayor separación temporal."
            )
        previous_motion_signal = current_motion_signal

    return final_detection, final_debug_context


def process_downloaded_video(video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    final_video_path = video_path  # updated below when compat conversion runs
    try:
        logger.info(f"Procesando video: {video_path}")

        final_video_path = convert_video_codec(video_path)
        final_detection, final_debug_context = run_detection_with_retries(
            final_video_path,
            num_frames=10,
            debug_base_dir=CONFIG["main_runs_dir"],
        )

        final_stats = final_detection.get(
            "stats",
            {
                "total_frames_extracted": 0,
                "black_frames": 0,
                "valid_frames_used": 0,
                "discarded_frames": 0,
                "pairs_total": 0,
                "pairs_valid": 0,
                "pairs_invalid": 0,
                "regions_valid": 0,
                "regions_invalid": 0,
                "avg_eac_score": "N/A",
                "avg_cubic_score": "N/A",
                "score_margin": "N/A",
                "final_classification": "unknown",
                "final_confidence": 0.0,
            },
        )
        format_final_stats(final_stats)

        detected_projection = final_detection.get("projection_type", "unknown")

        # Post-detection conversion: attempt to produce equirectangular output.
        # Conversion runs after all detection is complete and never affects it.
        # Pass the original video_path as name_source_path so that when a
        # compatibility-transcoded intermediate was used the output file name
        # is derived from the original file name, not the temp-file name.
        conversion_output_dir = final_debug_context.get("run_dir") or None
        conversion_result = convert_detected_projection_to_equirectangular(
            video_path=final_video_path,
            projection_type=detected_projection,
            output_dir=conversion_output_dir,
            name_source_path=video_path if final_video_path != video_path else None,
        )

        converted_to_equirectangular = bool(
            conversion_result.get("success") and not conversion_result.get("skipped")
        )
        converted_video_path: Optional[str] = (
            conversion_result.get("output_path") if converted_to_equirectangular else None
        )

        return {
            "success": True,
            "video_path": final_video_path,
            "original_video_path": video_path,
            "debug_run_dir": final_debug_context.get("run_dir"),
            "projection_type": detected_projection,
            "confidence": float(final_detection.get("confidence", 0.0)),
            "frames_analyzed": int(final_detection.get("frames_analyzed", 0)),
            "frames_with_line": int(final_detection.get("frames_with_line", 0)),
            "frames_saved_for_analysis": final_detection.get("frames_with_line_saved", []),
            "motion_reliable": bool(final_detection.get("motion_reliable", False)),
            "motion_reliability_reason": final_detection.get("motion_reliability_reason", ""),
            "motion_pairs_total": int(final_detection.get("motion_pairs_total", 0)),
            "motion_pairs_valid": int(final_detection.get("motion_pairs_valid", 0)),
            "motion_confidence": float(final_detection.get("motion_confidence", 0.0)),
            "stats": final_stats,
            "conversion": conversion_result,
            "converted_to_equirectangular": converted_to_equirectangular,
            "converted_video_path": converted_video_path,
        }
    except Exception as exc:
        logger.exception("Error procesando video")
        return {"success": False, "error": str(exc)}
    finally:
        # Clean up any compatibility-transcoded intermediate file.
        # The temp file is only created when the source needed a codec
        # fallback; in that case final_video_path differs from the original.
        if final_video_path != video_path and os.path.exists(final_video_path):
            try:
                os.unlink(final_video_path)
                logger.debug("[CLEANUP] Removed compat temp file: %s", final_video_path)
            except OSError as exc:
                logger.warning("[CLEANUP] Could not remove compat temp file %s: %s", final_video_path, exc)
