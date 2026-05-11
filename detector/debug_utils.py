import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .line_detection import draw_line_debug


logger = logging.getLogger(__name__)
_RUN_FILE_HANDLER: Optional[logging.FileHandler] = None
_RUN_FILE_HANDLER_LOCK = threading.Lock()


class ColorizedFormatter(logging.Formatter):
    RESET = "\033[0m"
    WHITE = "\033[37m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        message = str(record.getMessage())
        if "[SUCCESS]" in message:
            color = self.GREEN
        elif "[DISCARD]" in message or "[ERROR]" in message:
            color = self.RED
        elif record.levelno >= logging.WARNING:
            color = self.YELLOW
        else:
            color = self.WHITE
        return f"{color}{base}{self.RESET}"


def configure_logging_color() -> None:
    root = logging.getLogger()
    if not root.handlers:
        return
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(
                ColorizedFormatter(
                    fmt="%(asctime)s [%(levelname)-8s] %(message)s",
                    datefmt="%H:%M:%S",
                )
            )


def log_success(message: str) -> None:
    logger.info(f"[SUCCESS] {message}")


def log_discard(message: str) -> None:
    logger.info(f"[DISCARD] {message}")


def save_frame_debug(filepath: str, image: np.ndarray, tag: str) -> bool:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ok = bool(cv2.imwrite(filepath, image))
    if ok:
        log_success(f"[{tag}] SAVED -> {filepath}")
    else:
        log_discard(f"[{tag}] SAVE_FAILED -> {filepath}")
    return ok


def configure_run_file_logging(logs_dir: str) -> Optional[str]:
    global _RUN_FILE_HANDLER
    with _RUN_FILE_HANDLER_LOCK:
        try:
            os.makedirs(logs_dir, exist_ok=True)
            root = logging.getLogger()

            if _RUN_FILE_HANDLER is not None:
                root.removeHandler(_RUN_FILE_HANDLER)
                _RUN_FILE_HANDLER.close()
                _RUN_FILE_HANDLER = None

            log_path = os.path.join(logs_dir, "debug.log.txt")
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.INFO)

            formatter = None
            for handler in root.handlers:
                if handler is not file_handler and handler.formatter is not None:
                    formatter = handler.formatter
                    break
            if formatter is None:
                formatter = logging.Formatter(
                    fmt="%(asctime)s [%(levelname)-8s] %(message)s",
                    datefmt="%H:%M:%S",
                )
            file_handler.setFormatter(formatter)

            root.addHandler(file_handler)
            _RUN_FILE_HANDLER = file_handler
            return log_path
        except Exception:
            logger.exception("No se pudo configurar logging por archivo para la ejecución")
            return None


def draw_search_band(
    frame: np.ndarray,
    debug_line_info: Dict[str, Any],
    found: bool,
    line_orientation: str = "horizontal",
) -> np.ndarray:
    result = {
        "debug_line_info": debug_line_info,
        "line_orientation": line_orientation,
    }
    if line_orientation == "vertical":
        result["has_vertical_line"] = found
    else:
        result["has_horizontal_line"] = found
    return draw_line_debug(frame, result)


def draw_regions(vis: np.ndarray, region_bounds: List[Tuple[int, int, int, int, int, int]]) -> np.ndarray:
    out = vis.copy()
    if not region_bounds:
        return out
    max_col = max(col for _, col, _, _, _, _ in region_bounds)
    max_row = max(row for row, _, _, _, _, _ in region_bounds)
    for _, col, _, _, x0, x1 in region_bounds:
        if col < max_col:
            cv2.line(out, (x1, 0), (x1, out.shape[0]), (180, 180, 180), 1)
    for row, _, _, y1, _, _ in region_bounds:
        if row < max_row:
            cv2.line(out, (0, y1), (out.shape[1], y1), (180, 180, 180), 1)
    return out


def draw_motion_vectors(
    vis: np.ndarray,
    flow: np.ndarray,
    umbral_magnitud: float = 1.0,
    step: int = 28,
) -> np.ndarray:
    out = vis.copy()
    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            fx, fy = flow[y, x]
            m = float(np.sqrt(fx * fx + fy * fy))
            if m <= umbral_magnitud:
                continue
            end_x = int(x + fx * 2.0)
            end_y = int(y + fy * 2.0)
            cv2.arrowedLine(out, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.25)
    return out


def create_run_debug_dir(video_path: str, base_dir: str) -> Dict[str, str]:
    os.makedirs(base_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0].strip() or "video"
    now = datetime.now()
    run_base = f"{video_name}-{now.strftime('%Y-%m-%d')}-{now.strftime('%H-%M-%S')}"

    run_dir = os.path.join(base_dir, run_base)
    if os.path.exists(run_dir):
        counter = 1
        while os.path.exists(os.path.join(base_dir, f"{run_base}-{counter:02d}")):
            counter += 1
        run_dir = os.path.join(base_dir, f"{run_base}-{counter:02d}")

    run_id = os.path.basename(run_dir)
    main_frames_dir = os.path.join(run_dir, "main_frames")
    secondary_frames_dir = os.path.join(run_dir, "secondary_frames")
    motion_vectors_dir = os.path.join(run_dir, "motion_vectors")
    line_detection_dir = os.path.join(run_dir, "line_detection")
    logs_dir = os.path.join(run_dir, "logs")

    for d in [run_dir, main_frames_dir, secondary_frames_dir, motion_vectors_dir, line_detection_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Directorio debug de ejecución: {run_dir}")
    return {
        "run_dir": run_dir,
        "run_id": run_id,
        "main_frames_dir": main_frames_dir,
        "secondary_frames_dir": secondary_frames_dir,
        "motion_vectors_dir": motion_vectors_dir,
        "line_detection_dir": line_detection_dir,
        "logs_dir": logs_dir,
        "secondary_sequences_dir": secondary_frames_dir,
        "motion_visualizations_dir": motion_vectors_dir,
        "horizontal_line_dir": line_detection_dir,
        "video_tag": run_id,
    }


def save_line_detected_frame(
    frame: np.ndarray,
    line_data: Dict[str, Any],
    frame_idx: int,
    output_dir: str,
) -> str:
    try:
        out = frame.copy()
        height, width = frame.shape[:2]
        x1 = int(line_data["x1"])
        y1 = int(line_data["y1"])
        x2 = int(line_data["x2"])
        y2 = int(line_data["y2"])
        is_vertical = "center_x" in line_data

        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        if is_vertical:
            center_x = int(line_data["center_x"])
            cv2.circle(out, (center_x, height // 2), 10, (0, 255, 0), -1)
        else:
            center_y = int(line_data["center_y"])
            cv2.circle(out, (width // 2, center_y), 10, (0, 255, 0), -1)

        os.makedirs(output_dir, exist_ok=True)
        video_pos = int(line_data.get("video_position", -1))
        pos_tag = f"{video_pos:06d}" if video_pos >= 0 else "unknown"
        suffix = "vertical_detected" if is_vertical else "detected"
        filename = f"line_main_{frame_idx:03d}_video_{pos_tag}_{suffix}.jpg"
        filepath = os.path.join(output_dir, filename)
        save_frame_debug(filepath, out, "LINE")
        return filepath
    except Exception:
        logger.exception("Error guardando frame con línea")
        return ""


def save_line_visual_debug(
    frame: np.ndarray,
    frame_idx: int,
    output_dir: str,
    debug_line_info: Dict[str, Any],
    found: bool,
    line_orientation: str = "horizontal",
) -> str:
    try:
        vis = draw_search_band(frame, debug_line_info, found, line_orientation=line_orientation)
        os.makedirs(output_dir, exist_ok=True)
        status_tag = "found" if found else "not_found"
        if line_orientation == "vertical":
            filename = f"frame_{frame_idx:02d}_vertical_line_debug_{status_tag}.jpg"
        else:
            filename = f"frame_{frame_idx:02d}_line_debug_{status_tag}.jpg"
        filepath = os.path.join(output_dir, filename)
        save_frame_debug(filepath, vis, "LINE_DEBUG")
        return filepath
    except Exception:
        logger.exception("Error guardando visual debug de línea")
        return ""


def save_stereo_halves(frame_idx: int, top: np.ndarray, bottom: np.ndarray, output_dir: str) -> None:
    top_path = os.path.join(output_dir, f"top_half_frame_{frame_idx+1:03d}.jpg")
    bottom_path = os.path.join(output_dir, f"bottom_half_frame_{frame_idx+1:03d}.jpg")
    save_frame_debug(top_path, top, "STEREO")
    save_frame_debug(bottom_path, bottom, "STEREO")


def format_final_stats(stats: Dict[str, Any]) -> None:
    def _fmt_pct(value: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{(value / total) * 100.0:.1f}%"

    logger.info("=" * 70)
    logger.info("  RESUMEN FINAL DE ESTADISTICAS")
    logger.info("=" * 70)
    logger.info(f"  Frames extraidos (main): {stats['total_frames_extracted']}")
    logger.info(
        f"  Frames negros: {stats['black_frames']} ({_fmt_pct(stats['black_frames'], stats['total_frames_extracted'])})"
    )
    logger.info(
        f"  Frames validos usados: {stats['valid_frames_used']} ({_fmt_pct(stats['valid_frames_used'], stats['total_frames_extracted'])})"
    )
    logger.info(
        f"  Frames descartados: {stats['discarded_frames']} ({_fmt_pct(stats['discarded_frames'], stats['total_frames_extracted'])})"
    )
    logger.info(f"  Pares analizados: {stats['pairs_total']}")
    logger.info(f"  Pares validos: {stats['pairs_valid']}")
    logger.info(f"  Pares invalidos: {stats['pairs_invalid']}")
    logger.info(f"  Regiones validas: {stats['regions_valid']}")
    logger.info(f"  Regiones invalidas: {stats['regions_invalid']}")
    logger.info(f"  Score EAC medio: {stats['avg_eac_score']}")
    logger.info(f"  Score CUBIC medio: {stats['avg_cubic_score']}")
    logger.info(f"  Margen de score: {stats['score_margin']}")

    final_msg = (
        f"Clasificacion final: {str(stats['final_classification']).upper()} "
        f"(confianza={float(stats['final_confidence']):.1%})"
    )
    if stats.get("final_classification") == "unknown":
        log_discard(final_msg)
    else:
        log_success(final_msg)
    logger.info("=" * 70)
