import os
import json
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .projection_conversion import detect_ffmpeg_h264_encoder, _is_hardware_encoder_runtime_failure


class FrameExtractorError(Exception):
    """Excepción base para errores de extracción de frames."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "frame_extraction_error",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


# ---------------------------------------------------------------------------
# 1c — probe_video_stream cache
# ---------------------------------------------------------------------------
_PROBE_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}


def _probe_cache_key(video_path: str) -> Tuple[str, float]:
    abs_path = os.path.abspath(video_path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0.0
    return (abs_path, mtime)


def clear_probe_cache() -> None:
    """Limpia la caché de resultados de probe_video_stream."""
    _PROBE_CACHE.clear()


def _can_decode_with_opencv(video_path: str) -> bool:
    cap_test = cv2.VideoCapture(video_path)
    can_decode = cap_test.isOpened()
    if can_decode:
        ret, _ = cap_test.read()
        can_decode = bool(ret)
    cap_test.release()
    return can_decode


def _run_ffprobe_json(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise FrameExtractorError(f"ffprobe falló: {stderr or 'sin detalle'}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FrameExtractorError("ffprobe devolvió JSON inválido") from exc


def _parse_ratio(value: Optional[str]) -> float:
    if not value:
        return 0.0
    txt = str(value).strip()
    if "/" in txt:
        left, right = txt.split("/", 1)
        try:
            den = float(right)
            if den == 0:
                return 0.0
            return float(left) / den
        except ValueError:
            return 0.0
    try:
        return float(txt)
    except ValueError:
        return 0.0


def probe_video_stream(video_path: str) -> Dict[str, Any]:
    """Devuelve metadatos de stream principal usando ffprobe cuando está disponible."""
    cache_key = _probe_cache_key(video_path)
    if cache_key in _PROBE_CACHE:
        return dict(_PROBE_CACHE[cache_key])

    metadata: Dict[str, Any] = {
        "path": video_path,
        "codec_name": None,
        "pix_fmt": None,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration": 0.0,
        "total_frames": 0,
        "source": "unavailable",
    }
    try:
        raw = _run_ffprobe_json(video_path)
    except Exception:
        return metadata

    streams = raw.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {}) or {}
    fmt = raw.get("format") or {}
    fps = _parse_ratio(video_stream.get("avg_frame_rate")) or _parse_ratio(video_stream.get("r_frame_rate"))
    duration = 0.0
    try:
        duration = float(video_stream.get("duration") or fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    total_frames = 0
    try:
        total_frames = int(video_stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        total_frames = 0
    if total_frames <= 0 and duration > 0 and fps > 0:
        total_frames = max(1, int(round(duration * fps)))

    metadata.update(
        {
            "codec_name": video_stream.get("codec_name"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "width": int(video_stream.get("width") or 0),
            "height": int(video_stream.get("height") or 0),
            "fps": float(fps),
            "duration": float(duration),
            "total_frames": int(total_frames),
            "source": "ffprobe",
        }
    )
    _PROBE_CACHE[cache_key] = dict(metadata)
    return metadata


def _extract_single_frame_ffmpeg(video_path: str, timestamp_seconds: float, timeout: int = 25) -> Optional[np.ndarray]:
    frame, _diag = _extract_single_frame_ffmpeg_with_diagnostics(
        video_path,
        timestamp_seconds,
        timeout=timeout,
    )
    return frame


def _extract_single_frame_ffmpeg_with_diagnostics(
    video_path: str,
    timestamp_seconds: float,
    timeout: int = 25,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    ts = max(0.0, float(timestamp_seconds))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{ts:.6f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    diag: Dict[str, Any] = {
        "mode": "single",
        "timestamp": ts,
        "timeout_seconds": timeout,
        "timeout": False,
        "returncode": None,
        "stderr": "",
        "decoded": False,
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        diag["timeout"] = True
        diag["stderr"] = f"ffmpeg single-frame timeout: {exc}"
        diag["error_code"] = "ffmpeg_single_timeout"
        return None, diag
    except Exception as exc:
        diag["stderr"] = f"ffmpeg single-frame exception: {exc}"
        diag["error_code"] = "ffmpeg_single_exception"
        return None, diag

    diag["returncode"] = proc.returncode
    diag["stderr"] = proc.stderr.decode("utf-8", errors="ignore").strip()
    if proc.returncode != 0 or not proc.stdout:
        diag["error_code"] = "ffmpeg_single_no_frame"
        return None, diag

    buffer = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        diag["error_code"] = "ffmpeg_single_decode_failed"
        return None, diag
    diag["decoded"] = True
    return frame, diag


def _compute_even_positions(start_frame: int, end_frame: int, num_frames: int) -> List[int]:
    if num_frames <= 1:
        return [start_frame]
    return [
        start_frame + int(i * (end_frame - start_frame) / (num_frames - 1))
        for i in range(num_frames)
    ]


def _split_mjpeg_stream(data: bytes) -> List[bytes]:
    """Divide un stream MJPEG concatenado en imágenes JPEG individuales."""
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"
    images: List[bytes] = []
    pos = 0
    while pos < len(data):
        start = data.find(SOI, pos)
        if start == -1:
            break
        end = data.find(EOI, start + 2)
        if end == -1:
            break
        images.append(data[start : end + 2])
        pos = end + 2
    return images


def _extract_batch_frames_ffmpeg(
    video_path: str,
    timestamps_seconds: List[float],
    timeout: int = 120,
    return_diagnostics: bool = False,
) -> Union[List[Optional[np.ndarray]], Tuple[List[Optional[np.ndarray]], Dict[str, Any]]]:
    """Extrae todos los frames indicados en un único proceso ffmpeg usando el filtro select."""
    if not timestamps_seconds:
        empty_result: List[Optional[np.ndarray]] = []
        if return_diagnostics:
            return empty_result, {
                "mode": "batch",
                "requested_timestamps": [],
                "timeout_seconds": timeout,
                "timeout": False,
                "returncode": 0,
                "stderr": "",
                "raw_images_found": 0,
                "frames_decoded": 0,
                "error_code": None,
            }
        return empty_result

    select_expr = "+".join(
        f"lt(prev_pts*TB,{t:.6f})*gte(pts*TB,{t:.6f})" for t in timestamps_seconds
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "vfr",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    diag: Dict[str, Any] = {
        "mode": "batch",
        "requested_timestamps": [round(float(t), 6) for t in timestamps_seconds],
        "timeout_seconds": timeout,
        "timeout": False,
        "returncode": None,
        "stderr": "",
        "raw_images_found": 0,
        "frames_decoded": 0,
        "error_code": None,
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        diag["timeout"] = True
        diag["stderr"] = f"ffmpeg batch timeout: {exc}"
        diag["error_code"] = "ffmpeg_batch_timeout"
        results = [None] * len(timestamps_seconds)
        return (results, diag) if return_diagnostics else results
    except Exception as exc:
        diag["stderr"] = f"ffmpeg batch exception: {exc}"
        diag["error_code"] = "ffmpeg_batch_exception"
        results = [None] * len(timestamps_seconds)
        return (results, diag) if return_diagnostics else results

    diag["returncode"] = proc.returncode
    diag["stderr"] = proc.stderr.decode("utf-8", errors="ignore").strip()
    if proc.returncode != 0 or not proc.stdout:
        diag["error_code"] = "ffmpeg_batch_no_output"
        results = [None] * len(timestamps_seconds)
        return (results, diag) if return_diagnostics else results

    raw_images = _split_mjpeg_stream(proc.stdout)
    diag["raw_images_found"] = len(raw_images)
    results: List[Optional[np.ndarray]] = []
    for i in range(len(timestamps_seconds)):
        if i < len(raw_images):
            buf = np.frombuffer(raw_images[i], dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            results.append(frame if (frame is not None and frame.size > 0) else None)
        else:
            results.append(None)
    diag["frames_decoded"] = sum(1 for f in results if f is not None)
    if diag["frames_decoded"] == 0:
        diag["error_code"] = "ffmpeg_batch_decoded_empty"
    return (results, diag) if return_diagnostics else results


# ---------------------------------------------------------------------------
# 1b — Shared VideoCapture session
# ---------------------------------------------------------------------------

class _CachedCapture:
    """Wrapper sobre cv2.VideoCapture que gestiona el ciclo de vida de forma segura."""

    def __init__(self, video_path: str) -> None:
        self._path = video_path
        self._cap = cv2.VideoCapture(video_path)
        self.can_decode: bool = self._cap.isOpened()
        if self.can_decode:
            ret, _ = self._cap.read()
            self.can_decode = bool(ret)
            if self.can_decode:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.can_decode else 0
        self.fps: float = float(self._cap.get(cv2.CAP_PROP_FPS)) if self.can_decode else 0.0

    def read_frame(self, position: int) -> Optional[np.ndarray]:
        if not self.can_decode:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self._cap.read()
        if ret and frame is not None:
            return frame
        return None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "_CachedCapture":
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def open_video_capture(video_path: str) -> "_CachedCapture":
    """Abre un VideoCapture gestionado como context manager."""
    return _CachedCapture(video_path)


def convert_video_codec(video_path: str) -> str:
    """Asegura compatibilidad de decodificación con OpenCV."""
    if not os.path.exists(video_path):
        raise FrameExtractorError(f"El archivo de vídeo no existe: {video_path}")

    if _can_decode_with_opencv(video_path):
        return video_path

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=8)
    except Exception as exc:
        raise FrameExtractorError(
            "OpenCV no puede decodificar este video y ffmpeg no está disponible para fallback."
        ) from exc

    input_dir = os.path.dirname(os.path.abspath(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    with tempfile.NamedTemporaryFile(
        prefix=f"{base_name}_compat_",
        suffix=".mp4",
        dir=input_dir,
        delete=False,
    ) as tmp:
        compat_path = tmp.name

    encoder = detect_ffmpeg_h264_encoder()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "auto",
        "-i",
        video_path,
        "-c:v",
        encoder,
    ]
    if encoder == "libx264":
        cmd.extend(["-preset", "veryfast", "-crf", "23"])
    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-an",
            compat_path,
        ]
    )

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
        if proc.returncode != 0 and encoder != "libx264":
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            if _is_hardware_encoder_runtime_failure(stderr, encoder):
                fallback_cmd = list(cmd)
                idx = fallback_cmd.index("-c:v") + 1
                fallback_cmd[idx] = "libx264"
                insert_args = []
                if "-preset" not in fallback_cmd:
                    insert_args.extend(["-preset", "veryfast"])
                if "-crf" not in fallback_cmd:
                    insert_args.extend(["-crf", "23"])
                if insert_args:
                    fallback_cmd[idx + 1:idx + 1] = insert_args
                proc = subprocess.run(fallback_cmd, capture_output=True, timeout=900)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            raise FrameExtractorError(f"Fallback ffmpeg falló: {stderr}")

        cap_compat = cv2.VideoCapture(compat_path)
        compat_ok = cap_compat.isOpened()
        if compat_ok:
            ret, _ = cap_compat.read()
            compat_ok = bool(ret)
        cap_compat.release()

        if not compat_ok:
            raise FrameExtractorError("El video convertido no pudo abrirse con OpenCV")

        return compat_path

    except Exception:
        if os.path.exists(compat_path):
            try:
                os.unlink(compat_path)
            except OSError:
                pass
        raise


def extract_main_frames(
    video_path: str,
    num_frames: int = 5,
    modo_extraccion: str = "equiespaciados",
    delay_segundos: float = 3.0,
    padding_segundos: float = 3.0,
    output_dir: Optional[str] = None,
    guardar_frames: bool = False,
    frame_filename_prefix: Optional[str] = None,
    save_image_fn: Optional[Callable[[str, np.ndarray, str], bool]] = None,
    log_success_fn: Optional[Callable[[str], None]] = None,
    log_discard_fn: Optional[Callable[[str], None]] = None,
    cap_session: Optional[_CachedCapture] = None,
) -> Dict[str, Any]:
    """Extrae frames principales con estrategia equiespaciada o primera/última."""
    if not os.path.exists(video_path):
        raise FrameExtractorError(f"El archivo de vídeo no existe: {video_path}")
    if modo_extraccion not in ["equiespaciados", "primera_ultima"]:
        raise FrameExtractorError(f"Modo de extracción inválido: {modo_extraccion}")

    _owns_cap = cap_session is None
    try:
        video_path_procesado = video_path
        if cap_session is None:
            cap_session = _CachedCapture(video_path)

        can_decode_opencv = cap_session.can_decode
        metadata = probe_video_stream(video_path)
        total_frames = cap_session.total_frames if can_decode_opencv else int(metadata.get("total_frames", 0))
        fps = cap_session.fps if can_decode_opencv else float(metadata.get("fps", 0.0))
        if total_frames <= 0:
            total_frames = int(metadata.get("total_frames", 0))
        if fps <= 0:
            fps = float(metadata.get("fps", 0.0))
        duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else float(metadata.get("duration", 0.0))
        if total_frames <= 0:
            if _owns_cap:
                cap_session.release()
            raise FrameExtractorError("No se pudo determinar el total de frames del vídeo")
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        frame_positions: List[int] = []
        if modo_extraccion == "equiespaciados":
            padding_frames = int(padding_segundos * fps) if fps > 0 else 0
            start_frame = min(padding_frames, total_frames // 10)
            end_frame = max(total_frames - 1 - padding_frames, start_frame)
            usable = end_frame - start_frame + 1
            if num_frames > usable:
                num_frames = usable
            frame_positions = _compute_even_positions(start_frame, end_frame, num_frames)
        else:
            if fps <= 0:
                raise FrameExtractorError("No se pudo obtener FPS válido del vídeo")
            first_frame_pos = int(delay_segundos * fps)
            if first_frame_pos >= total_frames:
                first_frame_pos = total_frames - 1
            frame_positions = [first_frame_pos, total_frames - 1]

        frames: List[np.ndarray] = []
        frame_metadata: List[Dict[str, Any]] = []
        fallback_ffmpeg_frames = 0
        fallback_ffmpeg_single_frames = 0
        extraction_attempts: List[Dict[str, Any]] = []
        has_timeout = False

        extraction_position_plans: List[List[int]] = [list(frame_positions)]
        if modo_extraccion == "equiespaciados" and len(frame_positions) >= 4:
            reduced_count = max(2, min(len(frame_positions) - 2, len(frame_positions) // 2))
            reduced_positions = _compute_even_positions(frame_positions[0], frame_positions[-1], reduced_count)
            if reduced_positions != frame_positions:
                extraction_position_plans.append(reduced_positions)

        for attempt_idx, active_positions in enumerate(extraction_position_plans, start=1):
            if fps > 0:
                selected_timestamps = [p / fps for p in active_positions]
            elif duration > 0 and total_frames > 1:
                selected_timestamps = [(p / (total_frames - 1)) * duration for p in active_positions]
            else:
                selected_timestamps = [0.0 for _ in active_positions]

            attempt_diag: Dict[str, Any] = {
                "attempt": attempt_idx,
                "requested_positions": list(active_positions),
                "requested_timestamps": [round(float(t), 6) for t in selected_timestamps],
                "opencv_decode_capable": bool(can_decode_opencv),
                "ffmpeg_batch": None,
                "ffmpeg_single": [],
                "frames_recovered": 0,
            }

            ffmpeg_needed: List[int] = []
            candidate_frames: Dict[int, Optional[np.ndarray]] = {}
            for idx_pos, frame_pos in enumerate(active_positions):
                frame: Optional[np.ndarray] = None
                if can_decode_opencv:
                    frame = cap_session.read_frame(frame_pos)
                if frame is None:
                    ffmpeg_needed.append(idx_pos)
                else:
                    candidate_frames[idx_pos] = frame

            if ffmpeg_needed:
                batch_ts = [selected_timestamps[i] for i in ffmpeg_needed]
                batch_results, batch_diag = _extract_batch_frames_ffmpeg(
                    video_path,
                    batch_ts,
                    return_diagnostics=True,
                )
                attempt_diag["ffmpeg_batch"] = batch_diag
                if batch_diag.get("timeout"):
                    has_timeout = True
                for list_idx, orig_idx in enumerate(ffmpeg_needed):
                    frame_candidate = batch_results[list_idx] if list_idx < len(batch_results) else None
                    candidate_frames[orig_idx] = frame_candidate
                    if frame_candidate is not None:
                        fallback_ffmpeg_frames += 1

                remaining_missing = [idx for idx in ffmpeg_needed if candidate_frames.get(idx) is None]
                for missing_idx in remaining_missing:
                    single_frame, single_diag = _extract_single_frame_ffmpeg_with_diagnostics(
                        video_path,
                        selected_timestamps[missing_idx],
                    )
                    attempt_diag["ffmpeg_single"].append(single_diag)
                    if single_diag.get("timeout"):
                        has_timeout = True
                    candidate_frames[missing_idx] = single_frame
                    if single_frame is not None:
                        fallback_ffmpeg_single_frames += 1

            attempt_frames: List[np.ndarray] = []
            attempt_metadata: List[Dict[str, Any]] = []
            for idx, frame_pos in enumerate(active_positions, start=1):
                frame = candidate_frames.get(idx - 1)
                if frame is not None:
                    attempt_frames.append(frame.copy())  # type: ignore[union-attr]
                    attempt_metadata.append(
                        {
                            "position": frame_pos,
                            "timestamp": frame_pos / fps if fps > 0 else 0,
                        }
                    )
                    if log_success_fn:
                        log_success_fn(
                            f"[MAIN][idx={idx:02d}][video_frame={frame_pos:06d}] -> USED "
                            f"(captured for structural analysis)"
                        )
                elif log_discard_fn:
                    log_discard_fn(
                        f"[MAIN][idx={idx:02d}][video_frame={frame_pos:06d}] -> DISCARDED (read_failed)"
                    )

            attempt_diag["frames_recovered"] = len(attempt_frames)
            extraction_attempts.append(attempt_diag)
            if attempt_frames:
                frames = attempt_frames
                frame_metadata = attempt_metadata
                frame_positions = active_positions
                break

        if _owns_cap:
            cap_session.release()
        if not frames:
            error_code = "frame_extraction_timeout" if has_timeout else "frame_extraction_decode_failure"
            raise FrameExtractorError(
                "No se pudieron extraer frames del vídeo",
                code=error_code,
                details={
                    "video_path": video_path,
                    "opencv_decode_capable": bool(can_decode_opencv),
                    "mode": modo_extraccion,
                    "requested_frame_count": num_frames,
                    "attempts": extraction_attempts,
                },
            )

        result: Dict[str, Any] = {
            "frames": frames,
            "video_name": video_name,
            "video_path_procesado": video_path_procesado,
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "frames_metadata": frame_metadata,
            "frames_paths": [],
            "message": f"Extraídos {len(frames)} frames en memoria",
            "fallback_ffmpeg_frames": fallback_ffmpeg_frames,
            "fallback_ffmpeg_single_frames": fallback_ffmpeg_single_frames,
            "extraction_diagnostics": extraction_attempts,
        }

        if guardar_frames:
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "frames")
            os.makedirs(output_dir, exist_ok=True)
            prefix = f"{frame_filename_prefix}_" if frame_filename_prefix else ""
            if modo_extraccion == "equiespaciados":
                filenames = [
                    f"{prefix}main_frame_{i+1:03d}_video_{frame_metadata[i]['position']:06d}_used.jpg"
                    for i in range(len(frames))
                ]
                msg = f"Extraídos {len(frames)} frames equiespaciados"
            else:
                filenames = [
                    f"{prefix}main_frame_001_video_{frame_metadata[0]['position']:06d}_used.jpg",
                    f"{prefix}main_frame_002_video_{frame_metadata[-1]['position']:06d}_used.jpg",
                ]
                msg = "Frames inicio/final"

            for frame, filename in zip(frames, filenames):
                filepath = os.path.join(output_dir, filename)
                if save_image_fn is not None:
                    if save_image_fn(filepath, frame, "MAIN"):
                        result["frames_paths"].append(filepath)
                else:
                    ok = bool(cv2.imwrite(filepath, frame))
                    if ok:
                        result["frames_paths"].append(filepath)

            result["message"] = f"{msg} guardados en: {output_dir}"

        return result

    except FrameExtractorError:
        raise
    except Exception as exc:
        raise FrameExtractorError(
            f"Error extrayendo frames: {str(exc)}",
            code="frame_extraction_unexpected",
            details={"video_path": video_path},
        ) from exc


def extract_secondary_frames(
    video_path: str,
    frame_positions: List[int],
    total_frames: int,
    paso_frames: int = 5,
    cap_session: Optional[_CachedCapture] = None,
) -> List[List[Dict[str, Any]]]:
    """Extrae secuencias temporales 3-frame alrededor de cada posición principal."""
    secuencias: List[List[Dict[str, Any]]] = []
    metadata = probe_video_stream(video_path)
    fps = float(metadata.get("fps", 0.0))
    duration = float(metadata.get("duration", 0.0))

    _owns_cap = cap_session is None
    if cap_session is None:
        cap_session = _CachedCapture(video_path)
    cap_ok = cap_session.can_decode

    # Collect all window positions across all sequences that need ffmpeg fallback
    all_window_positions: List[Tuple[int, int]] = []  # (seq_idx, position)
    window_sets: List[List[int]] = []
    for pos in frame_positions:
        posiciones_ventana = sorted({
            max(0, pos - paso_frames),
            pos,
            min(total_frames - 1, pos + paso_frames),
        })
        window_sets.append(posiciones_ventana)

    # First pass: read via OpenCV where possible; record which need ffmpeg
    all_seq_frames: List[List[Optional[np.ndarray]]] = []
    ffmpeg_needed_positions: List[Tuple[int, int, float]] = []  # (seq_idx, win_idx, timestamp)

    for seq_idx, posiciones_ventana in enumerate(window_sets):
        seq_frames: List[Optional[np.ndarray]] = []
        for win_idx, p in enumerate(posiciones_ventana):
            frame: Optional[np.ndarray] = None
            if cap_ok:
                frame = cap_session.read_frame(p)
            if frame is None and total_frames > 0:
                if fps > 0:
                    ts = p / fps
                elif duration > 0 and total_frames > 1:
                    ts = (p / (total_frames - 1)) * duration
                else:
                    ts = None
                if ts is not None:
                    ffmpeg_needed_positions.append((seq_idx, win_idx, ts))
            seq_frames.append(frame)
        all_seq_frames.append(seq_frames)

    # Batch ffmpeg for all missing frames
    if ffmpeg_needed_positions:
        batch_ts = [t for (_, _, t) in ffmpeg_needed_positions]
        batch_results = _extract_batch_frames_ffmpeg(video_path, batch_ts)
        for list_idx, (seq_idx, win_idx, _) in enumerate(ffmpeg_needed_positions):
            f = batch_results[list_idx] if list_idx < len(batch_results) else None
            all_seq_frames[seq_idx][win_idx] = f

    # Build output structure
    for seq_idx, posiciones_ventana in enumerate(window_sets):
        seq: List[Dict[str, Any]] = []
        for win_idx, p in enumerate(posiciones_ventana):
            frame = all_seq_frames[seq_idx][win_idx]
            if frame is not None:
                seq.append({"position": p, "frame": frame.copy(), "valid": True})
            else:
                seq.append({"position": p, "frame": None, "valid": False})
        secuencias.append(seq)

    if _owns_cap:
        cap_session.release()
    return secuencias
