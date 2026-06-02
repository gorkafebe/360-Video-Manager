import os
import json
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np


class FrameExtractorError(Exception):
    """Excepción base para errores de extracción de frames."""


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
    return metadata


def _detect_ffmpeg_h264_encoder() -> str:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "libx264"
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for candidate in ("h264_nvenc", "h264_qsv", "h264_videotoolbox", "h264_amf", "libx264"):
        if candidate in text:
            return candidate
    return "libx264"


def _extract_single_frame_ffmpeg(video_path: str, timestamp_seconds: float, timeout: int = 25) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(timestamp_seconds)):.6f}",
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
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout:
        return None
    buffer = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


def _compute_even_positions(start_frame: int, end_frame: int, num_frames: int) -> List[int]:
    if num_frames <= 1:
        return [start_frame]
    return [
        start_frame + int(i * (end_frame - start_frame) / (num_frames - 1))
        for i in range(num_frames)
    ]


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

    encoder = _detect_ffmpeg_h264_encoder()
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
) -> Dict[str, Any]:
    """Extrae frames principales con estrategia equiespaciada o primera/última."""
    if not os.path.exists(video_path):
        raise FrameExtractorError(f"El archivo de vídeo no existe: {video_path}")
    if modo_extraccion not in ["equiespaciados", "primera_ultima"]:
        raise FrameExtractorError(f"Modo de extracción inválido: {modo_extraccion}")

    try:
        video_path_procesado = video_path
        cap = cv2.VideoCapture(video_path)
        can_decode_opencv = cap.isOpened()
        first_read_ok = False
        if can_decode_opencv:
            first_read_ok, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            can_decode_opencv = bool(first_read_ok)

        metadata = probe_video_stream(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if can_decode_opencv else int(metadata.get("total_frames", 0))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if can_decode_opencv else float(metadata.get("fps", 0.0))
        if total_frames <= 0:
            total_frames = int(metadata.get("total_frames", 0))
        if fps <= 0:
            fps = float(metadata.get("fps", 0.0))
        duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else float(metadata.get("duration", 0.0))
        if total_frames <= 0:
            cap.release()
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
        for idx, frame_pos in enumerate(frame_positions, start=1):
            frame = None
            ret = False
            if can_decode_opencv:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
            if not ret or frame is None:
                if fps > 0:
                    frame = _extract_single_frame_ffmpeg(video_path, frame_pos / fps)
                elif duration > 0 and total_frames > 1:
                    frame = _extract_single_frame_ffmpeg(video_path, (frame_pos / (total_frames - 1)) * duration)
                ret = frame is not None
                if ret:
                    fallback_ffmpeg_frames += 1
            if ret:
                frames.append(frame.copy())
                frame_metadata.append({
                    "position": frame_pos,
                    "timestamp": frame_pos / fps if fps > 0 else 0,
                })
                if log_success_fn:
                    log_success_fn(
                        f"[MAIN][idx={idx:02d}][video_frame={frame_pos:06d}] -> USED "
                        f"(captured for structural analysis)"
                    )
            else:
                if log_discard_fn:
                    log_discard_fn(
                        f"[MAIN][idx={idx:02d}][video_frame={frame_pos:06d}] -> DISCARDED (read_failed)"
                    )

        cap.release()
        if not frames:
            raise FrameExtractorError("No se pudieron extraer frames del vídeo")

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

    except Exception as exc:
        raise FrameExtractorError(f"Error extrayendo frames: {str(exc)}") from exc


def extract_secondary_frames(
    video_path: str,
    frame_positions: List[int],
    total_frames: int,
    paso_frames: int = 5,
) -> List[List[Dict[str, Any]]]:
    """Extrae secuencias temporales 3-frame alrededor de cada posición principal."""
    secuencias: List[List[Dict[str, Any]]] = []
    metadata = probe_video_stream(video_path)
    fps = float(metadata.get("fps", 0.0))
    duration = float(metadata.get("duration", 0.0))
    cap = cv2.VideoCapture(video_path)
    cap_ok = cap.isOpened()
    if cap_ok:
        probe_ret, _ = cap.read()
        cap_ok = bool(probe_ret)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for pos in frame_positions:
        posiciones_ventana = sorted({
            max(0, pos - paso_frames),
            pos,
            min(total_frames - 1, pos + paso_frames),
        })
        seq: List[Dict[str, Any]] = []
        for p in posiciones_ventana:
            frame = None
            ret = False
            if cap_ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                ret, frame = cap.read()
            if (not ret or frame is None) and total_frames > 0:
                if fps > 0:
                    frame = _extract_single_frame_ffmpeg(video_path, p / fps)
                elif duration > 0 and total_frames > 1:
                    frame = _extract_single_frame_ffmpeg(video_path, (p / (total_frames - 1)) * duration)
                ret = frame is not None
            if ret:
                seq.append({"position": p, "frame": frame.copy(), "valid": True})
            else:
                seq.append({"position": p, "frame": None, "valid": False})
        secuencias.append(seq)

    cap.release()
    return secuencias
