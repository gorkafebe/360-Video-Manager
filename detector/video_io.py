import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np


class FrameExtractorError(Exception):
    """Excepción base para errores de extracción de frames."""


def convert_video_codec(video_path: str) -> str:
    """Asegura compatibilidad de decodificación con OpenCV."""
    if not os.path.exists(video_path):
        raise FrameExtractorError(f"El archivo de vídeo no existe: {video_path}")

    cap_test = cv2.VideoCapture(video_path)
    can_decode = cap_test.isOpened()
    if can_decode:
        ret, _ = cap_test.read()
        can_decode = bool(ret)
    cap_test.release()

    if can_decode:
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

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "none",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        compat_path,
    ]

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
        video_path_procesado = convert_video_codec(video_path)
        cap = cv2.VideoCapture(video_path_procesado)
        if not cap.isOpened():
            raise FrameExtractorError(f"No se pudo abrir el vídeo: {video_path_procesado}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        frame_positions: List[int] = []
        if modo_extraccion == "equiespaciados":
            padding_frames = int(padding_segundos * fps) if fps > 0 else 0
            start_frame = min(padding_frames, total_frames // 10)
            end_frame = max(total_frames - 1 - padding_frames, start_frame)
            usable = end_frame - start_frame + 1
            if num_frames > usable:
                num_frames = usable
            for i in range(num_frames):
                if num_frames > 1:
                    frame_pos = start_frame + int(i * (end_frame - start_frame) / (num_frames - 1))
                else:
                    frame_pos = start_frame
                frame_positions.append(frame_pos)
        else:
            if fps <= 0:
                raise FrameExtractorError("No se pudo obtener FPS válido del vídeo")
            first_frame_pos = int(delay_segundos * fps)
            if first_frame_pos >= total_frames:
                first_frame_pos = total_frames - 1
            frame_positions = [first_frame_pos, total_frames - 1]

        frames: List[np.ndarray] = []
        frame_metadata: List[Dict[str, Any]] = []
        for idx, frame_pos in enumerate(frame_positions, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return secuencias

    for pos in frame_positions:
        posiciones_ventana = sorted({
            max(0, pos - paso_frames),
            pos,
            min(total_frames - 1, pos + paso_frames),
        })
        seq: List[Dict[str, Any]] = []
        for p in posiciones_ventana:
            cap.set(cv2.CAP_PROP_POS_FRAMES, p)
            ret, frame = cap.read()
            if ret:
                seq.append({"position": p, "frame": frame.copy(), "valid": True})
            else:
                seq.append({"position": p, "frame": None, "valid": False})
        secuencias.append(seq)

    cap.release()
    return secuencias
