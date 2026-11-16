import concurrent.futures
import logging
import os
import json
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .projection_conversion import (
    build_encoder_command_parts,
    detect_ffmpeg_h264_encoder,
    _is_hardware_encoder_runtime_failure,
)

logger = logging.getLogger(__name__)

_AV1_CODEC_NAMES = frozenset({"av1", "libaom-av1", "libdav1d"})


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


# Hard ceiling on the OpenCV decode probe. Every ffmpeg subprocess in this
# module is timeout-guarded; cv2.VideoCapture.read() is a blocking call into a
# C extension that Python cannot interrupt, so without an external deadline a
# malformed or partially-downloaded file could wedge the pipeline forever.
_OPENCV_PROBE_TIMEOUT_SECONDS = 20.0


def _probe_opencv_decode(video_path: str) -> Tuple[bool, int, float]:
    """Open *video_path* with OpenCV and decode one frame.

    Returns ``(can_decode, total_frames, fps)``. Runs in the calling thread;
    callers that need a deadline should go through :func:`_can_decode_with_opencv`
    or :class:`_CachedCapture`, which wrap this in a watchdog.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        can_decode = cap.isOpened()
        if can_decode:
            ret, _ = cap.read()
            can_decode = bool(ret)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if can_decode else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if can_decode else 0.0
    finally:
        cap.release()
    return can_decode, total_frames, fps


def _run_with_deadline(fn: Callable[[], Any], timeout: float, label: str) -> Optional[Any]:
    """Run *fn* on a worker thread, returning None if it outlives *timeout*.

    The thread is left running (a blocked OpenCV read cannot be cancelled from
    Python) but it is a daemon, so it never keeps the process alive. The point
    is that the *pipeline* stops waiting and can fall back to ffmpeg.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "[DECODE] %s exceeded %.0fs and was abandoned; falling back to ffmpeg.",
            label, timeout,
        )
        return None
    finally:
        # Do not join a possibly-wedged worker.
        pool.shutdown(wait=False)


def _can_decode_with_opencv(video_path: str) -> bool:
    result = _run_with_deadline(
        lambda: _probe_opencv_decode(video_path),
        _OPENCV_PROBE_TIMEOUT_SECONDS,
        f"OpenCV decode probe for {os.path.basename(video_path)}",
    )
    return bool(result[0]) if result else False


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


# Upper bound on concurrent ffmpeg frame-extraction subprocesses.
_MAX_EXTRACTION_WORKERS = 4

# While a batch extraction is in flight, emit a heartbeat no more often than
# this. Without it the whole stage is silent and indistinguishable from a hang.
_EXTRACTION_HEARTBEAT_SECONDS = 5.0


def _extract_single_frame_to_temp(
    video_path: str,
    ts: float,
    tmp_path: str,
    per_frame_timeout: int,
) -> Dict[str, Any]:
    """Decode the single frame at *ts* into *tmp_path*; return a diagnostics dict.

    ``-ss`` is placed **before** ``-i`` so ffmpeg seeks to the nearest preceding
    keyframe and decodes only forward from there.  This is what keeps extraction
    proportional to the number of frames requested rather than to the length of
    the file: a ``select=`` filter has no seek at all and forces a full linear
    decode of the source, which on a 4K AV1 video costs ~150s for ten frames.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, float(ts)):.6f}", "-i", video_path,
        "-frames:v", "1", "-q:v", "2", "-y", tmp_path,
    ]
    diag: Dict[str, Any] = {
        "timestamp": round(float(ts), 6),
        "ok": False,
        "timeout": False,
        "returncode": None,
        "stderr": "",
        "elapsed": 0.0,
    }
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=per_frame_timeout)
    except subprocess.TimeoutExpired:
        diag["timeout"] = True
        diag["stderr"] = f"ffmpeg frame timeout after {per_frame_timeout}s"
        diag["elapsed"] = time.perf_counter() - started
        return diag
    except Exception as exc:
        diag["stderr"] = f"ffmpeg frame exception: {exc}"
        diag["elapsed"] = time.perf_counter() - started
        return diag

    diag["elapsed"] = time.perf_counter() - started
    diag["returncode"] = proc.returncode
    diag["stderr"] = proc.stderr.decode("utf-8", errors="ignore").strip()
    diag["ok"] = proc.returncode == 0 and os.path.exists(tmp_path)
    return diag


def _extract_batch_frames_ffmpeg(
    video_path: str,
    timestamps_seconds: List[float],
    timeout: int = 120,
    return_diagnostics: bool = False,
) -> Union[List[Optional[np.ndarray]], Tuple[List[Optional[np.ndarray]], Dict[str, Any]]]:
    """Extract the frames at *timestamps_seconds* using bounded parallel fast-seek.

    One short-lived ffmpeg process per timestamp, at most
    ``_MAX_EXTRACTION_WORKERS`` at a time, each using keyframe fast-seek.

    Progress is logged as frames land so that this stage, which can be long,
    is visibly alive.
    """
    n = len(timestamps_seconds)
    diag: Dict[str, Any] = {
        "mode": "parallel_fast_seek",
        "requested_timestamps": [round(float(t), 6) for t in timestamps_seconds],
        "timeout_seconds": timeout,
        "timeout": False,
        "returncode": 0,
        "stderr": "",
        "raw_images_found": 0,
        "frames_decoded": 0,
        "error_code": None,
        "per_frame": [],
        "elapsed": 0.0,
    }
    if not n:
        empty: List[Optional[np.ndarray]] = []
        return (empty, diag) if return_diagnostics else empty

    results: List[Optional[np.ndarray]] = [None] * n
    per_frame: List[Optional[Dict[str, Any]]] = [None] * n
    # Each frame gets the full budget: the timeout bounds a single fast-seek
    # decode, not the batch, and slicing it across frames made large batches
    # spuriously time out.
    per_frame_timeout = max(15, int(timeout))
    started = time.perf_counter()
    completed = 0
    last_heartbeat = started

    logger.info(
        "[EXTRACT] Extracting %d frame(s) from %s (fast-seek, %d workers, %ds/frame budget)…",
        n, os.path.basename(video_path), min(_MAX_EXTRACTION_WORKERS, n), per_frame_timeout,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = [os.path.join(tmpdir, f"frame_{i:04d}.jpg") for i in range(n)]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_MAX_EXTRACTION_WORKERS, n)
        ) as executor:
            futures = {
                executor.submit(
                    _extract_single_frame_to_temp,
                    video_path, ts, tmp_paths[i], per_frame_timeout,
                ): i
                for i, ts in enumerate(timestamps_seconds)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    fdiag = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    fdiag = {
                        "timestamp": round(float(timestamps_seconds[idx]), 6),
                        "ok": False, "timeout": False, "returncode": None,
                        "stderr": f"worker exception: {exc}", "elapsed": 0.0,
                    }
                per_frame[idx] = fdiag
                if fdiag.get("timeout"):
                    diag["timeout"] = True
                if fdiag.get("ok"):
                    frame = cv2.imread(tmp_paths[idx])
                    if frame is not None and frame.size > 0:
                        results[idx] = frame
                completed += 1
                now = time.perf_counter()
                if now - last_heartbeat >= _EXTRACTION_HEARTBEAT_SECONDS or completed == n:
                    logger.info(
                        "[EXTRACT] %d/%d frame(s) done (%.1fs elapsed)",
                        completed, n, now - started,
                    )
                    last_heartbeat = now

    diag["per_frame"] = [d for d in per_frame if d is not None]
    diag["frames_decoded"] = sum(1 for f in results if f is not None)
    diag["raw_images_found"] = sum(1 for d in diag["per_frame"] if d.get("ok"))
    diag["elapsed"] = time.perf_counter() - started
    if diag["frames_decoded"] == 0:
        diag["error_code"] = (
            "ffmpeg_batch_timeout" if diag["timeout"] else "ffmpeg_batch_no_output"
        )
        diag["stderr"] = "; ".join(
            d.get("stderr", "") for d in diag["per_frame"] if d.get("stderr")
        )[:2000]

    logger.info(
        "[EXTRACT] Recovered %d/%d frame(s) in %.1fs.",
        diag["frames_decoded"], n, diag["elapsed"],
    )
    return (results, diag) if return_diagnostics else results


# ---------------------------------------------------------------------------
# 1b — Shared VideoCapture session
# ---------------------------------------------------------------------------

# Maximum distance, in frames, that :meth:`_CachedCapture.read_frames` will cover by
# decoding forward instead of issuing a fresh ``CAP_PROP_POS_FRAMES`` seek.
#
# A seek to an arbitrary position in a long-GOP file forces the decoder to restart at
# the preceding keyframe, so it costs far more than a sequential ``read()``. Measured
# on a real 3840x2048 H.264 source: ~3.16 s per random seek versus ~0.02 s per
# sequential frame — break-even around 160 frames. The threshold is set well below
# that so the behaviour stays predictable: it comfortably covers a secondary window
# (the widest retry step is 13, i.e. a 27-frame span) while never bridging two main
# frames, which sit hundreds to thousands of frames apart in real sources.
_SEQUENTIAL_READ_MAX_GAP = 48


class _CachedCapture:
    """Wrapper sobre cv2.VideoCapture que gestiona el ciclo de vida de forma segura."""

    def __init__(self, video_path: str) -> None:
        self._path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self.can_decode: bool = False
        self.total_frames: int = 0
        self.fps: float = 0.0
        # Index the next ``read()`` will return, or None when unknown (nothing
        # decoded yet, or a read failed and left the decoder at an unknown offset).
        self._pos: Optional[int] = None

        # Opening and decoding the first frame is the one unbounded call in the
        # extraction path; give it the same deadline as the standalone probe so
        # an undecodable file degrades to the ffmpeg path instead of wedging.
        opened = _run_with_deadline(
            lambda: self._open(video_path),
            _OPENCV_PROBE_TIMEOUT_SECONDS,
            f"OpenCV capture open for {os.path.basename(video_path)}",
        )
        if opened is None:
            # Abandoned: the worker may still hold the capture, so drop our
            # reference and let the ffmpeg fallback take over.
            self._cap = None
            return
        self._cap, self.can_decode, self.total_frames, self.fps = opened
        # _open() decodes one frame to prove decodability, then rewinds to 0.
        self._pos = 0 if self.can_decode else None

    @staticmethod
    def _open(video_path: str) -> Tuple[Any, bool, int, float]:
        cap = cv2.VideoCapture(video_path)
        can_decode = cap.isOpened()
        if can_decode:
            ret, _ = cap.read()
            can_decode = bool(ret)
            if can_decode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if can_decode else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if can_decode else 0.0
        return cap, can_decode, total_frames, fps

    def _advance_to(self, target: int) -> bool:
        """Position the decoder so the next ``read()`` returns frame *target*.

        Reads forward when *target* is already close ahead, and only falls back to a
        seek when it is behind us or far enough away that decoding forward would cost
        more than restarting at a keyframe.
        """
        if self._cap is None:
            return False
        if self._pos is not None and 0 <= target - self._pos <= _SEQUENTIAL_READ_MAX_GAP:
            while self._pos < target:
                ok, _ = self._cap.read()
                if not ok:
                    self._pos = None
                    return False
                self._pos += 1
            return True
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._pos = target
        return True

    def read_frames(self, positions: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """Decode every position in *positions*, seeking once per nearby cluster.

        Requesting the whole batch at once is what makes this cheap: positions are
        sorted, grouped into clusters no more than ``_SEQUENTIAL_READ_MAX_GAP`` apart,
        and each cluster is served by a single seek followed by sequential reads
        (intermediate frames are decoded and dropped). Frames returned are exactly the
        frames the per-position seek path returned, so callers are unaffected.
        """
        wanted = sorted({int(p) for p in positions})
        results: Dict[int, Optional[np.ndarray]] = {p: None for p in wanted}
        if not self.can_decode or self._cap is None or not wanted:
            return {int(p): results.get(int(p)) for p in positions}

        idx = 0
        while idx < len(wanted):
            cluster = [wanted[idx]]
            nxt = idx + 1
            while nxt < len(wanted) and wanted[nxt] - cluster[-1] <= _SEQUENTIAL_READ_MAX_GAP:
                cluster.append(wanted[nxt])
                nxt += 1

            if self._advance_to(cluster[0]):
                targets = set(cluster)
                cursor = cluster[0]
                last = cluster[-1]
                while cursor <= last:
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        self._pos = None
                        break
                    self._pos = cursor + 1
                    if cursor in targets:
                        results[cursor] = frame
                    cursor += 1
            idx = nxt

        return {int(p): results.get(int(p)) for p in positions}

    def read_frame(self, position: int) -> Optional[np.ndarray]:
        return self.read_frames([position]).get(int(position))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._pos = None

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

    _meta = probe_video_stream(video_path)
    _codec = _meta.get("codec_name", "").lower()
    _is_av1_input = _codec in _AV1_CODEC_NAMES
    if _is_av1_input:
        logger.warning(
            "AV1 codec detected — hardware decode not supported by OpenCV on this platform; "
            "re-encoding to H.264 for analysis (temp file, not stored)."
        )

    def _build_cmd(for_encoder: Optional[str]) -> Tuple[List[str], str]:
        """Assemble the normalization command for *for_encoder* (None = auto-detect).

        Built in one place so the hardware and fallback commands are always
        internally consistent: patching ``-c:v`` alone would leave a VAAPI device
        and an ``hwupload`` filter attached to a software encoder.
        """
        in_args, vf, enc_args, chosen = build_encoder_command_parts(
            for_encoder,
            software_quality=["-preset", "veryfast", "-crf", "23"],
        )
        built = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        built.extend(in_args)
        # -hwaccel auto is skipped for AV1 (OpenCV cannot consume it) and whenever a
        # filter chain is present: a hardware-decoded frame cannot be read by the
        # software `format` filter that precedes hwupload.
        if not _is_av1_input and not vf:
            built.extend(["-hwaccel", "auto"])
        built.extend(["-i", video_path])
        if vf:
            built.extend(["-vf", vf])
        built.extend(enc_args)
        built.extend(["-an", compat_path])
        return built, chosen

    # Resolved here rather than inside the builder so this module keeps ownership of
    # which encoder it asks for; the builder only assembles the arguments for it.
    cmd, encoder = _build_cmd(detect_ffmpeg_h264_encoder())

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
        if proc.returncode != 0 and encoder != "libx264":
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            if _is_hardware_encoder_runtime_failure(stderr, encoder):
                fallback_cmd, _ = _build_cmd("libx264")
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
    if _owns_cap:
        cap_session = _CachedCapture(video_path)
    try:
        video_path_procesado = video_path

        can_decode_opencv = cap_session.can_decode
        metadata = probe_video_stream(video_path)
        if not can_decode_opencv:
            _input_codec = metadata.get("codec_name", "").lower()
            if _input_codec in _AV1_CODEC_NAMES:
                logger.warning(
                    "AV1 video — OpenCV cannot decode AV1 on this platform; "
                    "using ffmpeg subprocess for frame extraction."
                )
        total_frames = cap_session.total_frames if can_decode_opencv else int(metadata.get("total_frames", 0))
        fps = cap_session.fps if can_decode_opencv else float(metadata.get("fps", 0.0))
        if total_frames <= 0:
            total_frames = int(metadata.get("total_frames", 0))
        if fps <= 0:
            fps = float(metadata.get("fps", 0.0))
        duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else float(metadata.get("duration", 0.0))
        if total_frames <= 0:
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
            # One batched read: read_frames() sorts and clusters the positions so
            # nearby ones share a seek instead of paying one seek each.
            decoded = (
                cap_session.read_frames(list(active_positions))
                if can_decode_opencv
                else {}
            )
            for idx_pos, frame_pos in enumerate(active_positions):
                frame: Optional[np.ndarray] = decoded.get(int(frame_pos))
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
    finally:
        if _owns_cap:
            cap_session.release()


def extract_secondary_frames(
    video_path: str,
    frame_positions: List[int],
    total_frames: int,
    paso_frames: int = 5,
    cap_session: Optional[_CachedCapture] = None,
    known_frames: Optional[Dict[int, np.ndarray]] = None,
) -> List[List[Dict[str, Any]]]:
    """Extrae secuencias temporales 3-frame alrededor de cada posición principal.

    *known_frames* maps an absolute frame position to a frame the caller has already
    decoded. Each window is centred on its main frame, so without this every main
    frame would be decoded a second time here purely to sit at the centre of its own
    window.
    """
    secuencias: List[List[Dict[str, Any]]] = []
    metadata = probe_video_stream(video_path)
    fps = float(metadata.get("fps", 0.0))
    duration = float(metadata.get("duration", 0.0))
    known = {int(k): v for k, v in (known_frames or {}).items() if v is not None}

    _owns_cap = cap_session is None
    if cap_session is None:
        cap_session = _CachedCapture(video_path)
    cap_ok = cap_session.can_decode

    window_sets: List[List[int]] = []
    for pos in frame_positions:
        posiciones_ventana = sorted({
            max(0, pos - paso_frames),
            pos,
            min(total_frames - 1, pos + paso_frames),
        })
        window_sets.append(posiciones_ventana)

    # Decode every window of every sequence in one batched call, so read_frames()
    # can sort and cluster globally instead of re-seeking per position.
    needed: List[int] = sorted(
        {p for window in window_sets for p in window if p not in known}
    )
    decoded: Dict[int, Optional[np.ndarray]] = (
        cap_session.read_frames(needed) if (cap_ok and needed) else {}
    )

    all_seq_frames: List[List[Optional[np.ndarray]]] = []
    ffmpeg_needed_positions: List[Tuple[int, int, float]] = []  # (seq_idx, win_idx, timestamp)

    for seq_idx, posiciones_ventana in enumerate(window_sets):
        seq_frames: List[Optional[np.ndarray]] = []
        for win_idx, p in enumerate(posiciones_ventana):
            frame: Optional[np.ndarray] = known.get(p)
            if frame is None:
                frame = decoded.get(p)
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
