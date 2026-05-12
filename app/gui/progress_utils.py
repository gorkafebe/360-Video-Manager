"""Helpers for GUI progress tracking and yt-dlp progress payload parsing."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


DOWNLOAD_PROGRESS_UPDATE_MS = 120


def clamp_progress(value: float) -> float:
    """Clamp a progress value to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def as_positive_float(value: Any) -> Optional[float]:
    """Return a finite positive float, or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def extract_download_progress_fraction(payload: Dict[str, Any]) -> Optional[float]:
    """Extract download progress from yt-dlp hook payload."""
    downloaded = as_positive_float(payload.get("downloaded_bytes"))
    total = as_positive_float(payload.get("total_bytes"))
    if total is None:
        total = as_positive_float(payload.get("total_bytes_estimate"))
    if downloaded is not None and total is not None:
        return clamp_progress(downloaded / total)

    pct_raw = str(payload.get("_percent_str") or "").strip()
    if pct_raw.endswith("%"):
        pct_raw = pct_raw[:-1]
    try:
        return clamp_progress(float(pct_raw.strip()) / 100.0)
    except ValueError:
        return None


def compute_progress_update_delay_ms(
    *,
    last_update_monotonic: float,
    now_monotonic: float,
    min_interval_ms: int = DOWNLOAD_PROGRESS_UPDATE_MS,
) -> int:
    """Return coalesced scheduling delay in milliseconds."""
    elapsed_ms = int((now_monotonic - last_update_monotonic) * 1000)
    return max(0, min_interval_ms - elapsed_ms)
