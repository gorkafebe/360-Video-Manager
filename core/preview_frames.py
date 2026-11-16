"""Preview frame extraction for UI and thumbnails.

This module is *separate* from the detector's analysis frame extraction.
Preview frames are lower-stakes: we extract a small number of evenly-spaced
frames and save them as JPEG files for display in the GUI or as thumbnails.

The detector pipeline does its own frame extraction internally — these
preview frames must not be confused with or substituted for analysis frames.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def extract_preview_frames(
    video_path: str,
    num_frames: int = 5,
    output_dir: Optional[str] = None,
    prefix: str = "preview",
    padding_ratio: float = 0.05,
) -> List[str]:
    """Extract *num_frames* evenly-spaced preview frames from *video_path*.

    The extracted frames are saved as JPEG files inside *output_dir* (or a
    ``previews/`` sub-directory next to the video when *output_dir* is
    ``None``) and their paths are returned.

    Args:
        video_path: Path to the source video.
        num_frames: How many frames to extract (default: 5).
        output_dir: Directory to save preview images.  Created if needed.
        prefix: Filename prefix for saved images.
        padding_ratio: Fraction of video duration to skip at the start and end
            to avoid black/credits frames.

    Returns:
        List of absolute paths to the saved preview images.
        Returns an empty list if extraction fails.
    """
    if not os.path.exists(video_path):
        logger.warning("Preview extraction: video not found: %s", video_path)
        return []

    if output_dir is None:
        video_dir = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, "previews", video_stem)

    os.makedirs(output_dir, exist_ok=True)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Preview extraction: cannot open %s", video_path)
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total <= 0:
            logger.warning("Preview extraction: could not determine frame count for %s", video_path)
            cap.release()
            return []

        padding = max(0, int(total * padding_ratio))
        start = padding
        end = max(start + 1, total - padding)
        usable = end - start

        count = min(num_frames, usable)
        if count <= 0:
            cap.release()
            return []

        positions = [
            start + int(i * usable / count) if count > 1 else start
            for i in range(count)
        ]

        saved_paths: List[str] = []
        for i, pos in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            ts = pos / fps
            fname = f"{prefix}_frame_{i + 1:03d}_t{int(ts):04d}s.jpg"
            fpath = os.path.join(output_dir, fname)

            ok = cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                saved_paths.append(fpath)
            else:
                logger.warning("Preview extraction: could not write %s", fpath)

        cap.release()
        logger.info(
            "Extracted %d/%d preview frames to %s", len(saved_paths), count, output_dir
        )
        return saved_paths

    except Exception as exc:
        logger.exception("Preview extraction failed for %s: %s", video_path, exc)
        return []
