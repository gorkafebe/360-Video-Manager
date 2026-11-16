"""Frame preprocessing utilities for the detector pipeline."""

from typing import Optional

import cv2
import numpy as np


def downscale_to_max_edge(frame: np.ndarray, max_edge: Optional[int]) -> np.ndarray:
    """Return *frame* shrunk so its longer side is at most *max_edge* pixels.

    Used to give dense optical flow, ORB, and homography a bounded working
    resolution. Dense-flow cost grows ~quadratically with pixel count, and
    360 sources are commonly 4K–8K (a single TV-L1 flow at 3840×2048 exceeds
    four minutes on modest hardware), so an uncapped native-resolution flow is
    the difference between seconds and hours per run.

    Downscaling is safe for what the layout logic actually consumes: per-region
    motion *angles* are invariant to uniform scaling, and the fraction-based
    validity thresholds (active-pixel ratio, concentration) are scale-free. Only
    absolute pixel-magnitude thresholds shift — which is precisely why callers
    should measure motion at one *fixed* working resolution rather than at each
    source's native size, so those thresholds mean the same thing for every video.

    Returns the frame unchanged when *max_edge* is falsy or already large enough.
    INTER_AREA is used because it is the correct (anti-aliasing) filter for
    shrinking and it improves flow coherence versus naive subsampling.
    """
    if not max_edge or max_edge <= 0 or frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= int(max_edge):
        return frame
    scale = float(max_edge) / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def prepare_frame_for_flow(
    frame: np.ndarray,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
) -> np.ndarray:
    """Convert *frame* to grayscale and apply Gaussian blur, ready for optical flow."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(1, int(gaussian_kernel_size))
    if k % 2 == 0:
        k += 1
    sigma = max(float(gaussian_sigma), 0.0)
    return cv2.GaussianBlur(gray, (k, k), sigma)


# Median luminance (8-bit) below which a frame gets local-contrast normalisation
# before line detection. Measured on real 360 sources: night footage sits at a
# median of ~15 while ordinary daylight/overcast footage sits at 108-126, so the
# separation is wide and normal footage is left bit-identical.
_LOW_LIGHT_MEDIAN = 64.0
# Contrast-limited, so flat regions cannot have their noise amplified without
# bound — the "contrast-limited" half of CLAHE is what makes it safe to apply to
# a frame that may be partly featureless.
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = (8, 8)


def is_low_light(gray: np.ndarray) -> bool:
    """Whether *gray* is dark enough to need local-contrast normalisation.

    Uses the median rather than the mean so a small bright region (an aurora, a
    light source) does not mask an otherwise black frame.
    """
    if gray is None or gray.size == 0:
        return False
    return float(np.median(gray)) < _LOW_LIGHT_MEDIAN


def prepare_frame_for_line_detection(
    frame: np.ndarray,
    blur_kernel: int = 3,
    enhance_low_light: bool = True,
) -> np.ndarray:
    """Convert *frame* to grayscale and apply a light Gaussian blur for line detection.

    Dark frames additionally get CLAHE first. The seam detectors downstream
    (Canny/Hough/LSD) threshold on *absolute* gradient magnitude, so a frame whose
    whole scene occupies a few dozen 8-bit code values carries real edges that
    those thresholds cannot see: on a real night-sky 360 video, 0 of 6 non-black
    frames yielded a seam, which starved the pipeline's frame-sufficiency gate
    (0.83 discard ratio) even though the face-continuity evidence computed from
    the very same frames was consistent and correct.

    The normalisation is *conditional* precisely so it cannot perturb the
    tuned seam thresholds on ordinary footage: above ``_LOW_LIGHT_MEDIAN`` the
    output is byte-for-byte unchanged.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if enhance_low_light and is_low_light(gray):
        gray = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID
        ).apply(gray)
    return cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
