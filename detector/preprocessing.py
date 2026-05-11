"""Frame preprocessing utilities for the detector pipeline."""

import cv2
import numpy as np


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


def prepare_frame_for_line_detection(
    frame: np.ndarray,
    blur_kernel: int = 3,
) -> np.ndarray:
    """Convert *frame* to grayscale and apply a light Gaussian blur for line detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
