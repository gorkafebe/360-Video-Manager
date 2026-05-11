"""Optical-flow motion analysis for the detector pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .preprocessing import prepare_frame_for_flow


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
    flow_algorithm: str = "farneback",
) -> np.ndarray:
    """Compute dense optical flow between *frame1* and *frame2*.

    Supports ``"farneback"`` (default) and ``"dis"`` algorithms.
    """
    gray1 = prepare_frame_for_flow(frame1, gaussian_kernel_size, gaussian_sigma)
    gray2 = prepare_frame_for_flow(frame2, gaussian_kernel_size, gaussian_sigma)
    if flow_algorithm == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis.calc(gray1, gray2, None)
    return cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def split_into_regions(
    frame: np.ndarray,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Split *frame* into a 2×3 grid and return (row, col, y0, y1, x0, x1) tuples."""
    height, width = frame.shape[:2]
    half_h = height // 2
    third_w = width // 3
    region_bounds: List[Tuple[int, int, int, int, int, int]] = []
    for row in range(2):
        y0 = 0 if row == 0 else half_h
        y1 = half_h if row == 0 else height
        for col in range(3):
            x0 = col * third_w
            x1 = (col + 1) * third_w if col < 2 else width
            region_bounds.append((row, col, y0, y1, x0, x1))
    return region_bounds


def compute_region_motion(
    flow: np.ndarray,
    region: Tuple[int, int, int, int, int, int],
    umbral_magnitud: float = 1.0,
    proporcion_minima_pixeles: float = 0.01,
) -> Dict[str, Any]:
    """Aggregate motion direction and magnitude for a single *region* of *flow*."""
    _, _, y0, y1, x0, x1 = region
    region_flow = flow[y0:y1, x0:x1]
    if region_flow.size == 0:
        return {"valid": False, "angle": 0.0, "magnitude": 0.0, "concentration": 0.0, "active_ratio": 0.0}

    mag, ang = cv2.cartToPolar(region_flow[..., 0], region_flow[..., 1])
    mask = mag > float(umbral_magnitud)
    min_count = max(int(mag.size * float(proporcion_minima_pixeles)), 1)
    active_count = int(mask.sum())
    if active_count < min_count:
        return {
            "valid": False, "angle": 0.0, "magnitude": 0.0, "concentration": 0.0,
            "active_ratio": active_count / float(mag.size),
        }

    m = mag[mask]
    a = ang[mask]
    w = np.maximum(m, 1e-6)
    sx = float(np.sum(np.cos(a) * w))
    sy = float(np.sum(np.sin(a) * w))
    sw = float(np.sum(w))
    if sw <= 0:
        return {
            "valid": False, "angle": 0.0, "magnitude": 0.0, "concentration": 0.0,
            "active_ratio": active_count / float(mag.size),
        }

    angle = float(np.arctan2(sy, sx))
    concentration = float(np.sqrt(sx * sx + sy * sy) / sw)
    return {
        "valid": True,
        "angle": angle,
        "magnitude": float(np.mean(m)),
        "concentration": concentration,
        "active_ratio": active_count / float(mag.size),
    }


def compute_region_affine_angles(
    frame1: np.ndarray,
    frame2: np.ndarray,
    region_bounds: List[Tuple[int, int, int, int, int, int]],
    min_inliers: int = 6,
    orb_nfeatures: int = 300,
) -> Dict[tuple, Optional[float]]:
    """Estimate per-region rotation angle using ORB features + partial affine RANSAC."""
    result: Dict[tuple, Optional[float]] = {}
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for row, col, y0, y1, x0, x1 in region_bounds:
        try:
            roi1 = frame1[y0:y1, x0:x1]
            roi2 = frame2[y0:y1, x0:x1]
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                result[(row, col)] = None
                continue
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)
            if len(matches) < min_inliers:
                result[(row, col)] = None
                continue
            src_pts = np.array(
                [kp1[m.queryIdx].pt for m in matches], dtype=np.float32
            ).reshape(-1, 1, 2)
            dst_pts = np.array(
                [kp2[m.trainIdx].pt for m in matches], dtype=np.float32
            ).reshape(-1, 1, 2)
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if M is None:
                result[(row, col)] = None
                continue
            inlier_count = int(np.sum(inliers)) if inliers is not None else 0
            if inlier_count < min_inliers:
                result[(row, col)] = None
                continue
            result[(row, col)] = float(np.arctan2(M[1, 0], M[0, 0]))
        except Exception:
            result[(row, col)] = None
    return result
