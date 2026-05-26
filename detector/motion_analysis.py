"""Optical-flow motion analysis for the detector pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .preprocessing import prepare_frame_for_flow


_OPENCV_CAPABILITIES_CACHE: Optional[Dict[str, bool]] = None
_FLOW_CAPABILITY_KEY_BY_ALGO = {
    "dis": "has_dis",
    "tvl1": "has_tvl1",
    "deepflow": "has_deepflow",
    "pcaflow": "has_pcaflow",
    "sparse_to_dense": "has_sparse_to_dense",
}


def _build_opencv_capabilities() -> Dict[str, bool]:
    optflow = getattr(cv2, "optflow", None)
    return {
        "has_optflow_module": optflow is not None,
        "has_dis": hasattr(cv2, "DISOpticalFlow_create"),
        "has_tvl1": bool(optflow is not None and hasattr(optflow, "createOptFlow_DualTVL1")),
        "has_deepflow": bool(optflow is not None and hasattr(optflow, "createOptFlow_DeepFlow")),
        "has_pcaflow": bool(optflow is not None and hasattr(optflow, "createOptFlow_PCAFlow")),
        "has_sparse_to_dense": bool(optflow is not None and hasattr(optflow, "createOptFlow_SparseToDense")),
        "has_variational_refinement": bool(optflow is not None and hasattr(optflow, "createVariationalRefinement")),
        "has_usac_magsac": hasattr(cv2, "USAC_MAGSAC"),
    }


def get_opencv_capabilities(force_refresh: bool = False) -> Dict[str, bool]:
    global _OPENCV_CAPABILITIES_CACHE
    if _OPENCV_CAPABILITIES_CACHE is None or force_refresh:
        _OPENCV_CAPABILITIES_CACHE = _build_opencv_capabilities()
    return dict(_OPENCV_CAPABILITIES_CACHE)


def normalize_flow_algorithm_name(flow_algorithm: str) -> str:
    raw = str(flow_algorithm or "farneback").strip().lower()
    aliases = {
        "sparse-to-dense": "sparse_to_dense",
        "sparse2dense": "sparse_to_dense",
        "dual_tvl1": "tvl1",
        "dualtvl1": "tvl1",
    }
    return aliases.get(raw, raw)


def is_flow_algorithm_available(
    flow_algorithm: str,
    capabilities: Optional[Dict[str, bool]] = None,
) -> bool:
    algo = normalize_flow_algorithm_name(flow_algorithm)
    if algo == "farneback":
        return True
    caps = capabilities or get_opencv_capabilities()
    capability_key = _FLOW_CAPABILITY_KEY_BY_ALGO.get(algo)
    if capability_key is None:
        return False
    return bool(caps.get(capability_key, False))


def build_flow_fallback_chain(
    preferred_algorithm: str,
    capabilities: Optional[Dict[str, bool]] = None,
    profile: str = "baseline",
    tier_b_algorithms: Optional[List[str]] = None,
    tier_c_algorithms: Optional[List[str]] = None,
) -> List[str]:
    caps = capabilities or get_opencv_capabilities()
    profile_key = str(profile or "baseline").strip().lower()
    tier_b = [normalize_flow_algorithm_name(a) for a in (tier_b_algorithms or ["tvl1", "dis"])]
    tier_c = [normalize_flow_algorithm_name(a) for a in (tier_c_algorithms or ["deepflow", "pcaflow", "sparse_to_dense"])]
    preferred = normalize_flow_algorithm_name(preferred_algorithm)

    if preferred == "farneback":
        return ["farneback"]

    ordered: List[str] = [preferred]
    if profile_key in {"robust", "high_accuracy"}:
        ordered.extend(tier_b)
    elif profile_key == "baseline":
        ordered.append("dis")
    ordered.extend(tier_c)
    ordered.append("farneback")

    deduped: List[str] = []
    for algo in ordered:
        if algo in deduped:
            continue
        deduped.append(algo)

    available = [algo for algo in deduped if is_flow_algorithm_available(algo, capabilities=caps)]
    if "farneback" not in available:
        available.append("farneback")
    return available


def _compute_farneback(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
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


def _compute_contrib_flow(gray1: np.ndarray, gray2: np.ndarray, flow_algorithm: str) -> Optional[np.ndarray]:
    caps = get_opencv_capabilities()
    optflow = getattr(cv2, "optflow", None)
    if optflow is None:
        return None
    try:
        if flow_algorithm == "tvl1" and caps["has_tvl1"]:
            algo = optflow.createOptFlow_DualTVL1()
            return algo.calc(gray1, gray2, None)
        if flow_algorithm == "deepflow" and caps["has_deepflow"]:
            algo = optflow.createOptFlow_DeepFlow()
            return algo.calc(gray1, gray2, None)
        if flow_algorithm == "pcaflow" and caps["has_pcaflow"]:
            algo = optflow.createOptFlow_PCAFlow()
            return algo.calc(gray1, gray2, None)
        if flow_algorithm == "sparse_to_dense" and caps["has_sparse_to_dense"]:
            algo = optflow.createOptFlow_SparseToDense()
            return algo.calc(gray1, gray2, None)
    except Exception:
        return None
    return None


def _apply_variational_refinement(
    gray1: np.ndarray,
    gray2: np.ndarray,
    flow: np.ndarray,
    enable_refinement: bool,
) -> np.ndarray:
    if not enable_refinement:
        return flow
    caps = get_opencv_capabilities()
    if not caps["has_variational_refinement"]:
        return flow
    optflow = getattr(cv2, "optflow", None)
    if optflow is None:
        return flow
    try:
        refiner = optflow.createVariationalRefinement()
        return refiner.calc(gray1, gray2, flow)
    except Exception:
        return flow


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
    flow_algorithm: str = "farneback",
    enable_refinement: bool = False,
) -> np.ndarray:
    """Compute dense optical flow between *frame1* and *frame2* with safe fallbacks."""
    gray1 = prepare_frame_for_flow(frame1, gaussian_kernel_size, gaussian_sigma)
    gray2 = prepare_frame_for_flow(frame2, gaussian_kernel_size, gaussian_sigma)
    caps = get_opencv_capabilities()
    flow_chain = build_flow_fallback_chain(
        preferred_algorithm=flow_algorithm,
        capabilities=caps,
        profile="baseline",
    )

    flow: Optional[np.ndarray] = None
    for candidate in flow_chain:
        try:
            if candidate == "farneback":
                flow = _compute_farneback(gray1, gray2)
            elif candidate == "dis" and caps.get("has_dis", False):
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = dis.calc(gray1, gray2, None)
            else:
                flow = _compute_contrib_flow(gray1, gray2, candidate)
        except Exception:
            flow = None
        if flow is not None:
            break

    if flow is None:
        flow = _compute_farneback(gray1, gray2)

    return _apply_variational_refinement(gray1, gray2, flow, enable_refinement=enable_refinement)


def compute_forward_backward_consistency_mask(
    frame1: np.ndarray,
    frame2: np.ndarray,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.2,
    flow_algorithm: str = "farneback",
    enable_refinement: bool = False,
    fb_threshold: float = 1.5,
) -> np.ndarray:
    """Return per-pixel mask where forward/backward flow vectors are consistent."""
    flow_fwd = compute_optical_flow(
        frame1,
        frame2,
        gaussian_kernel_size=gaussian_kernel_size,
        gaussian_sigma=gaussian_sigma,
        flow_algorithm=flow_algorithm,
        enable_refinement=enable_refinement,
    )
    flow_bwd = compute_optical_flow(
        frame2,
        frame1,
        gaussian_kernel_size=gaussian_kernel_size,
        gaussian_sigma=gaussian_sigma,
        flow_algorithm=flow_algorithm,
        enable_refinement=enable_refinement,
    )
    fb_err = np.linalg.norm(flow_fwd + flow_bwd, axis=2)
    return fb_err <= float(max(0.1, fb_threshold))


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
    valid_mask: Optional[np.ndarray] = None,
    outlier_percentile: float = 97.0,
) -> Dict[str, Any]:
    """Aggregate motion direction and magnitude for a single *region* of *flow*."""
    _, _, y0, y1, x0, x1 = region
    region_flow = flow[y0:y1, x0:x1]
    if region_flow.size == 0:
        return {"valid": False, "angle": 0.0, "magnitude": 0.0, "concentration": 0.0, "active_ratio": 0.0}

    mag, ang = cv2.cartToPolar(region_flow[..., 0], region_flow[..., 1])
    motion_mask = mag > float(umbral_magnitud)
    if valid_mask is not None:
        region_mask = valid_mask[y0:y1, x0:x1]
        if region_mask.shape == motion_mask.shape:
            motion_mask = np.logical_and(motion_mask, region_mask)

    active_count = int(motion_mask.sum())
    min_count = max(int(mag.size * float(proporcion_minima_pixeles)), 1)
    if active_count < min_count:
        return {
            "valid": False,
            "angle": 0.0,
            "magnitude": 0.0,
            "concentration": 0.0,
            "active_ratio": active_count / float(mag.size),
        }

    m = mag[motion_mask]
    a = ang[motion_mask]
    if m.size == 0:
        return {
            "valid": False,
            "angle": 0.0,
            "magnitude": 0.0,
            "concentration": 0.0,
            "active_ratio": 0.0,
        }

    cap = float(np.percentile(m, float(np.clip(outlier_percentile, 85.0, 99.5))))
    clipped = np.minimum(m, cap)
    w = np.maximum(clipped, 1e-6)

    sx = float(np.sum(np.cos(a) * w))
    sy = float(np.sum(np.sin(a) * w))
    sw = float(np.sum(w))
    if sw <= 0:
        return {
            "valid": False,
            "angle": 0.0,
            "magnitude": 0.0,
            "concentration": 0.0,
            "active_ratio": active_count / float(mag.size),
        }

    angle = float(np.arctan2(sy, sx))
    concentration = float(np.sqrt(sx * sx + sy * sy) / sw)
    return {
        "valid": True,
        "angle": angle,
        "magnitude": float(np.mean(clipped)),
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


def compute_global_geometry_evidence(
    frame1: np.ndarray,
    frame2: np.ndarray,
    min_inliers: int = 12,
    orb_nfeatures: int = 500,
) -> Dict[str, Any]:
    """Compute robust global geometric-consistency evidence from matched keypoints."""
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(des1) < min_inliers or len(des2) < min_inliers:
            return {"valid": False, "inlier_ratio": 0.0, "reprojection_error": 999.0, "quality": 0.0}

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < min_inliers:
            return {"valid": False, "inlier_ratio": 0.0, "reprojection_error": 999.0, "quality": 0.0}
        matches = sorted(matches, key=lambda m: m.distance)
        src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        caps = get_opencv_capabilities()
        method = cv2.USAC_MAGSAC if bool(caps.get("has_usac_magsac", False)) else cv2.RANSAC
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, method=method, ransacReprojThreshold=3.0)
        if H is None or inlier_mask is None:
            return {"valid": False, "inlier_ratio": 0.0, "reprojection_error": 999.0, "quality": 0.0}

        inlier_mask_bool = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.sum(inlier_mask_bool))
        if inlier_count < min_inliers:
            return {"valid": False, "inlier_ratio": 0.0, "reprojection_error": 999.0, "quality": 0.0}

        src_in = src_pts[inlier_mask_bool].reshape(-1, 1, 2)
        dst_in = dst_pts[inlier_mask_bool].reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src_in, H).reshape(-1, 2)
        target = dst_in.reshape(-1, 2)
        reproj_error = float(np.mean(np.linalg.norm(projected - target, axis=1)))
        inlier_ratio = float(inlier_count / max(1, len(matches)))
        quality = float(
            np.clip((inlier_ratio - 0.25) / 0.55, 0.0, 1.0) * 0.70
            + np.clip((4.0 - reproj_error) / 4.0, 0.0, 1.0) * 0.30
        )
        return {
            "valid": True,
            "inlier_ratio": inlier_ratio,
            "reprojection_error": reproj_error,
            "quality": quality,
            "eac_score": quality,
            "cubic_score": 1.0 - quality,
        }
    except Exception:
        return {"valid": False, "inlier_ratio": 0.0, "reprojection_error": 999.0, "quality": 0.0}
