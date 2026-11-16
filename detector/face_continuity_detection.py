"""EAC-vs-cubic discrimination from face-boundary content continuity.

A sibling of :mod:`detector.equirectangular_detection`: where that module looks
for wrap-around continuity at the left/right frame borders, this one looks for
continuity at the *internal* borders of a packed 3x2 cube atlas.

Why this works
--------------
A 3x2 atlas concatenates six faces for compactness, and only some adjacent cells
hold faces that actually meet along a shared cube edge. Derived numerically in
:func:`detector.projection_geometry.internal_border_adjacency` from ffmpeg's own
``vf_v360.c`` face tables:

* **EAC** — the four *within-row* borders are genuine shared cube edges (each row
  is a contiguous band around the cube). The three cross-row borders are not.
* **cubic (c3x2)** — with ffmpeg's default ``rludfb`` / ``000000`` ordering,
  **none** of the seven internal borders is a shared edge.

So real image content flows across EAC's within-row borders and across nothing at
all in a c3x2 frame. The discriminant is the **contrast** between the
would-be-edge borders and the would-be-seam borders, because absolute continuity
is not usable on its own: ordinary equirectangular content is continuous
everywhere and scores high at every border.

What the correlation must NOT measure
-------------------------------------
Comparing the two border-adjacent strips as raw pixel intensities does not
measure "does the content continue" — it measures "are these two strips bright
in the same places". Those are not the same thing, and on real 360 footage they
come apart badly:

* The four EAC-edge borders are all *within-row*, so they join two side-facing
  faces. Both look out at the same horizon, so both carry the same coarse
  top-to-bottom luminance profile (bright sky above, dark ground/sea below)
  whether or not the content actually continues.
* Measured on real footage relabelled to cubic via ``v360=eac:c3x2``, that
  confound supplied essentially the *entire* raw correlation. Decomposing each
  border into its coarse along-border profile and the residual fine detail gave,
  for Antarctica-as-cubic: full ``+0.496``, profile component ``+0.497``,
  residual ``-0.000``. The metric was reading the shared horizon, not continuity.

So instead of the raw correlation a **structural** one is used, computed on
high-passed profiles in short windows along the border and aggregated by median:

* **High-pass** removes the shared illumination trend each strip carries on its
  own (the homomorphic-filtering idea: separate illumination from structure).
* **Short windows + median** stop any single shared feature from carrying the
  score. A horizon is one sharp step at the *same* height on both side faces, so
  it survives high-passing and would still correlate — but it occupies one or two
  windows out of ~24, and the median ignores it. Genuine continuity, by contrast,
  holds in *every* window. This is the same reason SSIM is computed per-window
  and pooled rather than globally.

Both steps are classical, cheap and applied symmetrically to the edge and seam
border groups, so the confound is removed rather than moved from one group to
the other.

This check needs a single frame and no motion whatsoever, which is exactly the
case the optical-flow route handles worst (see the README's "Static-scene
detection reliability" caveat).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .projection_geometry import internal_border_adjacency

logger = logging.getLogger(__name__)

# Borders of the 3x2 grid, split by whether an EAC packing makes them real cube
# edges. Computed once from the geometry module rather than hand-listed.
_EAC_ADJACENCY = internal_border_adjacency("eac")
EAC_EDGE_BORDERS = tuple(b for b, is_edge in _EAC_ADJACENCY.items() if is_edge)
EAC_SEAM_BORDERS = tuple(b for b, is_edge in _EAC_ADJACENCY.items() if not is_edge)

# --- structural-correlation geometry ---------------------------------------
# Windows along each border. ~24 is enough that one shared feature (a horizon
# step lands in one or two) cannot move the median, while each window still
# holds enough samples for a stable correlation.
_TARGET_WINDOWS = 24
_MIN_WINDOW_PX = 12
# A border needs this many measurable windows before its score is trusted.
_MIN_VALID_WINDOWS = 6

# --- adaptive flatness gate ------------------------------------------------
# A window is measurable when its high-passed structure stands above the frame's
# own noise floor. Using the frame's noise rather than a fixed level of 8-bit
# luminance is what lets dark-but-structured footage be measured at all: near
# black, real detail lives at an amplitude a fixed threshold rejects wholesale,
# while a genuinely flat (or fully black) band still has no structure to find and
# is still discarded.
_NOISE_SIGMA_K = 1.6
# Absolute floor in 8-bit levels. Below this, nothing survives quantisation, so
# a "signal" here is quantisation noise and must not be correlated.
_ABS_NOISE_FLOOR = 0.30
# Fewer measurable borders than this on either side and the frame abstains.
_MIN_EDGE_BORDERS = 3
_MIN_SEAM_BORDERS = 2


def _border_lines(gray: np.ndarray, border, strip_px: int):
    """Return the two pixel bands immediately touching *border*.

    Both are returned oriented so that **axis 0 runs along the border**, which
    lets the caller compare them position-by-position regardless of border
    orientation.
    """
    height, width = gray.shape[:2]
    face_w, face_h = width // 3, height // 2
    (row_a, col_a), (row_b, col_b) = border

    if row_a == row_b:  # vertical border between two cells in the same row
        x = (col_a + 1) * face_w
        if x - strip_px < 0 or x + strip_px > width:
            return None
        top, bottom = row_a * face_h, (row_a + 1) * face_h
        return gray[top:bottom, x - strip_px:x], gray[top:bottom, x:x + strip_px]

    y = face_h  # horizontal border between the two rows
    if y - strip_px < 0 or y + strip_px > height:
        return None
    left_a, right_a = col_a * face_w, (col_a + 1) * face_w
    left_b, right_b = col_b * face_w, (col_b + 1) * face_w
    return (gray[y - strip_px:y, left_a:right_a].T,
            gray[y:y + strip_px, left_b:right_b].T)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Robust estimate of the frame's per-pixel noise, in 8-bit levels.

    Median-absolute-deviation of horizontal first differences. The MAD is taken
    rather than the standard deviation so that real edges — which are large but
    rare — do not inflate the estimate of what "flat" costs.
    """
    if gray.size == 0:
        return 0.0
    sample = gray[::4, :].astype(np.float32)
    if sample.shape[1] < 2:
        return 0.0
    diff = np.diff(sample, axis=1).ravel()
    if diff.size == 0:
        return 0.0
    mad = float(np.median(np.abs(diff - np.median(diff))))
    # 1.4826 -> sigma for a normal; /sqrt(2) because a difference of two
    # independent samples has sqrt(2) times the single-sample noise.
    return float(1.4826 * mad / np.sqrt(2.0))


def _highpass(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Remove structure coarser than *sigma* samples from a 1-D signal."""
    column = signal.reshape(-1, 1).astype(np.float32)
    ksize = int(max(3, (int(round(sigma * 6.0)) | 1)))
    smooth = cv2.GaussianBlur(
        column, (1, ksize), sigmaX=0.0, sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    )
    return (column - smooth).ravel()


def _structural_correlation(
    band_a: np.ndarray,
    band_b: np.ndarray,
    min_std: float,
) -> Optional[float]:
    """Median windowed correlation of the two bands' high-passed profiles.

    Returns None when too few windows carry structure above *min_std*: a flat
    band is absence of evidence, and must not be averaged in as though it were a
    measured disagreement.
    """
    if band_a.size == 0 or band_a.shape != band_b.shape:
        return None
    # Collapse the (thin) perpendicular depth to a single profile running along
    # the border, then strip the shared illumination trend from each side.
    profile_a = band_a.astype(np.float32).mean(axis=1)
    profile_b = band_b.astype(np.float32).mean(axis=1)
    length = profile_a.size
    if length < _MIN_WINDOW_PX * _MIN_VALID_WINDOWS:
        return None

    window = int(max(_MIN_WINDOW_PX, length // _TARGET_WINDOWS))
    detrended_a = _highpass(profile_a, float(window))
    detrended_b = _highpass(profile_b, float(window))

    scores: List[float] = []
    for start in range(0, length - window + 1, window):
        a = detrended_a[start:start + window]
        b = detrended_b[start:start + window]
        a = a - a.mean()
        b = b - b.mean()
        if float(a.std()) < min_std or float(b.std()) < min_std:
            continue
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denominator <= 1e-9:
            continue
        scores.append(float(np.clip((a * b).sum() / denominator, -1.0, 1.0)))

    if len(scores) < _MIN_VALID_WINDOWS:
        return None
    return float(np.median(scores))


def compute_frame_face_continuity(
    frame: np.ndarray,
    strip_px: int = 2,
) -> Dict[str, Any]:
    """Face-boundary continuity evidence for a single packed frame.

    Args:
        frame: BGR frame believed to be a packed 3x2 cube atlas.
        strip_px: band thickness sampled on each side of a border.

    Returns:
        Dict with ``usable`` (bool), ``edge_continuity`` and ``seam_continuity``
        (mean structural correlation over each border group, in ``[-1, 1]``),
        ``contrast`` (``edge - seam``), the per-group border counts actually
        measured, the frame's ``noise_sigma``, and ``reason``.
    """
    base: Dict[str, Any] = {
        "usable": False,
        "edge_continuity": 0.0,
        "seam_continuity": 0.0,
        "contrast": 0.0,
        "edge_borders_measured": 0,
        "seam_borders_measured": 0,
        "noise_sigma": 0.0,
        "reason": "",
    }
    try:
        if frame is None or frame.size == 0:
            base["reason"] = "empty_frame"
            return base
        height, width = frame.shape[:2]
        if height < 32 or width < 48:
            base["reason"] = "frame_too_small"
            return base

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        noise_sigma = estimate_noise_sigma(gray)
        min_std = max(_ABS_NOISE_FLOOR, _NOISE_SIGMA_K * noise_sigma)
        base["noise_sigma"] = float(noise_sigma)

        def group_scores(borders) -> List[float]:
            values = []
            for border in borders:
                bands = _border_lines(gray, border, strip_px)
                if bands is None:
                    continue
                score = _structural_correlation(bands[0], bands[1], min_std)
                if score is not None:
                    values.append(score)
            return values

        edge_values = group_scores(EAC_EDGE_BORDERS)
        seam_values = group_scores(EAC_SEAM_BORDERS)

        # Fail safe: too little measurable structure means no vote, not a guess.
        if len(edge_values) < _MIN_EDGE_BORDERS or len(seam_values) < _MIN_SEAM_BORDERS:
            base["edge_borders_measured"] = len(edge_values)
            base["seam_borders_measured"] = len(seam_values)
            base["reason"] = (
                f"insufficient_texture_at_borders(edges={len(edge_values)}/"
                f"{len(EAC_EDGE_BORDERS)},seams={len(seam_values)}/{len(EAC_SEAM_BORDERS)},"
                f"min_std={min_std:.2f})"
            )
            return base

        edge_continuity = float(np.mean(edge_values))
        seam_continuity = float(np.mean(seam_values))
        return {
            "usable": True,
            "edge_continuity": edge_continuity,
            "seam_continuity": seam_continuity,
            "contrast": float(edge_continuity - seam_continuity),
            "edge_borders_measured": len(edge_values),
            "seam_borders_measured": len(seam_values),
            "noise_sigma": float(noise_sigma),
            "reason": "",
        }
    except Exception as exc:  # pragma: no cover - defensive, mirrors sibling module
        logger.debug("face-continuity evidence failed on frame: %s", exc)
        base["reason"] = str(exc)
        return base


def aggregate_face_continuity_evidence(
    frame_results: List[Dict[str, Any]],
    eac_contrast_threshold: float = 0.30,
    min_eac_edge_continuity: float = 0.35,
    cubic_contrast_threshold: float = 0.20,
    max_atlas_seam_continuity: float = 0.45,
    max_cubic_edge_continuity: float = 0.25,
    decisive_cubic_edge_continuity: float = 0.20,
    min_usable_frames: int = 2,
) -> Dict[str, Any]:
    """Aggregate per-frame continuity evidence into an EAC-vs-cubic verdict.

    Uses the median contrast so one anomalous frame cannot swing the decision,
    mirroring :func:`detector.equirectangular_detection.aggregate_equirectangular_evidence`.

    Both verdicts are gated on an **absolute** edge-continuity test as well as on
    the contrast differential, and symmetrically so:

    * EAC needs the would-be-edge borders to be genuinely continuous
      (``min_eac_edge_continuity``), not merely more continuous than the seams.
      Without this floor a frame can win on an unusually *discontinuous* seam
      group while showing no real edge continuity at all.
    * cubic needs them to be genuinely discontinuous (``max_cubic_edge_continuity``),
      and an edge score below ``decisive_cubic_edge_continuity`` is **sufficient on
      its own**, without the contrast differential. This matters because contrast
      is a difference of two noisy groups: a real cubic frame measured edge
      ``+0.055`` — decisively discontinuous — but seam ``-0.140``, which pushed
      ``contrast`` to ``+0.195`` and, on other frames of the same file, over the
      ceiling entirely. Requiring the differential there discarded unambiguous
      edge evidence and abstained on content the edge score alone settles.

    Args:
        frame_results: dicts from :func:`compute_frame_face_continuity`.
        eac_contrast_threshold: median contrast at or above which frames may be
            called EAC.
        min_eac_edge_continuity: absolute edge-continuity floor for an EAC verdict.
        cubic_contrast_threshold: median contrast strictly below which frames may
            be called cubic.
        max_atlas_seam_continuity: above this median seam continuity the frame is
            not a packed atlas (it is continuous across borders that no packing
            makes continuous), so no verdict is issued. This is what keeps
            equirectangular content from being mislabelled cubic.
        max_cubic_edge_continuity: absolute edge-continuity ceiling for cubic.
        decisive_cubic_edge_continuity: edge continuity below which cubic is
            called on the edge evidence alone.
        min_usable_frames: minimum frames carrying measurable structure.

    Returns:
        Dict with ``classification`` (``"eac"``, ``"cubic"`` or ``None``),
        ``confidence`` in ``[0, 1]``, the median statistics, and ``reason``.
    """
    usable = [r for r in frame_results if r.get("usable")]
    base = {
        "classification": None,
        "confidence": 0.0,
        "usable_frames": len(usable),
        "median_contrast": 0.0,
        "median_edge_continuity": 0.0,
        "median_seam_continuity": 0.0,
        "reason": "",
    }

    if len(usable) < int(min_usable_frames):
        base["reason"] = f"insufficient_usable_frames:{len(usable)}<{min_usable_frames}"
        return base

    contrasts = np.array([float(r["contrast"]) for r in usable], dtype=np.float64)
    median_contrast = float(np.median(contrasts))
    base.update(
        median_contrast=median_contrast,
        median_edge_continuity=float(np.median([r["edge_continuity"] for r in usable])),
        median_seam_continuity=float(np.median([r["seam_continuity"] for r in usable])),
    )

    median_seam = base["median_seam_continuity"]
    median_edge = base["median_edge_continuity"]

    # Neither verdict is meaningful unless the frame behaves like a packed atlas:
    # some internal border must actually break. Equirectangular content fails
    # this and abstains rather than being mislabelled cubic.
    if median_seam >= float(max_atlas_seam_continuity):
        base["reason"] = (
            f"not_a_packed_atlas:seam_continuity={median_seam:.3f}"
            f">={max_atlas_seam_continuity}"
        )
        return base

    if (median_contrast >= float(eac_contrast_threshold)
            and median_edge >= float(min_eac_edge_continuity)):
        base["classification"] = "eac"
        base["confidence"] = float(np.clip(median_edge, 0.0, 1.0))
        base["reason"] = "face_edges_continuous_seams_not"
        return base

    if median_edge < float(max_cubic_edge_continuity) and (
        median_contrast < float(cubic_contrast_threshold)
        or median_edge < float(decisive_cubic_edge_continuity)
    ):
        base["classification"] = "cubic"
        # Confidence grows as the edge score falls further below the EAC floor.
        base["confidence"] = float(
            np.clip(
                (float(min_eac_edge_continuity) - median_edge)
                / max(1e-6, float(min_eac_edge_continuity)),
                0.0, 1.0,
            )
        )
        base["reason"] = "no_face_border_shows_continuity"
        return base

    base["reason"] = (
        f"ambiguous:contrast={median_contrast:.3f},edge={median_edge:.3f}"
        f"(cubic:edge<{max_cubic_edge_continuity},eac:edge>={min_eac_edge_continuity})"
    )
    return base
