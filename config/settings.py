"""Central application settings.

All configuration is loaded from environment variables (and optionally a
``.env`` file at the project root).  A single :class:`Settings` instance is
shared across the application via :func:`get_settings`.

Environment Variables
---------------------
YOUTUBE_API_KEY         YouTube Data API v3 key.
CMS_API_URL             MediaCMS API endpoint, e.g. https://cms.example.com/api/v1/media.
CMS_USER                MediaCMS username for HTTP basic auth.
CMS_PASSWORD            MediaCMS password for HTTP basic auth.
CMS_TOKEN               MediaCMS CSRF token.

VPD_PROJECT_ROOT        Override the auto-detected project root.
VPD_FRAMES_OUTPUT_DIR   Directory for detector analysis frame output.
VPD_DEBUG_OUTPUT_DIR    Directory for detector debug images.
VPD_MIN_FRAMES_ANALYZED Minimum analysed frames required (default: 4).
VPD_MAX_DISCARD_RATIO   Max allowable frame-discard ratio (default: 0.65).
VPD_MIN_LAYOUT_SCORE_MARGIN Minimum score margin for layout decision (default: 0.10).
VPD_LINE_CENTER_BAND_RATIO  Not used by the detector (default: 0.08).  Exposed on Settings
                        and in as_detector_config() only.  The strict search band is driven
                        by VPD_LINE_CENTER_SEARCH_BAND_RATIO; the centring tolerance by
                        VPD_LINE_CENTER_MAX_DISTANCE_RATIO.
VPD_LINE_CENTER_SEARCH_BAND_RATIO  Height (width) of the central ROI strip searched for a
                        horizontal (vertical) seam, as a fraction of frame height (width)
                        (default: 0.02).  Passed as detect_*_line(search_band_ratio=...);
                        the ROI half-height is round(dim * ratio / 2), min 2 px.
VPD_LINE_CENTER_MAX_DISTANCE_RATIO  Max relative distance of line from centre (default: 0.02).
                        Passed as detect_*_line(center_tolerance_ratio=...).
VPD_LINE_MAX_SLOPE      Maximum allowed line slope for detection (default: 0.05).
VPD_LINE_MIN_COVERAGE_RATIO  Minimum line coverage fraction (default: 0.20).
VPD_LINE_MIN_QUALITY_SCORE  Min structural quality for non-fallback candidates (default: 0.62).
VPD_LINE_FALLBACK_MIN_QUALITY_SCORE  Min quality for LSD/fitLine candidates (default: 0.78).
VPD_LINE_FFT_MIN_DOMINANCE  Min FFT orientation dominance to confirm a seam (default: 0.015).
                        The statistic is area-normalised (mean energy per frequency bin), so
                        its seamless baseline is ~0.000 at every resolution.  Raise to reduce
                        false positives; values above ~0.025 reject most real seams.
VPD_LINE_STRONG_COVERAGE_RATIO  Span at which the slope-tightness rule is relaxed (default: 0.40).
VPD_LINE_MORPH_LENGTH_RATIO  Morphological close kernel length ratio (default: 0.02).
VPD_LINE_ENABLE_PROFILE_GATE  Enforce the projection-profile check (default: false, advisory).
VPD_LINE_PROFILE_MIN_COVERAGE_RATIO  Profile-gate column/row agreement fraction (default: 0.20).
VPD_LINE_PROFILE_MIN_PROMINENCE  Profile-gate peak/median gradient ratio (default: 3.0).
VPD_STEREO_HIST_THRESHOLD   Histogram correlation threshold for stereo (default: 0.92).
VPD_SAVE_STEREO_HALVES  Whether to save stereo half-frame debug images (default: true).
VPD_FLOW_ALGORITHM      Optical-flow algorithm (default: dis).
VPD_FLOW_ENABLE_REFINEMENT  Enable optional variational-refinement pass (default: false).
VPD_FLOW_ENABLE_FB_CHECK    Enable forward-backward consistency filtering (default: false).
VPD_FLOW_FB_THRESHOLD       Forward-backward error threshold in pixels (default: 1.5).
VPD_ENABLE_FACE_CONTINUITY  Use face-boundary continuity to decide EAC vs cubic
                        (default: true).  Single-frame and motion-independent.  Set false
                        to decide by the optical-flow angular vote instead.
VPD_FACE_CONTINUITY_EAC_CONTRAST   Median edge-minus-seam continuity at or above which
                        frames may be called EAC (default: 0.30).
VPD_FACE_CONTINUITY_EAC_EDGE_FLOOR Absolute edge continuity an EAC verdict must clear
                        on its own (default: 0.35).  Real EAC measures 0.68-0.85.
VPD_FACE_CONTINUITY_CUBIC_CONTRAST Median contrast below which frames may be called cubic
                        (default: 0.20).
VPD_FACE_CONTINUITY_CUBIC_EDGE_CEILING  Absolute edge continuity a cubic verdict must stay
                        under (default: 0.25).  Real cubic measures -0.02..0.10.
VPD_FACE_CONTINUITY_CUBIC_EDGE_DECISIVE Edge continuity below which cubic is called on the
                        edge evidence alone, without the contrast differential
                        (default: 0.20).
VPD_FACE_CONTINUITY_MIN_FRAMES     Minimum frames with measurable border texture
                        required before any verdict is issued (default: 2).
                        NB: these thresholds are calibrated on real footage for a
                        windowed structural continuity metric.
VPD_EQUI_JUNCTION_MIN_CORR  Equirectangular evidence, signal A: minimum correlation
                        between the frame's last and first column (the 360° join)
                        (default: 0.84).
VPD_EQUI_JUNCTION_MAX_RATIO Signal A: maximum join difference relative to the median
                        adjacent-column difference inside the frame (default: 11.0).
VPD_EQUI_JUNCTION_MIN_TEXTURE  Signal A: minimum std (grey levels) of each edge column;
                        below it the frame is unusable rather than scored (default: 1.0).
VPD_EQUI_POLAR_MAX_ANISOTROPY  Signal B: maximum horizontal/vertical gradient ratio in the
                        top and bottom bands; equirectangular poles are stretched
                        sideways (default: 0.41).
VPD_EQUI_BAND_MIN_TEXTURE   Signals B/C: minimum mean gradient (grey levels) of a band;
                        a constant band fails instead of passing trivially (default: 0.01).
VPD_EQUI_EQUATOR_MIN_ANISOTROPY  Signal C: minimum gradient ratio in the middle band; in
                        top-bottom stereo the middle rows are poles (default: 0.27).
VPD_EQUI_MAX_HALF_SIMILARITY  Signal C: maximum similarity between the frame's two halves
                        on either axis; stereo eyes are near-copies (default: 0.66).
VPD_EQUI_MIN_PASS_RATIO     Minimum fraction of usable frames passing A, B and C for
                        positive equirectangular evidence (default: 0.5).
VPD_EQUI_MIN_FRAMES         Minimum usable frames before any equirectangular verdict
                        (default: 2).
                        NB: calibrated on real footage with leave-one-source-out
                        validation.
VPD_ENABLE_GEOMETRY_EVIDENCE  Enable the per-pair ORB+homography geometry pass
                        (default: false).  Telemetry only: it does not influence the
                        EAC-vs-cubic decision.
VPD_GEOMETRY_EVIDENCE_WEIGHT  Geometry evidence blend weight (default: 0.2).
VPD_MOTION_ROLLOUT_PROFILE  Motion feature profile: baseline|robust|high_accuracy (default: high_accuracy).
VPD_FORCE_FULL_CODEC_NORMALIZATION  Force a full-file codec-normalization transcode before detection (default: false).
DOWNLOADS_DIR           Override default download directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_dotenv(dotenv_path: str) -> None:
    """Minimal .env parser used when python-dotenv is unavailable."""
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def _load_dotenv(project_root: str) -> None:
    dotenv_path = os.path.join(project_root, ".env")
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=dotenv_path, override=False)
    except Exception:
        _parse_dotenv(dotenv_path)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


# ---------------------------------------------------------------------------
# Settings class
# ---------------------------------------------------------------------------

class Settings:
    """Immutable snapshot of all application configuration values."""

    def __init__(self) -> None:
        # Resolve project root from env, or by walking up from this file —
        # except when frozen by PyInstaller, where this file lives inside a
        # temporary extraction directory rather than next to the actual
        # executable, so .env discovery and runtime dirs must anchor on
        # sys.executable's directory instead.
        if getattr(sys, "frozen", False):
            _auto_root = os.path.dirname(sys.executable)
        else:
            _auto_root = str(Path(__file__).resolve().parent.parent)
        self.project_root: str = os.getenv("VPD_PROJECT_ROOT", _auto_root)

        # Load .env once so subsequent os.getenv calls see it.
        _load_dotenv(self.project_root)

        # Re-read project root after .env loading (allows override inside .env).
        self.project_root = os.getenv("VPD_PROJECT_ROOT", self.project_root)

        # ---- Runtime directories ----
        _data_dir = os.path.join(self.project_root, "data")
        self.downloads_dir: str = _env_str("DOWNLOADS_DIR", os.path.join(_data_dir, "downloads"))
        self.frames_output_dir: str = _env_str(
            "VPD_FRAMES_OUTPUT_DIR", os.path.join(_data_dir, "frames")
        )
        self.debug_output_dir: str = _env_str(
            "VPD_DEBUG_OUTPUT_DIR", self.frames_output_dir
        )
        self.logs_dir: str = os.path.join(_data_dir, "logs")
        self.jobs_dir: str = os.path.join(_data_dir, "jobs")
        self.tmp_dir: str = os.path.join(_data_dir, "tmp")

        # ---- External credentials ----
        self.youtube_api_key: Optional[str] = os.getenv("YOUTUBE_API_KEY") or None
        self.cms_api_url: Optional[str] = os.getenv("CMS_API_URL") or None
        self.cms_user: Optional[str] = (
            os.getenv("CMS_USER") or os.getenv("USER") or None
        )
        self.cms_password: Optional[str] = os.getenv("CMS_PASSWORD") or None
        self.cms_token: Optional[str] = os.getenv("CMS_TOKEN") or None

        # ---- Detector thresholds ----
        self.min_frames_analyzed: int = _env_int("VPD_MIN_FRAMES_ANALYZED", 4)
        self.max_discard_ratio: float = _env_float("VPD_MAX_DISCARD_RATIO", 0.65)
        self.min_layout_score_margin: float = _env_float("VPD_MIN_LAYOUT_SCORE_MARGIN", 0.10)
        # Not consumed by the detector: detect_horizontal_line/detect_vertical_line use
        # line_center_search_band_ratio and line_center_max_distance_ratio
        # (detector/pipeline.py).  Setting VPD_LINE_CENTER_BAND_RATIO has no effect.
        self.line_center_band_ratio: float = _env_float("VPD_LINE_CENTER_BAND_RATIO", 0.08)
        self.line_center_search_band_ratio: float = _env_float(
            "VPD_LINE_CENTER_SEARCH_BAND_RATIO", 0.02
        )
        self.line_center_max_distance_ratio: float = _env_float(
            "VPD_LINE_CENTER_MAX_DISTANCE_RATIO", 0.02
        )
        self.line_max_slope: float = _env_float("VPD_LINE_MAX_SLOPE", 0.05)
        self.line_min_coverage_ratio: float = _env_float("VPD_LINE_MIN_COVERAGE_RATIO", 0.20)
        self.line_min_quality_score: float = _env_float("VPD_LINE_MIN_QUALITY_SCORE", 0.62)
        self.line_fallback_min_quality_score: float = _env_float(
            "VPD_LINE_FALLBACK_MIN_QUALITY_SCORE", 0.78
        )
        # The FFT `dominance` statistic in detector/line_detection.py compares mean
        # energy per frequency bin (comparing band *sums* over unequal bin counts would
        # offset it by -0.11..+0.06 depending on resolution), so the seamless baseline
        # is ~0.000 everywhere.  0.015 is the
        # end-to-end balanced-accuracy optimum on the synthetic corpus and the lowest
        # point on the TPR plateau at which the false-positive rate reaches zero.
        self.line_fft_min_dominance: float = _env_float("VPD_LINE_FFT_MIN_DOMINANCE", 0.015)
        self.line_strong_coverage_ratio: float = _env_float("VPD_LINE_STRONG_COVERAGE_RATIO", 0.40)
        self.line_morph_length_ratio: float = _env_float("VPD_LINE_MORPH_LENGTH_RATIO", 0.02)
        self.line_enable_profile_gate: bool = _env_bool("VPD_LINE_ENABLE_PROFILE_GATE", False)
        self.line_profile_min_coverage_ratio: float = _env_float(
            "VPD_LINE_PROFILE_MIN_COVERAGE_RATIO", 0.20
        )
        self.line_profile_min_prominence: float = _env_float(
            "VPD_LINE_PROFILE_MIN_PROMINENCE", 3.0
        )
        self.stereo_hist_similarity_threshold: float = _env_float(
            "VPD_STEREO_HIST_THRESHOLD", 0.92
        )
        self.stereo_seam_guard_ratio: float = _env_float("VPD_STEREO_SEAM_GUARD_RATIO", 0.02)
        self.stereo_min_valid_half_ratio: float = _env_float(
            "VPD_STEREO_MIN_VALID_HALF_RATIO", 0.22
        )
        self.stereo_edge_similarity_threshold: float = _env_float(
            "VPD_STEREO_EDGE_SIMILARITY_THRESHOLD", 0.08
        )
        self.stereo_min_seam_frames: int = _env_int("VPD_STEREO_MIN_SEAM_FRAMES", 2)
        self.stereo_min_stability_ratio: float = _env_float(
            "VPD_STEREO_MIN_STABILITY_RATIO", 0.55
        )
        self.save_stereo_halves: bool = _env_bool("VPD_SAVE_STEREO_HALVES", True)
        self.flow_algorithm: str = _env_str("VPD_FLOW_ALGORITHM", "dis")
        self.flow_enable_refinement: bool = _env_bool("VPD_FLOW_ENABLE_REFINEMENT", False)
        self.flow_enable_fb_check: bool = _env_bool("VPD_FLOW_ENABLE_FB_CHECK", False)
        self.flow_fb_threshold: float = _env_float("VPD_FLOW_FB_THRESHOLD", 1.5)
        # Face-boundary continuity is the EAC-vs-cubic discriminator (see
        # detector/face_continuity_detection.py). Single-frame and motion-independent.
        # The optical-flow angular vote is right only 40% of the time it decides —
        # below chance for a two-way decision — so disable this only for comparison.
        self.enable_face_continuity: bool = _env_bool("VPD_ENABLE_FACE_CONTINUITY", True)
        # Thresholds below are calibrated against real footage. The metric is a
        # windowed structural correlation (not a raw pixel correlation), which
        # avoids the shared-luminance confound between neighbouring faces.
        self.face_continuity_eac_contrast: float = _env_float(
            "VPD_FACE_CONTINUITY_EAC_CONTRAST", 0.30
        )
        # Absolute floor an EAC verdict must clear on its own.
        self.face_continuity_eac_edge_floor: float = _env_float(
            "VPD_FACE_CONTINUITY_EAC_EDGE_FLOOR", 0.35
        )
        self.face_continuity_cubic_contrast: float = _env_float(
            "VPD_FACE_CONTINUITY_CUBIC_CONTRAST", 0.20
        )
        # Symmetric counterpart of the EAC floor: absolute ceiling a cubic verdict
        # must stay under, and the level below which the edge score alone settles it.
        self.face_continuity_cubic_edge_ceiling: float = _env_float(
            "VPD_FACE_CONTINUITY_CUBIC_EDGE_CEILING", 0.25
        )
        self.face_continuity_cubic_edge_decisive: float = _env_float(
            "VPD_FACE_CONTINUITY_CUBIC_EDGE_DECISIVE", 0.20
        )
        self.face_continuity_min_frames: int = _env_int("VPD_FACE_CONTINUITY_MIN_FRAMES", 2)
        # Positive equirectangular evidence (detector/equirectangular_detection.py):
        # A = continuity across the 360° join, B = stretched poles, C = mono, not
        # stereo. Calibrated against real footage with leave-one-source-out
        # validation.
        self.equi_junction_min_corr: float = _env_float("VPD_EQUI_JUNCTION_MIN_CORR", 0.84)
        self.equi_junction_max_ratio: float = _env_float("VPD_EQUI_JUNCTION_MAX_RATIO", 11.0)
        self.equi_junction_min_texture: float = _env_float("VPD_EQUI_JUNCTION_MIN_TEXTURE", 1.0)
        self.equi_polar_max_anisotropy: float = _env_float("VPD_EQUI_POLAR_MAX_ANISOTROPY", 0.41)
        self.equi_band_min_texture: float = _env_float("VPD_EQUI_BAND_MIN_TEXTURE", 0.01)
        self.equi_equator_min_anisotropy: float = _env_float("VPD_EQUI_EQUATOR_MIN_ANISOTROPY", 0.27)
        self.equi_max_half_similarity: float = _env_float("VPD_EQUI_MAX_HALF_SIMILARITY", 0.66)
        self.equi_min_pass_ratio: float = _env_float("VPD_EQUI_MIN_PASS_RATIO", 0.5)
        self.equi_min_frames: int = _env_int("VPD_EQUI_MIN_FRAMES", 2)
        self.enable_geometry_evidence: bool = _env_bool("VPD_ENABLE_GEOMETRY_EVIDENCE", False)
        self.geometry_evidence_weight: float = _env_float("VPD_GEOMETRY_EVIDENCE_WEIGHT", 0.20)
        self.motion_rollout_profile: str = _env_str("VPD_MOTION_ROLLOUT_PROFILE", "high_accuracy")
        # Long-edge cap (px) for frames entering dense optical flow. 360 sources are
        # commonly 4K-8K; uncapped native-resolution flow (a single TV-L1 flow at
        # 3840x2048 exceeds four minutes) would make the motion stage appear to
        # hang. Angles are scale-invariant, so this is safe for layout scoring, and a
        # fixed working resolution also makes pixel-magnitude thresholds consistent
        # across source sizes. 0 disables the cap (flow at native resolution).
        self.motion_max_work_resolution: int = _env_int("VPD_MOTION_MAX_WORK_RESOLUTION", 960)
        self.force_full_codec_normalization: bool = _env_bool("VPD_FORCE_FULL_CODEC_NORMALIZATION", False)
        self.motion_feature_tiers: dict = {
            "tier_a_features": [
                "canny_morphology_houghlinesp",
                "line_segment_detector",
                "dft_orientation_checks",
                "orb_bfmatcher",
                "find_homography_ransac_usac_magsac",
                "estimate_affine_partial_2d",
                "forward_backward_consistency",
                "variational_refinement",
            ],
            "tier_b_flow_algorithms": ["dis", "tvl1"],
            "tier_c_flow_algorithms": ["deepflow", "pcaflow", "sparse_to_dense"],
        }

        # Detector-compatible dict view, consumed as the CONFIG dict in pipeline.py.
        self._detector_config = {
            "project_root": self.project_root,
            "frames_output_dir": self.frames_output_dir,
            "debug_output_dir": self.debug_output_dir,
            "main_runs_dir": self.debug_output_dir,
            "min_frames_analyzed_required": self.min_frames_analyzed,
            "max_discard_ratio": self.max_discard_ratio,
            "min_layout_score_margin": self.min_layout_score_margin,
            "line_center_band_ratio": self.line_center_band_ratio,
            "line_center_search_band_ratio": self.line_center_search_band_ratio,
            "line_center_max_distance_ratio": self.line_center_max_distance_ratio,
            "line_max_slope": self.line_max_slope,
            "line_min_coverage_ratio": self.line_min_coverage_ratio,
            "line_min_quality_score": self.line_min_quality_score,
            "line_fallback_min_quality_score": self.line_fallback_min_quality_score,
            "line_fft_min_dominance": self.line_fft_min_dominance,
            "line_strong_coverage_ratio": self.line_strong_coverage_ratio,
            "line_morph_length_ratio": self.line_morph_length_ratio,
            "line_enable_profile_gate": self.line_enable_profile_gate,
            "line_profile_min_coverage_ratio": self.line_profile_min_coverage_ratio,
            "line_profile_min_prominence": self.line_profile_min_prominence,
            "stereo_hist_similarity_threshold": self.stereo_hist_similarity_threshold,
            "stereo_seam_guard_ratio": self.stereo_seam_guard_ratio,
            "stereo_min_valid_half_ratio": self.stereo_min_valid_half_ratio,
            "stereo_edge_similarity_threshold": self.stereo_edge_similarity_threshold,
            "stereo_min_seam_frames": self.stereo_min_seam_frames,
            "stereo_min_stability_ratio": self.stereo_min_stability_ratio,
            "save_stereo_halves": self.save_stereo_halves,
            "flow_algorithm": self.flow_algorithm,
            "flow_enable_refinement": self.flow_enable_refinement,
            "flow_enable_fb_check": self.flow_enable_fb_check,
            "flow_fb_threshold": self.flow_fb_threshold,
            "enable_face_continuity": self.enable_face_continuity,
            "face_continuity_eac_contrast": self.face_continuity_eac_contrast,
            "face_continuity_eac_edge_floor": self.face_continuity_eac_edge_floor,
            "face_continuity_cubic_contrast": self.face_continuity_cubic_contrast,
            "face_continuity_cubic_edge_ceiling": self.face_continuity_cubic_edge_ceiling,
            "face_continuity_cubic_edge_decisive": self.face_continuity_cubic_edge_decisive,
            "face_continuity_min_frames": self.face_continuity_min_frames,
            "equi_junction_min_corr": self.equi_junction_min_corr,
            "equi_junction_max_ratio": self.equi_junction_max_ratio,
            "equi_junction_min_texture": self.equi_junction_min_texture,
            "equi_polar_max_anisotropy": self.equi_polar_max_anisotropy,
            "equi_band_min_texture": self.equi_band_min_texture,
            "equi_equator_min_anisotropy": self.equi_equator_min_anisotropy,
            "equi_max_half_similarity": self.equi_max_half_similarity,
            "equi_min_pass_ratio": self.equi_min_pass_ratio,
            "equi_min_frames": self.equi_min_frames,
            "enable_geometry_evidence": self.enable_geometry_evidence,
            "geometry_evidence_weight": self.geometry_evidence_weight,
            "motion_rollout_profile": self.motion_rollout_profile,
            "motion_max_work_resolution": self.motion_max_work_resolution,
            "motion_feature_tiers": self.motion_feature_tiers,
        }

    def ensure_runtime_dirs(self) -> None:
        """Create all required runtime directories."""
        for directory in (
            self.downloads_dir,
            self.frames_output_dir,
            self.debug_output_dir,
            self.logs_dir,
            self.jobs_dir,
            self.tmp_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    def as_detector_config(self) -> dict:
        """Return the detector-compatible config dict."""
        return dict(self._detector_config)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the application-wide :class:`Settings` singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
