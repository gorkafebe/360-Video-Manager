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
VPD_LINE_CENTER_BAND_RATIO  Band ratio around frame centre for line search (default: 0.08).
VPD_LINE_CENTER_MAX_DISTANCE_RATIO  Max relative distance of line from centre (default: 0.02).
VPD_LINE_MAX_SLOPE      Maximum allowed line slope for detection (default: 0.05).
VPD_LINE_MIN_COVERAGE_RATIO  Minimum line coverage fraction (default: 0.20).
VPD_STEREO_HIST_THRESHOLD   Histogram correlation threshold for stereo (default: 0.92).
VPD_SAVE_STEREO_HALVES  Whether to save stereo half-frame debug images (default: true).
VPD_FLOW_ALGORITHM      Optical-flow algorithm (default: deepflow).
VPD_FLOW_ENABLE_REFINEMENT  Enable optional variational-refinement pass (default: false).
VPD_FLOW_ENABLE_FB_CHECK    Enable forward-backward consistency filtering (default: false).
VPD_FLOW_FB_THRESHOLD       Forward-backward error threshold in pixels (default: 1.5).
VPD_ENABLE_GEOMETRY_EVIDENCE  Enable geometry quality evidence fusion (default: false).
VPD_GEOMETRY_EVIDENCE_WEIGHT  Geometry evidence blend weight (default: 0.2).
VPD_MOTION_ROLLOUT_PROFILE  Motion feature profile: baseline|robust|high_accuracy (default: high_accuracy).
DOWNLOADS_DIR           Override default download directory.
"""

from __future__ import annotations

import os
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
        # Resolve project root from env or by walking up from this file.
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
        self.line_center_band_ratio: float = _env_float("VPD_LINE_CENTER_BAND_RATIO", 0.08)
        self.line_center_max_distance_ratio: float = _env_float(
            "VPD_LINE_CENTER_MAX_DISTANCE_RATIO", 0.02
        )
        self.line_max_slope: float = _env_float("VPD_LINE_MAX_SLOPE", 0.05)
        self.line_min_coverage_ratio: float = _env_float("VPD_LINE_MIN_COVERAGE_RATIO", 0.20)
        self.stereo_hist_similarity_threshold: float = _env_float(
            "VPD_STEREO_HIST_THRESHOLD", 0.92
        )
        self.save_stereo_halves: bool = _env_bool("VPD_SAVE_STEREO_HALVES", True)
        self.flow_algorithm: str = _env_str("VPD_FLOW_ALGORITHM", "deepflow")
        self.flow_enable_refinement: bool = _env_bool("VPD_FLOW_ENABLE_REFINEMENT", False)
        self.flow_enable_fb_check: bool = _env_bool("VPD_FLOW_ENABLE_FB_CHECK", False)
        self.flow_fb_threshold: float = _env_float("VPD_FLOW_FB_THRESHOLD", 1.5)
        self.enable_geometry_evidence: bool = _env_bool("VPD_ENABLE_GEOMETRY_EVIDENCE", False)
        self.geometry_evidence_weight: float = _env_float("VPD_GEOMETRY_EVIDENCE_WEIGHT", 0.20)
        self.motion_rollout_profile: str = _env_str("VPD_MOTION_ROLLOUT_PROFILE", "high_accuracy")
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
            "tier_b_flow_algorithms": ["tvl1", "dis"],
            "tier_c_flow_algorithms": ["deepflow", "pcaflow", "sparse_to_dense"],
        }

        # Expose detector-compatible dict view (for backward compatibility with
        # detector modules that consume the CONFIG dict from pipeline.py).
        self._detector_config = {
            "project_root": self.project_root,
            "frames_output_dir": self.frames_output_dir,
            "debug_output_dir": self.debug_output_dir,
            "main_runs_dir": self.debug_output_dir,
            "min_frames_analyzed_required": self.min_frames_analyzed,
            "max_discard_ratio": self.max_discard_ratio,
            "min_layout_score_margin": self.min_layout_score_margin,
            "line_center_band_ratio": self.line_center_band_ratio,
            "line_center_max_distance_ratio": self.line_center_max_distance_ratio,
            "line_max_slope": self.line_max_slope,
            "line_min_coverage_ratio": self.line_min_coverage_ratio,
            "stereo_hist_similarity_threshold": self.stereo_hist_similarity_threshold,
            "save_stereo_halves": self.save_stereo_halves,
            "flow_algorithm": self.flow_algorithm,
            "flow_enable_refinement": self.flow_enable_refinement,
            "flow_enable_fb_check": self.flow_enable_fb_check,
            "flow_fb_threshold": self.flow_fb_threshold,
            "enable_geometry_evidence": self.enable_geometry_evidence,
            "geometry_evidence_weight": self.geometry_evidence_weight,
            "motion_rollout_profile": self.motion_rollout_profile,
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
