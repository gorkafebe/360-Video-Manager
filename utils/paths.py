"""Path helpers for 360-Video-Manager.

All runtime directories are derived from :func:`config.settings.get_settings`
so they remain consistent with the configured ``project_root`` and any
environment-variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory as a :class:`~pathlib.Path`."""
    from config.settings import get_settings
    return Path(get_settings().project_root)


def get_downloads_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().downloads_dir)


def get_frames_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().frames_output_dir)


def get_logs_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().logs_dir)


def get_jobs_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().jobs_dir)


def get_tmp_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().tmp_dir)


def ensure_runtime_dirs() -> None:
    """Create all required runtime directories."""
    from config.settings import get_settings
    get_settings().ensure_runtime_dirs()
