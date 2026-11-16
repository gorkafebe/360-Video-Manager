"""Job manifest persistence.

Each processed video job is serialised to a JSON file in ``data/jobs/``
so that runs are auditable and can be resumed or debugged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.models import JobResult


logger = logging.getLogger(__name__)


def _make_job_id() -> str:
    """Return a unique, timestamp-based job ID."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def save_job_manifest(result: JobResult, jobs_dir: str) -> Optional[str]:
    """Serialise *result* to a JSON manifest file inside *jobs_dir*.

    Returns the path of the written file, or ``None`` when writing fails.

    The manifest filename is ``job_<job_id>.json``.  If *result.job_id* is not
    set a new ID is generated.
    """
    try:
        os.makedirs(jobs_dir, exist_ok=True)
        job_id = result.job_id or _make_job_id()
        result.job_id = job_id
        manifest_path = os.path.join(jobs_dir, f"job_{job_id}.json")

        payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "job_id": job_id,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "result": result.to_dict(),
        }

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("Job manifest saved: %s", manifest_path)
        result.manifest_path = manifest_path
        return manifest_path
    except Exception as exc:
        logger.warning("Could not save job manifest: %s", exc)
        return None


def load_job_manifest(manifest_path: str) -> Optional[Dict[str, Any]]:
    """Load and return a previously saved job manifest dict.

    Returns ``None`` if the file does not exist or cannot be parsed.
    """
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not load manifest %s: %s", manifest_path, exc)
        return None


def new_job_id() -> str:
    """Generate a new unique job ID."""
    return _make_job_id()
