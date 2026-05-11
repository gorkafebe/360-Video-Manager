"""Region motion validation helpers."""

from typing import Any, Dict, Optional


def is_region_valid(
    region_motion_data: Optional[Dict[str, Any]],
    min_concentration: float = 0.25,
    min_active_ratio: float = 0.06,
) -> bool:
    """Return True when *region_motion_data* meets motion concentration thresholds."""
    if not region_motion_data:
        return False
    return bool(
        region_motion_data.get("valid", False)
        and float(region_motion_data.get("concentration", 0.0)) >= float(min_concentration)
        and float(region_motion_data.get("active_ratio", 0.0)) >= float(min_active_ratio)
    )
