"""Per-face pixel-to-ray geometry for the EAC and cubic (c3x2) 3x2 packings.

Pure geometry: no OpenCV, no I/O, no detection policy. Everything here is
transcribed from ffmpeg's own ``libavfilter/vf_v360.c`` (n6.1), because
``detector/projection_conversion.py`` drives that exact filter
(``v360=eac:equirect`` / ``v360=c3x2:equirect``), so ffmpeg's conventions — not a
paper's — are the ones this codebase must agree with.

Conventions
-----------
Axes follow ``cube_to_xyz``: ``+X`` right, ``+Y`` **down**, ``+Z`` forward. Faces
are named by the direction of their outward normal, with ``TOP`` meaning the
``-Y`` face and ``BOTTOM`` the ``+Y`` face (matching ffmpeg's ``UP``/``DOWN``).

The two packings
----------------
Both split the frame into a 3x2 grid of square faces, and both map a face's
normalised in-face coordinate ``u in [-0.5, +0.5]`` onto a tangent-plane
coordinate ``t in [-1, +1]``. **That mapping is the only geometric difference:**

* cubic (gnomonic): ``t = 2u``            -> angle = ``atan(2u)``, non-uniform
* EAC (equi-angular): ``t = tan(pi/2 u)`` -> angle = ``pi/2 u``, uniform

(`vf_v360.c:2698` — ``uf = tanf(M_PI_2 * uf)``.) Both agree exactly at the face
edges, where ``tan(+-pi/4) = +-1 = 2*(+-0.5)``; they differ only in the interior.
EAC therefore spends a constant ``2/pi`` pixels per radian, while cubic ranges
from ``0.5`` at a face centre to ``1.0`` at a face edge.

What differs beyond resampling
------------------------------
The packings also disagree on *which face sits in which cell* and *how each face
is rotated*, and that turns out to matter far more for telling them apart — see
:func:`internal_border_adjacency`.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# Face position in the 3x2 grid, as (row, col).
FacePos = Tuple[int, int]

# --- Layout tables, transcribed from vf_v360.c -----------------------------
#
# EAC: `prepare_eac_out` (vf_v360.c:2630) gives both the face order and the
# per-face rotation. The rotation is folded into `eac_to_xyz`'s switch
# (vf_v360.c:2710) rather than applied separately, so the ray formulas below are
# taken from that switch directly.
#
# Cubic: `cube3x2_to_xyz` -> `cube_to_xyz` (vf_v360.c:1070) with v360's defaults
# `in_forder="rludfb"` and `in_frot="000000"` — face order right/left/up/down/
# front/back across the grid, and no per-face rotation at all.

EAC_FACE_AT: Dict[FacePos, str] = {
    (0, 0): "LEFT", (0, 1): "FRONT", (0, 2): "RIGHT",
    (1, 0): "BOTTOM", (1, 1): "BACK", (1, 2): "TOP",
}

CUBIC_FACE_AT: Dict[FacePos, str] = {
    (0, 0): "RIGHT", (0, 1): "LEFT", (0, 2): "TOP",
    (1, 0): "BOTTOM", (1, 1): "FRONT", (1, 2): "BACK",
}


def _eac_ray(pos: FacePos, t_u: float, t_v: float) -> Tuple[float, float, float]:
    """Ray for EAC cell *pos* at tangent coordinates (t_u, t_v).

    Transcribed from ``eac_to_xyz``'s face switch (vf_v360.c:2710-2743). The
    bottom row's 90-degree rotations are already baked into these formulas.
    """
    if pos == (0, 0):   # TOP_LEFT — LEFT
        return (-1.0, t_v, t_u)
    if pos == (0, 1):   # TOP_MIDDLE — FRONT
        return (t_u, t_v, 1.0)
    if pos == (0, 2):   # TOP_RIGHT — RIGHT
        return (1.0, t_v, -t_u)
    if pos == (1, 0):   # BOTTOM_LEFT — DOWN, ROT_270
        return (-t_v, 1.0, -t_u)
    if pos == (1, 1):   # BOTTOM_MIDDLE — BACK, ROT_90
        return (-t_v, -t_u, -1.0)
    if pos == (1, 2):   # BOTTOM_RIGHT — UP, ROT_270
        return (-t_v, -1.0, t_u)
    raise ValueError(f"not a 3x2 cell: {pos}")


def _cubic_ray(pos: FacePos, t_u: float, t_v: float) -> Tuple[float, float, float]:
    """Ray for cubic (c3x2) cell *pos* at tangent coordinates (t_u, t_v).

    Transcribed from ``cube_to_xyz``'s direction switch (vf_v360.c:1076-1117),
    with the face order fixed by the default ``in_forder="rludfb"`` and no
    rotation (``in_frot="000000"``).
    """
    if pos == (0, 0):   # RIGHT
        return (1.0, t_v, -t_u)
    if pos == (0, 1):   # LEFT
        return (-1.0, t_v, t_u)
    if pos == (0, 2):   # UP  (the -Y face; this module calls it TOP)
        return (t_u, -1.0, t_v)
    if pos == (1, 0):   # DOWN (the +Y face; this module calls it BOTTOM)
        return (t_u, 1.0, -t_v)
    if pos == (1, 1):   # FRONT
        return (t_u, t_v, 1.0)
    if pos == (1, 2):   # BACK
        return (-t_u, t_v, -1.0)
    raise ValueError(f"not a 3x2 cell: {pos}")


def in_face_to_tangent(u_norm: float, projection: str) -> float:
    """Map a normalised in-face coordinate to its tangent-plane coordinate.

    Args:
        u_norm: in-face position in ``[-0.5, +0.5]`` (0 at the face centre).
        projection: ``"eac"`` or ``"cubic"``.

    Returns:
        Tangent-plane coordinate in ``[-1, +1]``.
    """
    if projection == "cubic":
        return 2.0 * u_norm
    if projection == "eac":
        return math.tan(math.pi / 2.0 * u_norm)
    raise ValueError(f"unknown projection: {projection!r}")


def tangent_to_in_face(tangent: float, projection: str) -> float:
    """Inverse of :func:`in_face_to_tangent`."""
    if projection == "cubic":
        return tangent / 2.0
    if projection == "eac":
        return math.atan(tangent) / (math.pi / 2.0)
    raise ValueError(f"unknown projection: {projection!r}")


def face_pixel_to_ray(
    pos: FacePos,
    u_norm: float,
    v_norm: float,
    projection: str,
) -> Tuple[float, float, float]:
    """Ray (not normalised) for a point inside one packed face.

    Args:
        pos: grid cell as ``(row, col)``, row 0 on top.
        u_norm, v_norm: in-face position in ``[-0.5, +0.5]``.
        projection: ``"eac"`` or ``"cubic"`` — selects both the resampling
            function and the face-order/rotation table.
    """
    t_u = in_face_to_tangent(u_norm, projection)
    t_v = in_face_to_tangent(v_norm, projection)
    if projection == "eac":
        return _eac_ray(pos, t_u, t_v)
    if projection == "cubic":
        return _cubic_ray(pos, t_u, t_v)
    raise ValueError(f"unknown projection: {projection!r}")


def ray_to_equirect_uv(ray: Tuple[float, float, float]) -> Tuple[float, float]:
    """Project a ray onto equirectangular ``(u, v)``, both in ``[0, 1]``.

    Matches ffmpeg's ``xyz_to_equirect``: ``u`` from ``atan2(x, z)`` and ``v``
    from ``asin(y/|ray|)``, with ``+Y`` pointing down so ``v`` increases downward.
    """
    x, y, z = ray
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return 0.5, 0.5
    u = math.atan2(x, z) / (2.0 * math.pi) + 0.5
    v = math.asin(max(-1.0, min(1.0, y / norm))) / math.pi + 0.5
    return u, v


def face_grid_position(u_frac: float, v_frac: float) -> Tuple[FacePos, float, float]:
    """Split whole-frame fractional coordinates into a cell plus in-face offsets.

    Args:
        u_frac, v_frac: position across the whole packed frame, in ``[0, 1)``.

    Returns:
        ``(pos, u_norm, v_norm)`` with the offsets in ``[-0.5, +0.5]``.
    """
    col = min(2, max(0, int(u_frac * 3.0)))
    row = min(1, max(0, int(v_frac * 2.0)))
    return (row, col), (u_frac * 3.0 - col) - 0.5, (v_frac * 2.0 - row) - 0.5


# ---------------------------------------------------------------------------
# Internal-border adjacency
# ---------------------------------------------------------------------------

# The seven internal borders of a 3x2 grid, as (cell_a, cell_b). Within-row
# borders are vertical lines in the image; cross-row borders are horizontal.
INTERNAL_BORDERS: Tuple[Tuple[FacePos, FacePos], ...] = (
    ((0, 0), (0, 1)), ((0, 1), (0, 2)),          # top row, vertical borders
    ((1, 0), (1, 1)), ((1, 1), (1, 2)),          # bottom row, vertical borders
    ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)),  # cross-row, horizontal
)

_EDGE_TOL = 1e-6


def _border_rays(border, projection, samples=9):
    """Sample rays along both sides of a packed border, in traversal order."""
    (row_a, col_a), (row_b, col_b) = border
    same_row = row_a == row_b
    out_a, out_b = [], []
    for k in range(samples):
        s = -0.5 + (k + 0.5) / samples  # the coordinate running *along* the border
        if same_row:
            # Vertical border: cell A's right edge meets cell B's left edge.
            out_a.append(face_pixel_to_ray((row_a, col_a), +0.5, s, projection))
            out_b.append(face_pixel_to_ray((row_b, col_b), -0.5, s, projection))
        else:
            # Horizontal border: cell A's bottom edge meets cell B's top edge.
            out_a.append(face_pixel_to_ray((row_a, col_a), s, +0.5, projection))
            out_b.append(face_pixel_to_ray((row_b, col_b), s, -0.5, projection))
    return out_a, out_b


def _unit(ray):
    x, y, z = ray
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


def internal_border_adjacency(projection: str, samples: int = 9) -> Dict[Tuple[FacePos, FacePos], bool]:
    """Which internal borders of *projection*'s packing are real 3D cube edges?

    A packed 3x2 atlas concatenates faces for compactness; only some adjacent
    cells hold faces that actually meet along a shared cube edge **and** present
    that same edge at the shared border. Only those borders can show genuine
    content continuity — the rest butt unrelated parts of the sphere together.

    This is decided numerically rather than by a hand-written table: rays are
    sampled along both sides of each border and the border is a real edge only
    if the two sides trace the *same* set of directions in the same order.

    Returns:
        ``{border: is_real_3d_edge}`` for all seven internal borders.
    """
    result: Dict[Tuple[FacePos, FacePos], bool] = {}
    for border in INTERNAL_BORDERS:
        rays_a, rays_b = _border_rays(border, projection, samples)
        matched = all(
            max(abs(p - q) for p, q in zip(_unit(a), _unit(b))) < _EDGE_TOL
            for a, b in zip(rays_a, rays_b)
        )
        result[border] = bool(matched)
    return result


def real_edge_borders(projection: str) -> Tuple[Tuple[FacePos, FacePos], ...]:
    """The internal borders of *projection* that are genuine shared cube edges."""
    adjacency = internal_border_adjacency(projection)
    return tuple(border for border, is_edge in adjacency.items() if is_edge)
