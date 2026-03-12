"""
Local repair for the greedy initialiser.

Two repair phases:
  1. Snap cubes outside the boundary toward the polygon centroid.
  2. Resolve pairwise overlaps with small random nudges.

All boundary checks use the pure-Python ray-cast (is_inside_verts) — no Shapely.
The repair function is NOT called inside the SA hot loop (too expensive);
infeasible SA moves are simply rejected.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Optional

from .models import ProblemData, Solution, Placement
from .geometry import is_inside_verts, overlaps
from .config import REPAIR_MAX_TRIES, REPAIR_NUDGE

logger = logging.getLogger(__name__)


def _poly_centroid(verts: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute the centroid of a polygon given as a vertex list."""
    n = len(verts)
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n
    return cx, cy


def _snap_inside(
    verts: list[tuple[float, float]],
    centroid: tuple[float, float],
    pl: Placement,
    cube,
) -> Optional[Placement]:
    """
    Nudge `pl` toward the polygon centroid until the cube fits fully inside.
    Pure-Python boundary check (no Shapely).
    """
    if is_inside_verts(verts, cube, pl):
        return pl

    cx_poly, cy_poly = centroid
    w, h = cube.dims(pl.rot)
    cx_cube = pl.x + w / 2
    cy_cube = pl.y + h / 2

    dx = cx_poly - cx_cube
    dy = cy_poly - cy_cube
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return None

    ux, uy = dx / dist, dy / dist
    step = REPAIR_NUDGE

    for _ in range(REPAIR_MAX_TRIES):
        new_pl = Placement(cube_id=pl.cube_id,
                           x=pl.x + step * ux, y=pl.y + step * uy, rot=pl.rot)
        if is_inside_verts(verts, cube, new_pl):
            return new_pl
        step += REPAIR_NUDGE

    return None


def repair(
    data: ProblemData,
    sol: Solution,
    rng: random.Random,
) -> Optional[Solution]:
    """
    Attempt to repair all hard-constraint violations.
    Returns a feasible Solution or None if repair fails.
    Called by the greedy initialiser; NOT used in the SA main loop.
    """
    verts = data.outline_verts
    centroid = _poly_centroid(verts)
    cubes_map = {c.id: c for c in data.cubes}
    pls: dict[str, Placement] = dict(sol.placements)

    # Phase 1: snap out-of-bounds cubes
    for cid, pl in list(pls.items()):
        cube = cubes_map[cid]
        if not is_inside_verts(verts, cube, pl):
            fixed = _snap_inside(verts, centroid, pl, cube)
            if fixed is None:
                return None
            pls[cid] = fixed

    # Phase 2: resolve pairwise overlaps
    ids = list(pls.keys())
    for _attempt in range(REPAIR_MAX_TRIES):
        any_overlap = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ca, cb = cubes_map[a], cubes_map[b]
                if not overlaps(ca, pls[a], cb, pls[b]):
                    continue
                any_overlap = True
                nudge_id = a if ca.area <= cb.area else b
                pl = pls[nudge_id]
                angle = rng.uniform(0.0, 2 * math.pi)
                new_pl = Placement(
                    cube_id=nudge_id,
                    x=pl.x + REPAIR_NUDGE * math.cos(angle),
                    y=pl.y + REPAIR_NUDGE * math.sin(angle),
                    rot=pl.rot,
                )
                if is_inside_verts(verts, cubes_map[nudge_id], new_pl):
                    pls[nudge_id] = new_pl

        if not any_overlap:
            return Solution(placements=pls)

    return None
