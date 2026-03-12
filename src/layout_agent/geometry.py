"""
Core geometric operations.

Design principle: no Shapely in the hot-path.
- Rectangle–rectangle overlap: AABB arithmetic.
- Rectangle-inside-polygon: pure Python ray-casting + edge-intersection.
- Shapely is used only for:
    • loading / repairing the polygon from DXF (once at startup)
    • dead_space_score (called only in the final report)
    • derive_utility_zones (called once at startup)
"""
from __future__ import annotations

import math
from typing import Optional

from shapely.geometry import Polygon as SPolygon, box as sbox
from shapely.ops import unary_union

from .models import Cube, Placement


# ── Centre computation ────────────────────────────────────────────────────────

def cube_center(cube: Cube, pl: Placement) -> tuple[float, float]:
    """Return the centre (cx, cy) of a placed cube in mm."""
    w, h = cube.dims(pl.rot)
    return pl.x + w / 2.0, pl.y + h / 2.0


# ── Pure-Python polygon primitives (no Shapely) ───────────────────────────────

def _pip(px: float, py: float, verts: list[tuple[float, float]]) -> bool:
    """
    Ray-casting point-in-polygon test.
    Returns True if (px, py) is strictly inside the polygon defined by `verts`.
    O(n) in the number of vertices; no heap allocation.
    """
    inside = False
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _cross2d(ox: float, oy: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Signed 2-D cross product (OA × OB)."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _segments_cross(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """
    True iff segment AB and segment CD properly cross (not just touch).
    Uses the signed cross-product straddle test.
    """
    d1 = _cross2d(cx, cy, dx, dy, ax, ay)
    d2 = _cross2d(cx, cy, dx, dy, bx, by)
    d3 = _cross2d(ax, ay, bx, by, cx, cy)
    d4 = _cross2d(ax, ay, bx, by, dx, dy)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


def _rect_inside_verts(
    rx0: float, ry0: float, rx1: float, ry1: float,
    verts: list[tuple[float, float]],
) -> bool:
    """
    True iff the axis-aligned rectangle [rx0,ry0,rx1,ry1] is fully inside the
    polygon described by `verts` (a closed list; last vertex ≠ first).

    Two conditions are required (polygon may be non-convex):
      1. All 4 corners of the rectangle are inside the polygon.
      2. No polygon edge crosses any rectangle edge.
    """
    # Fast bounding-box rejection against polygon's own bbox
    # (verts bbox is cached outside, but this is cheap regardless)
    # 1. All corners inside
    for cx, cy in ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)):
        if not _pip(cx, cy, verts):
            return False

    # 2. No polygon edge crosses any rectangle edge
    n = len(verts)
    rect_edges = (
        (rx0, ry0, rx1, ry0),
        (rx1, ry0, rx1, ry1),
        (rx1, ry1, rx0, ry1),
        (rx0, ry1, rx0, ry0),
    )
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        for ex1, ey1, ex2, ey2 in rect_edges:
            if _segments_cross(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
                return False

    return True


# ── Outline preparation ───────────────────────────────────────────────────────

def build_outline_verts(poly: SPolygon) -> list[tuple[float, float]]:
    """
    Extract polygon vertices as a plain Python list of (x, y) floats.
    Call this once after loading the DXF and store in ProblemData.
    """
    return [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]


# ── Shapely-free feasibility predicates ──────────────────────────────────────

def is_inside_verts(
    verts: list[tuple[float, float]],
    cube: Cube,
    pl: Placement,
) -> bool:
    """True iff the placed cube rectangle lies fully inside the polygon (verts form)."""
    w, h = cube.dims(pl.rot)
    return _rect_inside_verts(pl.x, pl.y, pl.x + w, pl.y + h, verts)


def overlaps(cube_a: Cube, pa: Placement, cube_b: Cube, pb: Placement) -> bool:
    """
    True iff two placements geometrically overlap.
    Pure AABB arithmetic — no Shapely.  Touching edges are allowed.
    """
    wa, ha = cube_a.dims(pa.rot)
    wb, hb = cube_b.dims(pb.rot)
    return not (
        pa.x + wa <= pb.x or pb.x + wb <= pa.x or
        pa.y + ha <= pb.y or pb.y + hb <= pa.y
    )


def is_single_feasible(
    verts: list[tuple[float, float]],
    cube: Cube,
    pl: Placement,
    other_placements: dict[str, Placement],
    cubes_map: dict[str, Cube],
) -> bool:
    """
    Incremental feasibility check for ONE placement.
    Entirely Shapely-free (ray-cast + AABB arithmetic).
    """
    if not is_inside_verts(verts, cube, pl):
        return False
    for oid, opl in other_placements.items():
        if overlaps(cube, pl, cubes_map[oid], opl):
            return False
    return True


def count_violations(
    outline_verts: list[tuple[float, float]],
    cubes_map: dict[str, Cube],
    placements: dict[str, Placement],
) -> int:
    """
    Count hard-constraint violations (boundary + pairwise overlap).
    Entirely Shapely-free.
    """
    violations = 0
    ids = list(placements.keys())

    for pl in placements.values():
        if not is_inside_verts(outline_verts, cubes_map[pl.cube_id], pl):
            violations += 1

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if overlaps(cubes_map[ids[i]], placements[ids[i]],
                        cubes_map[ids[j]], placements[ids[j]]):
                violations += 1

    return violations


# ── Shapely helpers for non-hot-path code (repair, etc.) ─────────────────────

def placement_polygon(cube: Cube, pl: Placement) -> SPolygon:
    """Shapely Box for the placed cube. Used by repair / visualisation only."""
    w, h = cube.dims(pl.rot)
    return sbox(pl.x, pl.y, pl.x + w, pl.y + h)


def is_inside(poly: SPolygon, cube: Cube, pl: Placement) -> bool:
    """Shapely-based boundary check — use only outside the SA hot loop."""
    w, h = cube.dims(pl.rot)
    return poly.contains(sbox(pl.x, pl.y, pl.x + w, pl.y + h))


# ── Scoring helpers ───────────────────────────────────────────────────────────

def bounding_box_area(cubes_map: dict[str, Cube], placements: dict[str, Placement]) -> float:
    """Bounding-box area of all placed cubes (mm²) — Shapely-free."""
    xs: list[float] = []
    ys: list[float] = []
    for pl in placements.values():
        w, h = cubes_map[pl.cube_id].dims(pl.rot)
        xs += [pl.x, pl.x + w]
        ys += [pl.y, pl.y + h]
    if not xs:
        return 0.0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def dead_space_score(outline_poly: SPolygon, cubes_map: dict[str, Cube],
                     placements: dict[str, Placement]) -> float:
    """
    Fraction of the outline area NOT covered by any cube [0, 1].
    Uses Shapely union — called only in the final report (include_deadspace=True).
    """
    shapes = [
        sbox(pl.x, pl.y, pl.x + cubes_map[pl.cube_id].dims(pl.rot)[0],
             pl.y + cubes_map[pl.cube_id].dims(pl.rot)[1])
        for pl in placements.values()
    ]
    if not shapes:
        return 1.0
    covered = unary_union(shapes).intersection(outline_poly)
    return 1.0 - covered.area / outline_poly.area


