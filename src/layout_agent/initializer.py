"""
Greedy constructive initialiser.

Cubes are sorted by a priority score (area + flow degree + utility needs) and
placed one by one.  For each cube the best feasible anchor from a candidate
set (outline grid + polygon corners + induced corners from already-placed
cubes) is chosen using a cheap partial objective.
"""
from __future__ import annotations

import logging
import random

from shapely.geometry import Point as SPoint

from .models import ProblemData, Placement, Solution, Cube
from .geometry import is_inside_verts, overlaps, cube_center
from .objective import flow_cost, grouping_cost
from .config import INIT_GRID_STEP, INIT_TRIES_PER_CUBE, INIT_FEASIBLE_PER_CUBE, INIT_COMPACT_WEIGHT

logger = logging.getLogger(__name__)


# ── Priority scoring ──────────────────────────────────────────────────────────

def _priority(cube: Cube, data: ProblemData, *, noise: float = 0.0) -> float:
    """
    Higher score → place earlier.

    Three factors are normalised to roughly equal weight so no single term
    dominates by accident (cube.area is in mm² and can be 50 000+, while
    utility is 0/1/2):
      - area       normalised to [0, 1] by max area across all cubes
      - flow_deg   normalised by max total intensity seen
      - utility    0/1/2 already small; kept as-is on 0-2 scale
    Optional Gaussian noise is added on restarts to break ties differently
    without discarding priority structure entirely.
    """
    max_area = max(c.area for c in data.cubes) or 1.0
    max_flow = max(
        (sum(e.intensity for e in data.flows if e.src == c.id or e.dst == c.id)
         for c in data.cubes),
        default=1.0,
    ) or 1.0

    flow_deg = sum(e.intensity for e in data.flows if e.src == cube.id or e.dst == cube.id)
    utility = int(cube.needs_water) + int(cube.needs_electricity)

    score = (
        cube.area / max_area          # 0–1
        + flow_deg / max_flow         # 0–1
        + utility / 2.0               # 0–1
    )
    return score + noise


# ── Anchor generation ─────────────────────────────────────────────────────────

def _candidate_anchors(
    data: ProblemData,
    placed: dict[str, Placement],
    cubes_map: dict[str, Cube],
) -> list[tuple[float, float]]:
    """
    Candidate bottom-left anchor positions:
    1. Coarse bounding-box grid filtered to interior of outline polygon.
    2. Polygon exterior vertices.
    3. Right and top edges of already-placed cubes (corner-packing heuristic).
    """
    poly = data.outline_polygon
    minx, miny, maxx, maxy = poly.bounds
    anchors: list[tuple[float, float]] = []

    # 1. Interior grid
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if poly.contains(SPoint(x, y)):
                anchors.append((x, y))
            y += INIT_GRID_STEP
        x += INIT_GRID_STEP

    # 2. Polygon vertices
    for cx, cy in list(poly.exterior.coords):
        anchors.append((cx, cy))

    # 3. Induced corners from placed cubes
    for pl in placed.values():
        cube = cubes_map[pl.cube_id]
        w, h = cube.dims(pl.rot)
        anchors.extend([
            (pl.x + w, pl.y),
            (pl.x,     pl.y + h),
            (pl.x + w, pl.y + h),
        ])

    # Deduplicate: round to nearest mm so near-duplicate grid/vertex/corner
    # anchors don't waste feasibility checks.
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[float, float]] = []
    for ax, ay in anchors:
        key = (round(ax), round(ay))
        if key not in seen:
            seen.add(key)
            unique.append((ax, ay))
    return unique


# ── Partial objective (cheap ranking) ─────────────────────────────────────────

def _partial_cost(
    cube: Cube,
    pl: Placement,
    data: ProblemData,
    partial_sol: Solution,
) -> float:
    """
    flow + grouping cost + lightweight centroid-distance compactness proxy.

    The centroid proxy pulls candidates toward the already-placed mass centre,
    reducing scattered initial layouts without requiring bounding-box union.
    It is skipped when nothing is placed yet (no centroid to compute).
    """
    test_sol = Solution(placements={**partial_sol.placements, cube.id: pl})
    base = flow_cost(data, test_sol) + grouping_cost(data, test_sol)

    placed = partial_sol.placements
    if not placed:
        return base

    cubes_map = {c.id: c for c in data.cubes}
    cx = sum(cube_center(cubes_map[pid], ppl)[0] for pid, ppl in placed.items()) / len(placed)
    cy = sum(cube_center(cubes_map[pid], ppl)[1] for pid, ppl in placed.items()) / len(placed)
    nx, ny = cube_center(cube, pl)
    dist_to_centroid = ((nx - cx) ** 2 + (ny - cy) ** 2) ** 0.5

    return base + INIT_COMPACT_WEIGHT * dist_to_centroid


# ── Main initialiser ──────────────────────────────────────────────────────────

def greedy_initialize(
    data: ProblemData,
    seed: int = 42,
    n_restarts: int = 3,
) -> Solution:
    """
    Greedy constructive placement with multiple random restarts.
    Returns the restart that successfully placed the most cubes.
    """
    cubes_map = {c.id: c for c in data.cubes}
    rng = random.Random(seed)
    best_sol: Solution | None = None
    best_placed = -1

    for restart in range(n_restarts):
        placed: dict[str, Placement] = {}
        if restart == 0:
            # First pass: pure priority order
            order = sorted(data.cubes, key=lambda c: _priority(c, data), reverse=True)
        else:
            # Later restarts: add small Gaussian noise to priority scores so
            # cubes shift within their priority band rather than fully shuffling.
            # This preserves heuristic structure while exploring different orders.
            noise_scale = 0.15  # ~15 % of the 0-3 score range
            order = sorted(
                data.cubes,
                key=lambda c: _priority(c, data, noise=rng.gauss(0, noise_scale)),
                reverse=True,
            )

        for cube in order:
            anchors = _candidate_anchors(data, placed, cubes_map)
            rng.shuffle(anchors)

            best_pl: Placement | None = None
            best_score = float("inf")
            partial_sol = Solution(placements=placed)

            tries = 0        # total anchor×rot attempts (runtime cap)
            feasible = 0     # feasible placements scored (quality cap)
            for ax, ay in anchors:
                if tries >= INIT_TRIES_PER_CUBE or feasible >= INIT_FEASIBLE_PER_CUBE:
                    break
                for rot in [0, 1]:
                    tries += 1
                    pl = Placement(cube_id=cube.id, x=ax, y=ay, rot=rot)

                    if not is_inside_verts(data.outline_verts, cube, pl):
                        continue
                    if any(
                        overlaps(cube, pl, cubes_map[pid], ppl)
                        for pid, ppl in placed.items()
                    ):
                        continue

                    score = _partial_cost(cube, pl, data, partial_sol)
                    feasible += 1
                    if score < best_score:
                        best_score = score
                        best_pl = pl

            if best_pl is not None:
                placed[cube.id] = best_pl
            else:
                logger.warning(
                    f"Restart {restart}: could not place '{cube.id}' "
                    f"({cube.xdim:.0f}×{cube.ydim:.0f} mm)"
                )

        sol = Solution(placements=placed)
        n = len(sol.placements)
        logger.info(f"Greedy restart {restart}: placed {n}/{len(data.cubes)} cubes")

        if n > best_placed:
            best_placed = n
            best_sol = sol

    if best_sol is None:
        return Solution()

    logger.info(f"Best initial solution: {best_placed}/{len(data.cubes)} cubes placed")
    return best_sol
