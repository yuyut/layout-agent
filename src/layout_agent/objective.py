"""
Multi-term objective function — Shapely-free in the hot path.

total_cost =
    W_FLOW      * flow_cost
  + W_COMPACT   * bounding_box_area
  + W_DEADSPACE * dead_space_score   (only when include_deadspace=True)
  + W_COLOR     * grouping_cost      (colour + utility-need clusters)
  - W_CONTACT   * contact_reward     (reward for touching edges between cubes)
  + HARD_PENALTY * n_violations
"""
from __future__ import annotations

import math
from typing import Optional

from .models import ProblemData, Solution
from .geometry import (
    cube_center,
    count_violations,
    bounding_box_area,
    dead_space_score,
)
from .config import (
    W_FLOW, W_COMPACT, W_DEADSPACE, W_COLOR, W_CONTACT,
    HARD_PENALTY,
)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


# ── Individual cost terms ─────────────────────────────────────────────────────

def flow_cost(data: ProblemData, sol: Solution) -> float:
    cubes = {c.id: c for c in data.cubes}
    cost = 0.0
    for edge in data.flows:
        if edge.src not in sol.placements or edge.dst not in sol.placements:
            continue
        ax, ay = cube_center(cubes[edge.src], sol.placements[edge.src])
        bx, by = cube_center(cubes[edge.dst], sol.placements[edge.dst])
        cost += edge.intensity * _dist(ax, ay, bx, by)
    return cost


def grouping_cost(data: ProblemData, sol: Solution) -> float:
    """
    Pairwise-distance cost that penalises spreading cubes of the same group.

    Groups are formed by:
      - colour  (cubes sharing the same non-None color value)
      - water   (all cubes that need water)
      - electricity (all cubes that need electricity)

    Each cube can belong to multiple groups (e.g. same colour AND needs water).
    """
    cubes = {c.id: c for c in data.cubes}

    groups: dict[str, list[str]] = {}
    for cube in data.cubes:
        if cube.id not in sol.placements:
            continue
        if cube.color:
            groups.setdefault(f"color:{cube.color}", []).append(cube.id)
        if cube.needs_water:
            groups.setdefault("utility:water", []).append(cube.id)
        if cube.needs_electricity:
            groups.setdefault("utility:electricity", []).append(cube.id)

    cost = 0.0
    for cids in groups.values():
        if len(cids) < 2:
            continue
        centers = [cube_center(cubes[cid], sol.placements[cid]) for cid in cids]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                cost += _dist(*centers[i], *centers[j])
    return cost


def contact_reward(cubes_map: dict, placements: dict, tol: float = 1.0) -> float:
    """
    Sum of shared edge lengths between touching cube pairs. Pure AABB arithmetic.

    Two cubes are touching when the gap between their edges is ≤ tol (default 1 mm).
    Reward = length of the shared segment along the perpendicular axis.
    This incentivises tight packing without requiring bounding-box union.
    """
    items = list(placements.items())
    reward = 0.0
    for i in range(len(items)):
        cid_a, pa = items[i]
        wa, ha = cubes_map[cid_a].dims(pa.rot)
        ax0, ay0, ax1, ay1 = pa.x, pa.y, pa.x + wa, pa.y + ha
        for j in range(i + 1, len(items)):
            cid_b, pb = items[j]
            wb, hb = cubes_map[cid_b].dims(pb.rot)
            bx0, by0, bx1, by1 = pb.x, pb.y, pb.x + wb, pb.y + hb
            # Vertical shared edge (cubes touch along x)
            if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
                reward += max(0.0, min(ay1, by1) - max(ay0, by0))
            # Horizontal shared edge (cubes touch along y)
            if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
                reward += max(0.0, min(ax1, bx1) - max(ax0, bx0))
    return reward


# ── Composite objective ───────────────────────────────────────────────────────

def total_cost(
    data: ProblemData,
    sol: Solution,
    *,
    include_deadspace: bool = False,
    n_violations: Optional[int] = None,
) -> tuple[float, dict]:
    """
    Compute the full objective and return (total, breakdown_dict).

    n_violations : if provided, skip counting violations and use this value.
                   Pass 0 in the SA hot loop once feasibility is verified —
                   avoids O(n²) work per iteration.
    """
    cubes_map = {c.id: c for c in data.cubes}

    fc  = flow_cost(data, sol)
    bb  = bounding_box_area(cubes_map, sol.placements)
    ds  = dead_space_score(data.outline_polygon, cubes_map, sol.placements) \
          if include_deadspace else 0.0
    gc  = grouping_cost(data, sol)
    cr  = contact_reward(cubes_map, sol.placements)

    if n_violations is None:
        viols = count_violations(data.outline_verts, cubes_map, sol.placements)
    else:
        viols = n_violations

    total = (
        W_FLOW        * fc
        + W_COMPACT   * bb
        + W_DEADSPACE * ds
        + W_COLOR     * gc
        - W_CONTACT   * cr
        + HARD_PENALTY * viols
    )
    return total, {
        "total":           total,
        "flow_cost":       fc,
        "grouping_cost":   gc,
        "compactness_mm2": bb,
        "contact_reward":  cr,
        "deadspace_frac":  ds,
        "hard_violations": viols,
        "penalty":         HARD_PENALTY * viols,
        "cubes_placed":    len(sol.placements),
    }
