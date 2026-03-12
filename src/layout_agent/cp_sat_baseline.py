"""
Optional CP-SAT discretised baseline.

Approach
--------
1. Rasterise the outline interior at `grid_step` mm resolution to produce
   a finite set of feasible anchor positions.
2. For each (cube, anchor, rotation) triple that passes the boundary check,
   create a binary variable x[c,a,r].
3. Enforce: exactly one placement per cube.
4. Enforce non-overlap via CP-SAT's native no_overlap_2d constraint on
   optional interval variables.
5. Minimise a linearised approximation of the flow cost.

This baseline is intentionally coarser than the continuous SA solver.
Its purpose is to demonstrate the trade-off between model fidelity and
computational tractability that is central to the assignment discussion.

Usage
-----
    from layout_agent.cp_sat_baseline import run_cpsat
    result = run_cpsat(data, grid_step=150.0, time_limit=60.0)
    if result:
        cpsat_sol, cpsat_metrics = result
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def run_cpsat(
    data,
    grid_step: float = 150.0,
    time_limit: float = 60.0,
    seed: int = 42,
) -> Optional[tuple]:
    """
    Run the CP-SAT baseline.  Returns (Solution, metrics_dict) or None.

    Requires the `ortools` package (pip install ortools).
    Gracefully returns None if ortools is unavailable.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        logger.warning(
            "ortools not available – CP-SAT baseline skipped.  "
            "Install with: pip install ortools"
        )
        return None

    from shapely.geometry import Point as SPoint
    from .models import Solution, Placement
    from .geometry import is_inside, cube_center

    cubes_map = {c.id: c for c in data.cubes}
    poly = data.outline_polygon
    minx, miny, maxx, maxy = poly.bounds

    # ── 1. Generate candidate anchor grid ─────────────────────────────────────
    anchors: list[tuple[float, float]] = []
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if poly.contains(SPoint(x, y)):
                anchors.append((x, y))
            y += grid_step
        x += grid_step

    logger.info(
        f"CP-SAT: {len(anchors)} anchors × {len(data.cubes)} cubes × 2 rotations"
    )

    # ── 2. Filter feasible (cube, anchor_idx, rot) triples ────────────────────
    feasible: list[tuple[str, int, int]] = []
    for cube in data.cubes:
        for rot in [0, 1]:
            for ai, (ax, ay) in enumerate(anchors):
                pl = Placement(cube_id=cube.id, x=ax, y=ay, rot=rot)
                if is_inside(poly, cube, pl):
                    feasible.append((cube.id, ai, rot))

    if not feasible:
        logger.error("CP-SAT: no feasible placements found at this grid resolution.")
        return None

    logger.info(f"CP-SAT: {len(feasible)} feasible (cube, anchor, rot) triples")

    # ── 3. Build CP-SAT model ─────────────────────────────────────────────────
    model = cp_model.CpModel()

    x_vars: dict[tuple[str, int, int], object] = {
        key: model.new_bool_var(f"x_{key[0]}_{key[1]}_{key[2]}")
        for key in feasible
    }

    # Each cube placed exactly once
    for cube in data.cubes:
        cube_vars = [x_vars[k] for k in feasible if k[0] == cube.id]
        if not cube_vars:
            logger.warning(f"CP-SAT: no feasible placement for '{cube.id}' at this resolution.")
            continue
        model.add_exactly_one(cube_vars)

    # ── 4. Non-overlap via optional interval variables ────────────────────────
    all_x_ivs: list = []
    all_y_ivs: list = []

    for (cid, ai, rot), var in x_vars.items():
        cube = cubes_map[cid]
        w, h = cube.dims(rot)
        ax, ay = anchors[ai]

        ix = model.new_optional_fixed_size_interval_var(
            int(round(ax)), int(round(w)), var, f"ix_{cid}_{ai}_{rot}"
        )
        iy = model.new_optional_fixed_size_interval_var(
            int(round(ay)), int(round(h)), var, f"iy_{cid}_{ai}_{rot}"
        )
        all_x_ivs.append(ix)
        all_y_ivs.append(iy)

    model.add_no_overlap_2d(all_x_ivs, all_y_ivs)

    # ── 5. Objective: linearised flow cost ────────────────────────────────────
    # Pre-compute discrete centre positions
    centre: dict[tuple[str, int, int], tuple[float, float]] = {}
    for (cid, ai, rot) in feasible:
        cube = cubes_map[cid]
        w, h = cube.dims(rot)
        ax, ay = anchors[ai]
        centre[(cid, ai, rot)] = (ax + w / 2, ay + h / 2)

    COST_SCALE = 1000   # avoid integer overflow from large mm distances
    obj_terms: list = []

    for edge in data.flows:
        src_keys = [k for k in feasible if k[0] == edge.src]
        dst_keys = [k for k in feasible if k[0] == edge.dst]

        for sk in src_keys:
            for dk in dst_keys:
                sx, sy = centre[sk]
                dx, dy = centre[dk]
                dist_scaled = int(round(
                    math.hypot(dx - sx, dy - sy) * edge.intensity / COST_SCALE
                ))
                # Linearise product of two binary vars:
                # prod = x_vars[sk] AND x_vars[dk]
                prod = model.new_bool_var(
                    f"p_{sk[0]}_{sk[1]}_{sk[2]}_{dk[0]}_{dk[1]}_{dk[2]}"
                )
                model.add_bool_and([x_vars[sk], x_vars[dk]]).only_enforce_if(prod)
                model.add_bool_or(
                    [x_vars[sk].negated(), x_vars[dk].negated()]
                ).only_enforce_if(prod.negated())
                obj_terms.append(dist_scaled * prod)

    if obj_terms:
        model.minimize(sum(obj_terms))

    # ── 6. Solve ──────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning(
            f"CP-SAT: no feasible solution found within time limit "
            f"(status={status_name}).  Try increasing grid_step or time_limit."
        )
        return None

    # ── 7. Extract solution ───────────────────────────────────────────────────
    pls: dict[str, Placement] = {}
    for (cid, ai, rot), var in x_vars.items():
        if solver.boolean_value(var):
            ax, ay = anchors[ai]
            pls[cid] = Placement(cube_id=cid, x=ax, y=ay, rot=rot)

    sol = Solution(placements=pls)
    metrics = {
        "status":       status_name,
        "wall_time_s":  solver.wall_time,
        "cubes_placed": len(pls),
        "grid_step_mm": grid_step,
        "n_anchors":    len(anchors),
        "n_feasible":   len(feasible),
    }
    logger.info(
        f"CP-SAT: {len(pls)}/{len(data.cubes)} cubes placed  "
        f"status={status_name}  t={solver.wall_time:.1f}s"
    )
    return sol, metrics
