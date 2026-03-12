"""
Simulated-annealing optimiser — Shapely-free in the hot path.

Feasibility
-----------
* Single-cube moves: `is_single_feasible` (ray-cast + AABB, no Shapely).
* Swap / cluster moves: `count_violations` (ray-cast + AABB, no Shapely).

Objective
---------
* `total_cost(..., n_violations=0)` skips re-counting violations for accepted
  feasible candidates — saves O(n²) arithmetic every iteration.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Callable, Optional

from .models import ProblemData, Solution
from .objective import total_cost
from .moves import propose_move, apply_move
from .geometry import is_single_feasible, count_violations
from .config import SA_T_START, SA_T_END, SA_ALPHA, SA_MAX_ITER

logger = logging.getLogger(__name__)


def simulated_annealing(
    data: ProblemData,
    start: Solution,
    seed: int = 42,
    T_start: float = SA_T_START,
    T_end: float = SA_T_END,
    alpha: float = SA_ALPHA,
    max_iter: int = SA_MAX_ITER,
    on_progress: Optional[Callable[[int, float, float], None]] = None,
) -> tuple[Solution, dict]:
    rng = random.Random(seed)
    cubes_map = {c.id: c for c in data.cubes}
    verts = data.outline_verts          # pre-extracted polygon vertices

    cur = start
    cur_cost, _ = total_cost(data, cur, n_violations=0)
    best = cur
    best_cost = cur_cost

    T = T_start
    n_accepted = n_rejected = 0
    log_interval = max(1, max_iter // 20)

    # ── Convergence history (sampled every history_interval iterations) ────────
    history_interval = max(1, max_iter // 300)
    cost_history: list[float] = []
    temp_history: list[float] = []
    accept_rate_history: list[float] = []
    _win_accepted = 0   # accepted moves in the current window (Metropolis test passed)
    _win_feasible = 0   # feasible moves in the current window (reached Metropolis test)

    for iteration in range(max_iter):
        T = max(T_end, T * alpha)

        move = propose_move(data, cur, rng, T)
        candidate = apply_move(cur, move)
        mtype = move["type"]

        # ── Feasibility (Shapely-free) ────────────────────────────────────────
        if mtype in ("translate", "rotate", "flow_pull"):
            cid    = move["cube_id"]
            others = {k: v for k, v in candidate.placements.items() if k != cid}
            feasible = is_single_feasible(
                verts, cubes_map[cid], candidate.placements[cid], others, cubes_map
            )
        else:
            feasible = count_violations(verts, cubes_map, candidate.placements) == 0

        if feasible:
            # ── Objective (skip violation count — already verified feasible) ──
            cand_cost, _ = total_cost(data, candidate, n_violations=0)
            delta = cand_cost - cur_cost
            _win_feasible += 1

            if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                cur = candidate
                cur_cost = cand_cost
                n_accepted += 1
                _win_accepted += 1
                if cur_cost < best_cost:
                    best = cur
                    best_cost = cur_cost
            else:
                n_rejected += 1
        else:
            n_rejected += 1

        # ── Sample history ────────────────────────────────────────────────────
        if (iteration + 1) % history_interval == 0:
            cost_history.append(best_cost)
            temp_history.append(T)
            accept_rate_history.append(_win_accepted / max(1, _win_feasible))
            _win_accepted = 0
            _win_feasible = 0

        if iteration % log_interval == 0:
            logger.info(
                f"  iter {iteration:6d}  T={T:10.1f}  "
                f"cur={cur_cost:12.0f}  best={best_cost:12.0f}"
            )
            if on_progress:
                on_progress(iteration, T, best_cost)

    _, breakdown = total_cost(data, best, include_deadspace=True)
    metrics = {
        **breakdown,
        "T_start":              T_start,
        "T_end_actual":         T,
        "alpha":                alpha,
        "iterations":           max_iter,
        "n_accepted":           n_accepted,
        "n_rejected":           n_rejected,
        "accept_rate":          n_accepted / max(1, max_iter),
        "cost_history":         cost_history,
        "temp_history":         temp_history,
        "accept_rate_history":  accept_rate_history,
        "history_interval":     history_interval,
    }
    logger.info(
        f"SA finished  best={best_cost:.0f}  "
        f"accept={n_accepted}  reject={n_rejected}"
    )
    return best, metrics
