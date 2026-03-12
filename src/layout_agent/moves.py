"""
Neighbourhood move generators for simulated annealing.

Five move types:
  translate     – shift one cube by (dx, dy)
  rotate        – flip a cube's orientation (rot 0 ↔ 1)
  swap          – exchange positions of two cubes
  flow_pull     – move one cube toward its highest-flow neighbour
  cluster_shift – shift a same-colour group or flow neighbourhood together
"""
from __future__ import annotations

import math
import random

from .models import ProblemData, Solution, Placement


# ── Apply a move (immutable) ──────────────────────────────────────────────────

def apply_move(sol: Solution, move: dict) -> Solution:
    """Return a new Solution with the move applied; original is unchanged."""
    new_pls = dict(sol.placements)
    mtype = move["type"]

    if mtype in ("translate", "rotate", "flow_pull"):
        cid = move["cube_id"]
        new_pls[cid] = move["placement"]

    elif mtype == "swap":
        a, b = move["a"], move["b"]
        pa, pb = sol.placements[a], sol.placements[b]
        new_pls[a] = Placement(cube_id=a, x=pb.x, y=pb.y, rot=pa.rot)
        new_pls[b] = Placement(cube_id=b, x=pa.x, y=pa.y, rot=pb.rot)

    elif mtype == "cluster_shift":
        dx, dy = move["dx"], move["dy"]
        for cid in move["cube_ids"]:
            old = sol.placements[cid]
            new_pls[cid] = Placement(cube_id=cid, x=old.x + dx, y=old.y + dy, rot=old.rot)

    return Solution(placements=new_pls)


# ── Step size ─────────────────────────────────────────────────────────────────

def _step(T: float) -> float:
    """
    Perturbation magnitude (mm) that decreases as temperature falls.
    At T=1e5 → ~200 mm (large exploratory moves).
    At T=100  → ~20 mm.
    At T=1    → ~1 mm  (fine-grained tuning).
    """
    return max(1.0, min(300.0, T / 500.0))


# ── Move proposal ─────────────────────────────────────────────────────────────

def propose_move(
    data: ProblemData,
    sol: Solution,
    rng: random.Random,
    T: float,
) -> dict:
    """
    Sample one move uniformly at random from the five move types
    (weighted by expected usefulness).
    """
    cubes_map = {c.id: c for c in data.cubes}
    ids = list(sol.placements.keys())

    mtype = rng.choices(
        ["translate", "rotate", "swap", "flow_pull", "cluster_shift"],
        weights=[40, 15, 20, 15, 10],
    )[0]

    step = _step(T)

    # ── translate ──────────────────────────────────────────────────────────────
    if mtype == "translate":
        cid = rng.choice(ids)
        pl  = sol.placements[cid]
        dx  = rng.uniform(-step, step)
        dy  = rng.uniform(-step, step)
        return {
            "type":      "translate",
            "cube_id":   cid,
            "placement": Placement(cube_id=cid, x=pl.x + dx, y=pl.y + dy, rot=pl.rot),
        }

    # ── rotate ─────────────────────────────────────────────────────────────────
    elif mtype == "rotate":
        cid = rng.choice(ids)
        pl  = sol.placements[cid]
        # Keep anchor corner; swap width/height in place
        return {
            "type":      "rotate",
            "cube_id":   cid,
            "placement": Placement(cube_id=cid, x=pl.x, y=pl.y, rot=1 - pl.rot),
        }

    # ── swap ───────────────────────────────────────────────────────────────────
    elif mtype == "swap":
        if len(ids) < 2:
            return propose_move(data, sol, rng, T)   # fallback
        a, b = rng.sample(ids, 2)
        return {"type": "swap", "a": a, "b": b}

    # ── flow_pull ──────────────────────────────────────────────────────────────
    elif mtype == "flow_pull":
        if not data.flows:
            return propose_move(data, sol, rng, T)

        # Sample edge weighted by intensity
        weights = [e.intensity for e in data.flows]
        edge = rng.choices(data.flows, weights=weights)[0]
        src, dst = edge.src, edge.dst
        if src not in sol.placements or dst not in sol.placements:
            return propose_move(data, sol, rng, T)

        # Choose which endpoint to move toward the other
        cid   = rng.choice([src, dst])
        other = dst if cid == src else src

        pl_c  = sol.placements[cid]
        pl_o  = sol.placements[other]
        cube  = cubes_map[cid]
        ocube = cubes_map[other]

        w_c, h_c = cube.dims(pl_c.rot)
        w_o, h_o = ocube.dims(pl_o.rot)

        cx = pl_c.x + w_c / 2
        cy = pl_c.y + h_c / 2
        ox = pl_o.x + w_o / 2
        oy = pl_o.y + h_o / 2

        dist = math.hypot(ox - cx, oy - cy)
        if dist < 1.0:
            return propose_move(data, sol, rng, T)

        # Move by at most `step` mm toward the other cube's centre
        frac = min(step / dist, 0.5)
        new_cx = cx + frac * (ox - cx)
        new_cy = cy + frac * (oy - cy)

        return {
            "type":      "flow_pull",
            "cube_id":   cid,
            "placement": Placement(
                cube_id=cid,
                x=new_cx - w_c / 2,
                y=new_cy - h_c / 2,
                rot=pl_c.rot,
            ),
        }

    # ── cluster_shift ──────────────────────────────────────────────────────────
    else:
        cube_colors = {c.id: c.color for c in data.cubes}
        colors = list({cube_colors[cid] for cid in ids if cube_colors.get(cid)})
        if colors:
            target_color = rng.choice(colors)
            group = [cid for cid in ids if cube_colors.get(cid) == target_color]
        else:
            group = rng.sample(ids, max(1, len(ids) // 4))

        dx = rng.uniform(-step / 2, step / 2)
        dy = rng.uniform(-step / 2, step / 2)
        return {"type": "cluster_shift", "cube_ids": group, "dx": dx, "dy": dy}
