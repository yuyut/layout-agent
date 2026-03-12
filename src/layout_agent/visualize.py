"""
Matplotlib visualisation of a cube layout.

Outputs
-------
  <out_dir>/layout.png    – annotated layout figure
  <out_dir>/metrics.json  – objective breakdown
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import ProblemData, Solution
from .geometry import cube_center
from .objective import total_cost

logger = logging.getLogger(__name__)


def visualize(
    data: ProblemData,
    sol: Solution,
    out_dir: str | Path = "output",
    show: bool = False,
    title: str = "Cube Layout",
    draw_flows: bool = True,
    filename: str = "layout.png",
) -> None:
    """
    Render the layout and save to `out_dir/filename`.
    Also writes `out_dir/metrics.json` with the objective breakdown.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend unless show=True
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as mpe
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")

    # ── Building outline ──────────────────────────────────────────────────────
    xs, ys = data.outline_polygon.exterior.xy
    ax.fill(xs, ys, alpha=0.07, color="steelblue", zorder=0)
    ax.plot(xs, ys, color="steelblue", linewidth=2.0, label="Building outline", zorder=1)


    # ── Placed cubes ──────────────────────────────────────────────────────────
    cubes_map = {c.id: c for c in data.cubes}

    for cid, pl in sol.placements.items():
        cube = cubes_map[cid]
        w, h = cube.dims(pl.rot)
        raw_color = cube.color.strip().lower() if cube.color else None
        try:
            import matplotlib.colors as mcolors
            facecolor = raw_color if (raw_color and mcolors.is_color_like(raw_color)) else "#aaaaaa"
        except Exception:
            facecolor = "#aaaaaa"

        rect = mpatches.FancyBboxPatch(
            (pl.x, pl.y), w, h,
            boxstyle="square,pad=0",
            linewidth=1.2,
            edgecolor="#222222",
            facecolor=facecolor,
            alpha=0.78,
            zorder=3,
        )
        ax.add_patch(rect)

        # Centre label: ID + optional utility markers
        cx, cy = cube_center(cube, pl)
        suffix = ""
        if cube.needs_water:
            suffix += " W"
        if cube.needs_electricity:
            suffix += " E"
        label = f"{cid}{suffix}"

        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="black",
            path_effects=[mpe.withStroke(linewidth=2.0, foreground="white")],
            zorder=5,
        )

    # ── Flow arrows ───────────────────────────────────────────────────────────
    if draw_flows and data.flows:
        max_i = max(e.intensity for e in data.flows)
        for edge in data.flows:
            if edge.src not in sol.placements or edge.dst not in sol.placements:
                continue
            sc = cube_center(cubes_map[edge.src], sol.placements[edge.src])
            dc = cube_center(cubes_map[edge.dst], sol.placements[edge.dst])
            lw = 0.6 + 3.5 * edge.intensity / max_i
            ax.annotate(
                "",
                xy=dc, xytext=sc,
                arrowprops=dict(
                    arrowstyle="->",
                    color="crimson",
                    lw=lw,
                    alpha=0.65,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=4,
            )
            mx = (sc[0] + dc[0]) / 2
            my = (sc[1] + dc[1]) / 2
            ax.text(
                mx, my, f"{edge.intensity:.0f}",
                fontsize=6, color="darkred", alpha=0.9, zorder=5,
            )

    # ── Score annotation ──────────────────────────────────────────────────────
    cost, bd = total_cost(data, sol, include_deadspace=True)
    ann = (
        f"Total cost : {cost:,.0f}\n"
        f"  Flow     : {bd['flow_cost']:,.0f}\n"
        f"  Grouping : {bd['grouping_cost']:,.0f}\n"
        f"  Compact  : {bd['compactness_mm2']:,.0f} mm²\n"
        f"  Deadspace: {bd['deadspace_frac']:.1%}\n"
        f"  Violations: {bd['hard_violations']}\n"
        f"  Placed   : {bd['cubes_placed']}/{len(data.cubes)}"
    )
    ax.text(
        0.015, 0.985, ann,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#aaa"),
        zorder=10,
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    png_path = out_path / filename
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Layout saved → {png_path}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics_name = filename.replace(".png", "") + "_metrics.json"
    metrics_path = out_path / metrics_name
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in bd.items()},
            f,
            indent=2,
        )
    logger.info(f"Metrics saved → {metrics_path}")
