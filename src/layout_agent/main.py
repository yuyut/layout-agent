"""
Facility layout optimiser – main entry point.

Usage
-----
    python -m layout_agent.main --dxf data/outline.dxf --xlsx data/cubes.xlsx

Common options
--------------
  --out OUTPUT_DIR      Output directory (default: output/)
  --seed INT            Random seed (default: 42)
  --iters INT           SA iterations (default: from config.py)
  --T-start FLOAT       SA initial temperature
  --alpha FLOAT         SA cooling factor
  --cpsat               Also run CP-SAT baseline for comparison
  --show                Open the plot interactively (requires a display)
  --no-flows            Omit flow arrows from the visualisation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("layout_agent.main")


def _print_summary(sol, data, label: str) -> None:
    from .objective import total_cost
    cost, bd = total_cost(data, sol, include_deadspace=True)
    logger.info(
        f"[{label}]  cost={cost:,.0f}  "
        f"flow={bd['flow_cost']:,.0f}  grouping={bd['grouping_cost']:,.0f}  "
        f"violations={bd['hard_violations']}  "
        f"placed={bd['cubes_placed']}/{len(data.cubes)}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Place rectangular cubes optimally inside a 2D building outline."
    )
    parser.add_argument("--dxf",  required=True, help="Path to DXF building outline")
    parser.add_argument("--xlsx", required=True, help="Path to Excel workbook (cubes/flows)")
    parser.add_argument("--out",  default="output", help="Output directory")
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument("--iters", type=int,  default=None,
                        help="SA iterations (default: SA_MAX_ITER in config.py)")
    parser.add_argument("--T-start", dest="T_start", type=float, default=None)
    parser.add_argument("--alpha",   type=float, default=None)
    parser.add_argument("--cpsat",   action="store_true",
                        help="Run CP-SAT discretised baseline after SA")
    parser.add_argument("--cpsat-grid", dest="cpsat_grid", type=float, default=None,
                        help="Grid step for CP-SAT (mm, default: CPSAT_GRID_STEP)")
    parser.add_argument("--cpsat-time", dest="cpsat_time", type=float, default=None,
                        help="CP-SAT time limit (s, default: CPSAT_TIME_LIMIT)")
    parser.add_argument("--show",    action="store_true",
                        help="Show plot interactively")
    parser.add_argument("--no-flows", dest="no_flows", action="store_true",
                        help="Omit flow arrows in visualisation")
    args = parser.parse_args(argv)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading DXF outline …")
    from .io_dxf import load_outline
    outline = load_outline(args.dxf)

    logger.info("Loading Excel workbook …")
    from .io_excel import load_problem_excel
    cubes, flows = load_problem_excel(args.xlsx)

    from .models import ProblemData
    data = ProblemData(
        outline_polygon=outline,
        cubes=cubes,
        flows=flows,
    )

    logger.info(
        f"Problem: {len(cubes)} cubes, {len(flows)} flow edges, "
        f"outline {outline.bounds[2]:.0f} × {outline.bounds[3]:.0f} mm"
    )

    # ── Greedy initialisation ─────────────────────────────────────────────────
    logger.info("Greedy initialisation …")
    from .initializer import greedy_initialize
    init_sol = greedy_initialize(data, seed=args.seed, n_restarts=3)
    _print_summary(init_sol, data, "INIT")

    # ── Simulated annealing ───────────────────────────────────────────────────
    from .config import SA_T_START, SA_ALPHA, SA_MAX_ITER
    T_start = args.T_start or SA_T_START
    alpha   = args.alpha   or SA_ALPHA
    iters   = args.iters   or SA_MAX_ITER

    logger.info(
        f"Simulated annealing: T_start={T_start:.0f}  alpha={alpha}  iters={iters} …"
    )
    from .optimizer import simulated_annealing
    best_sol, sa_metrics = simulated_annealing(
        data, init_sol,
        seed=args.seed, T_start=T_start, alpha=alpha, max_iter=iters,
    )
    _print_summary(best_sol, data, "SA")

    # Save SA metrics
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    sa_metrics_path = out_path / "metrics.json"
    with open(sa_metrics_path, "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in sa_metrics.items()},
            f, indent=2,
        )

    # ── Visualise SA result ───────────────────────────────────────────────────
    from .visualize import visualize
    visualize(
        data, best_sol,
        out_dir=args.out,
        show=args.show,
        title=(
            f"Simulated Annealing  |  "
            f"cost={sa_metrics['total']:,.0f}  "
            f"violations={sa_metrics['hard_violations']}  "
            f"placed={sa_metrics['cubes_placed']}/{len(cubes)}"
        ),
        draw_flows=not args.no_flows,
        filename="layout_sa.png",
    )

    # ── Optional CP-SAT baseline ──────────────────────────────────────────────
    if args.cpsat:
        from .config import CPSAT_GRID_STEP, CPSAT_TIME_LIMIT
        grid_step  = args.cpsat_grid or CPSAT_GRID_STEP
        time_limit = args.cpsat_time or CPSAT_TIME_LIMIT

        logger.info(
            f"CP-SAT baseline: grid_step={grid_step:.0f} mm  "
            f"time_limit={time_limit:.0f} s …"
        )
        from .cp_sat_baseline import run_cpsat
        result = run_cpsat(data, grid_step=grid_step, time_limit=time_limit, seed=args.seed)

        if result:
            cpsat_sol, cpsat_metrics = result
            _print_summary(cpsat_sol, data, "CP-SAT")

            visualize(
                data, cpsat_sol,
                out_dir=args.out,
                show=args.show,
                title=(
                    f"CP-SAT Baseline  |  "
                    f"status={cpsat_metrics['status']}  "
                    f"placed={cpsat_metrics['cubes_placed']}/{len(cubes)}  "
                    f"t={cpsat_metrics['wall_time_s']:.1f}s"
                ),
                draw_flows=not args.no_flows,
                filename="layout_cpsat.png",
            )

            cpsat_metrics_path = out_path / "metrics_cpsat.json"
            with open(cpsat_metrics_path, "w") as f:
                json.dump(
                    {k: float(v) if isinstance(v, (int, float)) else v
                     for k, v in cpsat_metrics.items()},
                    f, indent=2,
                )

            # ── Comparison summary ────────────────────────────────────────────
            from .objective import total_cost
            sa_cost, _    = total_cost(data, best_sol)
            cpsat_cost, _ = total_cost(data, cpsat_sol)
            logger.info("── Comparison ───────────────────────────")
            logger.info(f"  SA     cost : {sa_cost:,.0f}")
            logger.info(f"  CP-SAT cost : {cpsat_cost:,.0f}")
            logger.info(f"  Gap         : {abs(sa_cost - cpsat_cost) / max(cpsat_cost, 1):.1%}")
        else:
            logger.warning("CP-SAT returned no solution.")

    logger.info("Done.  Outputs written to: " + str(out_path.resolve()))


if __name__ == "__main__":
    main()
