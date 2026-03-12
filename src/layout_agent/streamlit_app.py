"""
Streamlit visualization and experimentation interface.

Run with:
    streamlit run src/layout_agent/streamlit_app.py

The CLI workflow is unchanged:
    python -m layout_agent.main --dxf data/outline.dxf --xlsx data/cubes.xlsx
"""
from __future__ import annotations

import io
import json
import tempfile

import streamlit as st

st.set_page_config(page_title="Facility Layout Optimization", layout="wide")
st.title("Facility Layout Optimization Prototype")
st.caption(
    "Interactive tool for geometric layout optimization — "
    "placing rectangular modules inside an irregular building outline."
)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_outline(path_or_bytes, name: str):
    from layout_agent.io_dxf import load_outline
    if isinstance(path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
            f.write(path_or_bytes)
            tmp = f.name
        return load_outline(tmp)
    return load_outline(path_or_bytes)


@st.cache_data(show_spinner=False)
def _load_excel(path_or_bytes, name: str):
    from layout_agent.io_excel import load_problem_excel
    if isinstance(path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(path_or_bytes)
            tmp = f.name
        return load_problem_excel(tmp)
    return load_problem_excel(path_or_bytes)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # ── Inputs ────────────────────────────────────────────────────────────────
    with st.expander("Override input files (optional)"):
        st.caption("Default data is already loaded. Upload only if you want to use your own files.")
        dxf_file  = st.file_uploader("DXF building outline", type=["dxf"])
        xlsx_file = st.file_uploader("Excel workbook (cubes/flows)", type=["xlsx", "xls"])

    dxf_src  = dxf_file.read()  if dxf_file  else "data/outline.dxf"
    xlsx_src = xlsx_file.read() if xlsx_file else "data/cubes.xlsx"
    dxf_key  = dxf_file.name   if dxf_file  else "default"
    xlsx_key = xlsx_file.name  if xlsx_file else "default"

    st.divider()

    # ── Mode ──────────────────────────────────────────────────────────────────
    st.header("Solver")
    _method_map = {
        "Optimized":             "Simulated Annealing (recommended)",
        "Quick draft":           "Greedy Only",
        "Exact baseline (slow)": "CP-SAT Baseline",
    }
    method_display = st.selectbox(
        "Mode",
        list(_method_map.keys()),
        help=(
            "**Optimized** — simulated annealing, best quality.  \n"
            "**Quick draft** — greedy construction, instant but rough.  \n"
            "**Exact baseline** — CP-SAT integer solver; slower and coarser grid."
        ),
    )
    solver_choice = _method_map[method_display]

    if solver_choice.startswith("Simulated"):
        quality = st.selectbox(
            "Quality", ["Fast (~5s)", "Balanced (~15s)", "High Quality (~45s)"], index=1
        )
        _presets = {
            "Fast (~5s)":          (1e5, 0.999,  5_000),
            "Balanced (~15s)":     (1e5, 0.9997, 15_000),
            "High Quality (~45s)": (1e5, 0.9999, 30_000),
        }
        T_start, _alpha_preset, max_iter = _presets[quality]
    else:
        T_start, _alpha_preset, max_iter = 1e5, 0.9997, 15_000

    st.divider()

    # ── Optimization focus ────────────────────────────────────────────────────
    st.subheader("Optimization focus")
    _focus_map = {
        "Balanced":           (20, 25, 10, 40),
        "Minimize transport": (60, 10, 10, 20),
        "Compact packing":    (20, 20,  5, 55),
        "Group by type":      (20, 15, 40, 25),
        "Custom":             None,
    }
    focus = st.selectbox(
        "Goal",
        list(_focus_map.keys()),
        help=(
            "**Balanced** — equal weight to all objectives.  \n"
            "**Minimize transport** — pull high-flow modules as close as possible.  \n"
            "**Compact packing** — dense, edge-touching layout.  \n"
            "**Group by type** — cluster modules by colour / utility need.  \n"
            "**Custom** — tune each term manually."
        ),
    )

    if _focus_map[focus] is not None:
        p_flow, p_compact, p_group, p_contact = _focus_map[focus]
    else:
        p_flow    = st.slider("Transport cost (flow)",       0, 100, 20, 1, format="%d%%",
                              help="Pull modules with high material flow closer together.")
        p_compact = st.slider("Layout compactness",          0, 100, 25, 1, format="%d%%",
                              help="Penalise the overall bounding-box area — discourages scattered layouts.")
        p_group   = st.slider("Group similar modules",       0, 100, 10, 1, format="%d%%",
                              help="Cluster modules that share colour or utility needs (water / electricity).")
        p_contact = st.slider("Dense edge-touching packing", 0, 100, 40, 1, format="%d%%",
                              help="Reward modules that physically touch — encourages tight packing.")

    w_flow    = p_flow    / 100 * 5.0
    w_compact = p_compact / 100 * 20.0   # × 1e-4 when passed to cfg
    w_group   = p_group   / 100 * 2.0
    w_contact = p_contact / 100 * 50.0

    run_btn = st.button("Run Layout Optimization", type="primary", use_container_width=True)

    # ── Advanced ──────────────────────────────────────────────────────────────
    with st.expander("Advanced settings"):
        if solver_choice.startswith("Simulated"):
            st.caption("SA cooling schedule")
            alpha = st.slider(
                "Cooling rate α", 0.990, 0.9999, float(_alpha_preset), step=0.0001,
                format="%.4f",
                help=(
                    "Lower α = faster cooling = acceptance rate drops to 0 sooner. "
                    "Try 0.9950 to see the curve reach near 0."
                ),
            )
            _T_final = T_start * alpha ** max_iter
            st.caption(f"T_final ≈ {_T_final:,.0f}  (< 10 = cold fine-tuning phase)")
            st.divider()
            n_seeds = st.slider(
                "Number of seeds", 1, 10, 3,
                help="Runs SA this many times with consecutive seeds. Each run explores a different search path. The top 3 results by cost are shown as tabs.",
            )
        else:
            alpha   = _alpha_preset
            n_seeds = 1

        base_seed  = int(st.number_input("Base seed", value=42, step=1))
        st.divider()
        compare    = st.checkbox("Compare Greedy vs Best (side-by-side)")
        draw_flows = st.checkbox("Show material flows", value=True)


# ── Run solver ────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Loading input files…"):
        try:
            outline = _load_outline(dxf_src, dxf_key)
            cubes, flows = _load_excel(xlsx_src, xlsx_key)
        except Exception as e:
            st.error(f"Failed to load inputs: {e}")
            st.stop()

    from layout_agent.models import ProblemData
    from layout_agent.objective import total_cost as _total_cost
    import layout_agent.config as cfg

    # Apply run-time weights, restore on exit
    orig = (cfg.W_FLOW, cfg.W_COMPACT, cfg.W_COLOR, cfg.W_CONTACT)
    cfg.W_FLOW, cfg.W_COMPACT, cfg.W_COLOR, cfg.W_CONTACT = (
        w_flow, w_compact * 1e-4, w_group, w_contact
    )

    data = ProblemData(outline_polygon=outline, cubes=cubes, flows=flows)

    try:
        from layout_agent.initializer import greedy_initialize

        if solver_choice == "Greedy Only":
            with st.spinner("Running greedy initialisation…"):
                sol = greedy_initialize(data, seed=base_seed, n_restarts=3)
            cost, _ = _total_cost(data, sol)
            st.session_state["top3"]     = [(cost, sol, None)]
            st.session_state["init_sol"] = sol

        elif solver_choice.startswith("Simulated"):
            from layout_agent.optimizer import simulated_annealing
            seeds = list(range(base_seed, base_seed + n_seeds))
            results = []
            init_sol_first = None
            for i, s in enumerate(seeds):
                with st.spinner(f"SA run {i+1}/{n_seeds}  (seed={s})…"):
                    init = greedy_initialize(data, seed=s, n_restarts=3)
                    if i == 0:
                        init_sol_first = init
                    sol, metrics = simulated_annealing(
                        data, init,
                        seed=s, T_start=T_start, alpha=alpha, max_iter=max_iter,
                    )
                    cost, _ = _total_cost(data, sol)
                    results.append((cost, sol, metrics))
            results.sort(key=lambda x: x[0])
            st.session_state["top3"]     = results[:3]
            st.session_state["init_sol"] = init_sol_first

        else:  # CP-SAT
            with st.spinner("Running CP-SAT baseline…"):
                from layout_agent.cp_sat_baseline import run_cpsat
                result = run_cpsat(data, seed=base_seed)
            if result:
                cost, _ = _total_cost(data, result[0])
                st.session_state["top3"] = [(cost, result[0], result[1])]
            else:
                st.error("CP-SAT returned no solution.")
                st.stop()

        st.session_state["solution"]    = st.session_state["top3"][0][1]
        st.session_state["data"]        = data
        st.session_state["compare"]     = compare
        st.session_state["draw_flows"]  = draw_flows
        # Store weights used in this run — config is restored in finally,
        # so the metrics display must read from session_state, not cfg.*
        st.session_state["run_weights"] = (
            cfg.W_FLOW, cfg.W_COMPACT, cfg.W_COLOR, cfg.W_CONTACT
        )

    finally:
        cfg.W_FLOW, cfg.W_COMPACT, cfg.W_COLOR, cfg.W_CONTACT = orig


# ── Helper: render layout to PNG bytes ───────────────────────────────────────

def _render_layout(data, sol, title: str, draw_flows: bool) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as mpe
    import matplotlib.colors as mcolors
    from layout_agent.geometry import cube_center

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal")
    xs, ys = data.outline_polygon.exterior.xy
    ax.fill(xs, ys, alpha=0.07, color="steelblue")
    ax.plot(xs, ys, color="steelblue", linewidth=2.0, label="Outline")

    cubes_map = {c.id: c for c in data.cubes}
    for cid, pl in sol.placements.items():
        cube = cubes_map[cid]
        w, h = cube.dims(pl.rot)
        raw = cube.color.strip().lower() if cube.color else None
        fc  = raw if (raw and mcolors.is_color_like(raw)) else "#aaaaaa"
        ax.add_patch(mpatches.FancyBboxPatch(
            (pl.x, pl.y), w, h, boxstyle="square,pad=0",
            linewidth=1.0, edgecolor="#222", facecolor=fc, alpha=0.78, zorder=3,
        ))
        cx, cy = cube_center(cube, pl)
        suffix = ("W" if cube.needs_water else "") + ("E" if cube.needs_electricity else "")
        ax.text(cx, cy, f"{cid}" + (f"\n{suffix}" if suffix else ""),
                ha="center", va="center", fontsize=6, fontweight="bold", color="black",
                path_effects=[mpe.withStroke(linewidth=1.8, foreground="white")], zorder=5)

    if draw_flows and data.flows:
        max_i = max(e.intensity for e in data.flows)
        for edge in data.flows:
            if edge.src not in sol.placements or edge.dst not in sol.placements:
                continue
            sc = cube_center(cubes_map[edge.src], sol.placements[edge.src])
            dc = cube_center(cubes_map[edge.dst], sol.placements[edge.dst])
            lw = 0.6 + 3.5 * edge.intensity / max_i
            ax.annotate("", xy=dc, xytext=sc,
                        arrowprops=dict(arrowstyle="->", color="crimson",
                                        lw=lw, alpha=0.65, connectionstyle="arc3,rad=0.1"),
                        zorder=4)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── Main panel ────────────────────────────────────────────────────────────────

if "solution" not in st.session_state:
    st.info("Configure the solver in the sidebar and click **Run Layout Optimization**.")
    st.stop()

data        = st.session_state["data"]
sol         = st.session_state["solution"]
top3        = st.session_state.get("top3", [])
init_sol    = st.session_state.get("init_sol")
draw_flows  = st.session_state.get("draw_flows", True)
do_compare  = st.session_state.get("compare", False)
run_weights = st.session_state.get("run_weights")

from layout_agent.objective import total_cost
cost, bd = total_cost(data, sol, include_deadspace=False)

# ── Feasibility badge ─────────────────────────────────────────────────────────

if bd["hard_violations"] == 0:
    st.success(
        f"Feasible — all {bd['cubes_placed']} / {len(data.cubes)} cubes placed, "
        f"0 violations"
    )
else:
    st.error(
        f"{bd['hard_violations']} violation(s) — "
        f"{bd['cubes_placed']} / {len(data.cubes)} cubes placed"
    )

# ── Visualization ─────────────────────────────────────────────────────────────

if do_compare and init_sol:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Greedy initialisation")
        st.image(_render_layout(data, init_sol, "Greedy", draw_flows), use_container_width=True)
    with col2:
        st.subheader("Best optimized")
        st.image(_render_layout(data, sol, "Best", draw_flows), use_container_width=True)
    img_bytes = _render_layout(data, sol, "Best", draw_flows)

elif len(top3) > 1:
    medals     = ["Best", "2nd", "3rd"]
    tab_labels = [f"{medals[i]}  cost={top3[i][0]:,.0f}" for i in range(len(top3))]
    tabs = st.tabs(tab_labels)
    img_bytes = None
    for i, (tab, (c_i, sol_i, _)) in enumerate(zip(tabs, top3)):
        with tab:
            img = _render_layout(data, sol_i, f"Seed run #{i+1}  |  cost={c_i:,.0f}", draw_flows)
            col_img, _ = st.columns([2, 1])
            with col_img:
                st.image(img, use_container_width=True)
            if i == 0:
                img_bytes = img

else:
    img_bytes = _render_layout(data, sol, "Optimized Layout", draw_flows)
    col_img, col_pad = st.columns([2, 1])
    with col_img:
        st.image(img_bytes, use_container_width=True)

# ── Metrics ───────────────────────────────────────────────────────────────────

st.subheader("Objective breakdown  (best solution)")

# Use weights from the run, not the current (restored) config defaults
import layout_agent.config as _cfg
_rw = run_weights or (_cfg.W_FLOW, _cfg.W_COMPACT, _cfg.W_COLOR, _cfg.W_CONTACT)
rw_flow, rw_compact, rw_color, rw_contact = _rw

w_fc = rw_flow    * bd["flow_cost"]
w_gc = rw_color   * bd["grouping_cost"]
w_bb = rw_compact * bd["compactness_mm2"]
w_cr = rw_contact * bd["contact_reward"]
_total_abs = w_fc + w_gc + w_bb + w_cr or 1.0

def _pct(w): return f"{100 * w / _total_abs:.1f}%"

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total cost",      f"{cost:,.0f}")
m2.metric("Flow cost",       f"{bd['flow_cost']:,.0f}",
          delta=_pct(w_fc), delta_color="off")
m3.metric("Grouping cost",   f"{bd['grouping_cost']:,.0f}",
          delta=_pct(w_gc), delta_color="off")
m4.metric("Contact reward",  f"{bd['contact_reward']:,.1f}",
          delta=f"{_pct(w_cr)}", delta_color="off",
          help="Higher = more touching edges = better packing.")
m5.metric("Compactness mm²", f"{bd['compactness_mm2']:,.0f}",
          delta=_pct(w_bb), delta_color="off")
m6.metric("Violations",      bd["hard_violations"])

st.caption(
    f"Placed {bd['cubes_placed']}/{len(data.cubes)} cubes  ·  "
    f"% = weighted contribution out of Σ|weighted terms|"
)

# ── Convergence diagnostics ───────────────────────────────────────────────────

_sa_runs = [(i, c, m) for i, (c, _, m) in enumerate(top3)
            if m and "cost_history" in m]

if _sa_runs:
    with st.expander("Convergence diagnostics", expanded=False):
        import pandas as _pd

        hi    = _sa_runs[0][2]["history_interval"]
        iters = [hi * (k + 1) for k in range(len(_sa_runs[0][2]["cost_history"]))]

        ch1, ch2 = st.columns(2)

        with ch1:
            st.caption("Best cost vs iteration")
            cost_df = _pd.DataFrame(
                {f"Seed {i+1}  (cost={c:,.0f})": m["cost_history"] for i, c, m in _sa_runs},
                index=iters,
            )
            cost_df.index.name = "iteration"
            st.line_chart(cost_df)

        with ch2:
            st.caption("Temperature decay")
            temp_df = _pd.DataFrame(
                {"Temperature": _sa_runs[0][2]["temp_history"]},
                index=iters,
            )
            temp_df.index.name = "iteration"
            st.line_chart(temp_df)

        st.caption(
            "Metropolis acceptance rate per window  "
            "(accepted / feasible moves — near 1.0 early = exploring freely, "
            "near 0.0 late = only accepting improvements)"
        )
        ar_df = _pd.DataFrame(
            {f"Seed {i+1}": m["accept_rate_history"] for i, _, m in _sa_runs},
            index=iters,
        )
        ar_df.index.name = "iteration"
        st.line_chart(ar_df)

# ── Placement table ───────────────────────────────────────────────────────────

st.subheader("Placement table  (best solution)")
import pandas as pd
cubes_map = {c.id: c for c in data.cubes}
rows = []
for cid, pl in sorted(sol.placements.items()):
    cube = cubes_map[cid]
    w, h = cube.dims(pl.rot)
    rows.append({
        "cube_id": cid, "x (mm)": round(pl.x), "y (mm)": round(pl.y),
        "rotation": pl.rot, "width": round(w), "height": round(h),
        "color": cube.color, "needs_water": cube.needs_water,
        "needs_electricity": cube.needs_electricity,
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────

st.subheader("Export")
e1, e2, e3 = st.columns(3)
e1.download_button("Download layout PNG",   img_bytes or b"",
                   file_name="layout.png", mime="image/png")
e2.download_button("Download metrics JSON",
                   json.dumps({k: float(v) if isinstance(v, (int, float)) else v
                                for k, v in bd.items()}, indent=2),
                   file_name="metrics.json", mime="application/json")
e3.download_button("Download placements CSV",
                   df.to_csv(index=False).encode(),
                   file_name="placements.csv", mime="text/csv")
