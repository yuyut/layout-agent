# Facility Layout Optimisation

Placing 16 rectangular machines inside an irregular 2D floor plan. The core challenge is that the search space is continuous (any x, y position), the boundary is non-convex, and the objective mixes several concerns:  transport distances, grouping, and packing, that don't naturally reduce to one thing.

---

## 1. Problem Understanding

Each cube gets a bottom-left position `(x, y)` and a rotation flag — 0 for original orientation, 1 for 90°. That's 48 variables total (16 × 3), mixed continuous and discrete.

The hard constraints are straightforward: every cube must fit fully inside the outline, no two cubes can overlap (touching is fine), and all 16 need to be placed. What's less obvious is the objective.

The assignment mentions minimising flow distance, grouping by type/utility, and minimising dead space. I treated these as a single weighted cost:

```
total_cost =
    W_FLOW    × Σ  intensity × dist(centre_i, centre_j)
  + W_COMPACT × bounding_box_area
  + W_COLOR   × Σ  pairwise_distance_within_group
  - W_CONTACT × Σ  shared_edge_length(cube_i, cube_j)
```

Flow cost is the dominant term (~85% of the weighted total) because that's the clearest operational signal. Grouping pulls cubes of the same colour or utility need together, a cube can belong to multiple groups simultaneously, so a colored water-needing cube gets pulled toward other colored cubes and toward other water cubes independently. 
Compactness prevents the layout from spreading to opposite corners but is kept very small (W_COMPACT = 5e-4) because bounding-box area is in mm² while flow cost is in mm. The contact reward subtracts a bonus for every millimetre of shared edge between touching cubes, which is a cheaper proxy for packing density than computing actual area coverage.

A few things I'd want to clarify in a real project: 

are there fixed water/electricity connection points on the walls? 
Are there aisles that must stay clear? 
Are any machines forbidden from being adjacent? 

None of this is specified, so I treated utility needs as soft signals only and set clearance to zero.

---

## 2. Solution Approach

I went with simulated annealing as the main solver. The reason is practical: the feasible region is non-convex (irregular polygon), coordinates are continuous, and exact methods like CP-SAT or MILP require either discretisation or linearisation of the quadratic flow terms, both of which introduce error and scale poorly. 


The pipeline is:

```
DXF + Excel
    │
    ▼
Greedy initialiser  ──  always produces a starting layout based on constrains
    │
    ▼
SA, run across multiple seeds  ──  keep the top 3 results
    │
    ├──  CP-SAT baseline  ──  independent reference point
    │
    ▼
Best solution selected and visualised
```

The greedy initialiser sorts cubes by a priority score (area + flow degree + utility need, each normalised to [0,1]) and places them one by one using a candidate anchor grid. It always produces a valid solution with zero violations, which matters because SA quality is sensitive to starting point. Running it with small Gaussian noise on the priority scores gives different orderings on restarts without losing the heuristic structure entirely.

Running SA from multiple seeds is a lightweight way to get diversity. Each run is independent and takes ~4s, so running 3–5 in parallel is cheap. I keep the top 3 results for comparison rather than just the best, because the second-best often looks structurally different and is worth seeing.

The CP-SAT baseline is there to bound SA quality. It rasterises the floor plan at 150 mm resolution, creates binary variables for each (cube, anchor, rotation) triple, and uses OR-Tools' `no_overlap_2d` constraint. It only optimises flow cost (grouping and contact aren't modelled) and produces coarser placements, but it's independently derived and provides a useful sanity check.

### Move types

SA uses five move types:

- `translate` — nudge one cube by a random offset
- `rotate` — flip one cube's orientation
- `flow_pull` — move a cube toward its highest-flow neighbour
- `swap` — exchange two cubes' positions
- `cluster_shift` — shift an entire colour/utility group together

The broader structural moves (`flow_pull`, `cluster_shift`) are weighted higher at high temperature; `translate` and `rotate` dominate as the run cools.

---

## 3. Objective

I chose weighted scalarisation over Pareto methods because it's simpler to tune interactively and SA only needs a scalar cost to compare against. The downside is that the weights implicitly define the trade-off, if a user cares more about grouping than flow, they need to change a number rather than pick a point on a curve.

Each raw value is just the output of its function before multiplying by the weight:

- Flow — sum of intensity × distance for each flow edge (mm)
- Grouping — sum of pairwise distances between same-group cubes (mm)
- Contact — sum of shared edge lengths between touching cube pairs (mm)
- Compactness — bounding box area of all placed cubes (mm²)


---

## 4. Constraint Handling

**Containment.** The polygon is loaded from DXF and stored as a vertex list. During SA I use a ray-cast check rather than Shapely, because Shapely's `contains()` has significant call overhead. For single-cube moves, only the moved cube is rechecked.

**Non-overlap.** Axis-aligned bounding box arithmetic. For single-cube moves, I check the moved cube against all others (O(n)). Swap and cluster-shift check all pairs via a full violation count.

**Rotation.** Width and height are swapped when rotation is true. The containment and overlap checks use the post-rotation dimensions.

**Utility constraints.** Treated as soft constrains, water and electricity needs enter the objective via grouping cost. There are no physical connection points in the floorplan to enforce hard constraints against.

**SA feasibility.** The greedy initialiser always outputs a feasible solution. During SA, infeasible candidate moves are rejected immediately, the hard penalty term exists as a debugging safety net.

---

## 5. Technical Architecture

```
src/layout_agent/
  config.py           – weights and SA schedule (can edit to tune)
  models.py           – Cube, Placement, Solution, FlowEdge, ProblemData
  io_dxf.py           – DXF loader → Shapely polygon
  io_excel.py         – Excel loader (Characteristics / Media / Materialflow sheets)
  geometry.py         – ray-cast containment, AABB overlap, is_single_feasible
  initializer.py      – greedy constructive placement with multi-restart
  objective.py        – flow_cost, grouping_cost, contact_reward, total_cost
  moves.py            – translate, rotate, swap, flow_pull, cluster_shift
  repair.py           – snap-inside and nudge-apart (used by initialiser only)
  optimizer.py        – SA main loop, convergence history recording
  cp_sat_baseline.py  – optional CP-SAT solver (ortools)
  visualize.py        – matplotlib figure + metrics.json
  main.py             – CLI
  streamlit_app.py    – interactive UI
```

**Hot-path performance.** 
Shapely is only used once (DXF load + initial polygon prep).
Everything inside the SA loop is pure arithmetic — ray-casting against a vertex array, AABB comparisons. 
- O(n) — feasibility check scales linearly with number of cubes (only the moved cube is rechecked)
- O(n²) — contact reward checks every cube pair
- O(e) — flow cost loops over edges only, e — number of flow edges
- O(g·k²) — Grouping cost computes all pairwise distances within each group, and the number of pairs in a group of size k is k(k-1)/2, which is O(k²).

**Input data.**
`data/outline.dxf` 
`data/cubes.xlsx` — 16 cubes, dimensions assumed in mm

**Running.**

```bash
pip install -e .
python -m layout_agent.main --dxf data/outline.dxf --xlsx data/cubes.xlsx

# with interactive UI
pip install -e ".[ui]"
streamlit run src/layout_agent/streamlit_app.py

# with CP-SAT comparison
pip install -e ".[cpsat]"
python -m layout_agent.main --dxf data/outline.dxf --xlsx data/cubes.xlsx --cpsat
```

Outputs go to `output/layout_sa.png` and `output/layout_sa_metrics.json`.

---

## 6. Trade-offs

**Why SA works here.** The problem has a non-convex feasible region, continuous variables, and a mixed objective. SA doesn't care about any of that, it just needs to evaluate a candidate and decide whether to accept it. For 16 cubes it converges well within a few seconds, which makes multi-seed runs practical.

**Where it struggles.** If the building is nearly full, most random moves will be infeasible and the acceptance rate drops, SA spends most of its time generating and rejecting moves. The current schedule is also intentionally warm (T_final ≈ 1,100 for the Balanced preset) so it never really enters a fine-tuning phase. This is fine for finding a good layout from scratch, but it means SA won't squeeze out the last few mm of improvement.

**The main bottleneck at scale** is the contact reward, which checks all O(n²) cube pairs. For n > 100 this would need a spatial index to be practical. Flow cost and grouping cost are already linear in edges and groups, so they scale reasonably.

**Possible extensions**, 
- A\*-routed distances instead of Euclidean (if there is aisle layout)
- Pareto front instead of weighted sum, to make the flow/grouping trade-off explicit
- An adaptive move scheduler that learns which move types are most productive at each temperature

**Prototype to production.** The main gaps are: fixed utility connection points (currently ignored, would become hard placement constraints), clearance gaps between machines, and forbidden adjacency pairs. All three can be added as additional reject conditions. Scaling to larger floor plans would require the spatial index and probably a parallel multi-population SA rather than sequential multi-seed.


Current approach (sequential multi-seed):

```
seed 1 → SA run → best solution
seed 2 → SA run → best solution   (one after another)
seed 3 → SA run → best solution
```
Parallel approach:
```
seed 1 → SA run ─┐
seed 2 → SA run ─┼─ → pick best
seed 3 → SA run ─┘
```