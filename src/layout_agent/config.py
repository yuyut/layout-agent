"""Global hyper-parameters and objective weights."""

# ── Objective weights ────────────────────────────────────────────────────────
W_FLOW: float = 1.0       # material-flow cost  (intensity × Euclidean distance)
W_COMPACT: float = 5e-4   # bounding-box area   (mm² → scale down; global spread prevention)
W_CONTACT: float = 20.0   # contact reward: subtracted for each mm of shared edge between touching cubes
W_DEADSPACE: float = 0.0  # dead-space fraction (off by default; expensive to compute)
W_COLOR: float = 0.2      # grouping cost: pairwise distance within same colour OR utility need
HARD_PENALTY: float = 1e9 # per hard-constraint violation

# ── Simulated-annealing parameters ──────────────────────────────────────────
# T_start and alpha are chosen so T decays to ~T_end after SA_MAX_ITER steps.
# With T_start=1e5, alpha=0.9997, N=30_000:
#   T_final = 1e5 × 0.9997^30000 ≈ 12  (fine-grained moves dominate at end)
SA_T_START: float = 1e5
SA_T_END: float = 1.0
SA_ALPHA: float = 0.9997
SA_MAX_ITER: int = 15_000
SA_SEED: int = 42

# ── Initialiser ─────────────────────────────────────────────────────────────
INIT_GRID_STEP: float = 100.0      # mm – spacing of candidate anchor grid
INIT_TRIES_PER_CUBE: int = 200     # max total anchor×rot attempts per cube
INIT_FEASIBLE_PER_CUBE: int = 30   # stop scoring once this many feasible placements evaluated
INIT_COMPACT_WEIGHT: float = 0.3   # weight of centroid-distance proxy in _partial_cost

# ── Repair ──────────────────────────────────────────────────────────────────
REPAIR_MAX_TRIES: int = 30
REPAIR_NUDGE: float = 8.0   # mm – nudge distance per repair step

# ── CP-SAT baseline ──────────────────────────────────────────────────────────
CPSAT_GRID_STEP: float = 150.0  # coarser grid for discretisation (mm)
CPSAT_TIME_LIMIT: float = 60.0  # solver wall-time limit (seconds)
