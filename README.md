# PyPDMS

A high-performance Python solver for the **Critical Node Problem (CNP)**, using
**IRMS** (*Iterative Ruin and Memetic Search*) — a population-based dual memetic
metaheuristic. PyPDMS combines a fast C++ core (via pybind11) with a clean
Python API.

The Critical Node Problem asks: given a graph and a budget `k`, which `k`
vertices should be removed to minimize the residual pairwise connectivity (the
sum of `|C|*(|C|-1)/2` over all remaining connected components)?

PyPDMS solves two variants through the single entry point `Model.solve`:

- **CNP** (`problem="CNP"`, default): given a budget `k`, minimize the residual
  pairwise connectivity.
- **DCNP** (`problem="DCNP"`): the *distance-based* CNP — given a budget `k` and
  a distance `D`, minimize the number of unordered node pairs whose shortest-path
  distance is at most `D` in the residual graph.

## Installation

PyPDMS is built from source with Meson + Ninja + pybind11.

```bash
pip install -e . --no-build-isolation
```

After editing the C++ code, to recompile just the native extension:

```bash
python buildtools/build_extensions.py --build_type release
```

Requirements:
- Python ≥ 3.9
- A C++20-capable compiler (clang ≥ 17, gcc ≥ 11, or MSVC 2022)
- `meson`, `ninja`, `pybind11` (pulled in automatically by the `build`
  dependency group)

## Quick start

```python
from pypdms import Model, MaxIterations

model = Model()
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]
for u, v in edges:
    model.add_edge(u, v)

result = model.solve(
    budget=3,
    stopping_criterion=MaxIterations(50),
    seed=42,
)

print(f"Best objective: {result.best_obj_value}")
print(f"Removed nodes:  {sorted(result.best_solution)}")
print(f"Runtime:        {result.runtime:.2f}s")
```

### Reading a graph from a file

```python
import pypdms
from pypdms import Model, MaxRuntime

problem = pypdms.read("path/to/graph.adj")  # adjacency-list format
model = Model.from_data(problem)
result = model.solve(budget=100, stopping_criterion=MaxRuntime(60), seed=1)
```

`pypdms.read()` also auto-detects DIMACS edge-list files (a `p edge n m` line
plus several `e u v` lines), the format used by the bundled `Instances/DCNP`
data.

## Distance-based CNP (DCNP)

DCNP minimizes the number of node pairs still within `D` hops after removing
exactly `budget` nodes.

```python
import pypdms
from pypdms import Model, MaxRuntime

problem = pypdms.read("Instances/DCNP/R1/karate.txt")
model = Model.from_data(problem)

result = model.solve(
    problem="DCNP",
    budget=3,
    distance=2,
    stopping_criterion=MaxRuntime(10),
    seed=1,
)

print(f"Best D-hop pairs: {result.best_obj_value}")
print(f"Removed nodes:    {sorted(result.best_solution)}")
```

DCNP uses the **exact same dual-population memetic search** as CNP (feasible +
infeasible populations, RSC crossover, CHNS local search, periodic exchange).
The difference is cost: DCNP's objective rebuilds the K-hop tree on every CHNS
step, so a single local search is far more expensive than in CNP (tens of
seconds on a 300-node instance, versus milliseconds for CNP). Because of this,
DCNP uses a **different set of default hyperparameters**, applied automatically:

| Knob | CNP default | DCNP default | Why |
|------|-------------|--------------|-----|
| `population_size` | 6 | **4** | Initialization cost = `population x 2 x one CHNS`; a smaller population is needed to finish initialization and turn over within a time budget. |
| `transfer_interval` | 50 | **5** | Only a few dozen generations run in total; too large an interval means the exchange never fires. |
| CHNS `random_idle_product` | 2000 | **100** | Caps the per-run idle-step budget; at `D=2` this speeds up a single CHNS run about 3x with no quality loss. |
| CHNS `random_min_idle_steps` | 40 | **20** | A binding floor for large `D`, where each objective evaluation is expensive. |
| CHNS `random_max_idle_steps` | 1000 | **80** | Likewise a ceiling on the idle-step budget. |
| CHNS `random_batch_max` | 50 | **15** | A smaller ruin batch size per step. |

The `population_size` / `transfer_interval` substitution applies **only when you
leave them at the library defaults**; passing an explicit value (or any `chns_*`
field on `SolverParams`) overrides it. These defaults were tuned on 100–500 node
instances (USAir97, Circuit, Ecoli) with a few-minute budget. With the old CNP
defaults, DCNP runs **0** generations in 180 seconds on USAir97 (`k=30, D=3`) —
the population never escapes initialization; with the DCNP defaults it evolves
normally.

> **A note on large `D`.** When `D >= 3`, cost is dominated by the BFS
> maintenance of the K-hop tree. Profiling showed the bottleneck is neither the
> objective sum (already O(n)) nor the greedy selection scan, but the `bfsKTree`
> BFS hot path itself. It has been given a **precise data-structure
> optimization** (flattening the adjacency list into CSR, and replacing the
> per-inner-edge hash lookup with a `vector<uint8_t>` flag array), which speeds
> up a single CHNS run about **2.7–3.6x** while keeping the objective value
> byte-for-byte identical (USAir97 `k=30, D=3`: ~29s → ~11s; `k=20, D=2`: ~11s →
> ~3s). This lets medium `D >= 3` instances evolve normally within a few-minute
> budget.
>
> A further *fully incremental* objective (updating only the affected pairs on a
> removal, without rerunning BFS at all) is possible in theory, but because
> removing a node lengthens the shortest path of many pairs, the cost of a
> precise incremental update is comparable to the current partial BFS, so the
> payoff is limited. For `D >= 3` on larger instances, you can still push the
> per-run cost down further with a smaller `population_size=3` and
> `chns_random_min_idle_steps=10`.

For further tuning, the CHNS budget is exposed through `SolverParams`:

```python
from pypdms import Model, MaxRuntime, SolverParams

# e.g. make DCNP lighter for a large-D instance
params = SolverParams(
    population_size=3,
    transfer_interval=5,
    chns_random_idle_product=60,
    chns_random_min_idle_steps=10,
    chns_random_max_idle_steps=60,
    chns_theta=0.3,
)
result = model.solve(problem="DCNP", budget=30, distance=3,
                     stopping_criterion=MaxRuntime(120), seed=1, params=params)
```

### Directions for speeding up DCNP further

The CSR + flag-array hot-path optimization is already in place (zero-risk,
exact). If you need more speed, the following directions — ranked by
cost/benefit — are not yet implemented, for reference:

1. **Candidate list / neighborhood restriction** (the next best cost/benefit
   choice). `findBestNodeToRemove` currently does one precise "trial removal"
   evaluation per active node; you could pre-filter the top-M candidates with a
   cheap proxy (node degree, `treeSize_`, or precomputed betweenness centrality)
   and evaluate only those precisely. This drops O(n) precise evaluations to
   O(M) (M << n). Profiling shows this step is not the current bottleneck, so
   the speedup is limited and it introduces *small quality variance* (no longer
   exactly the same solution).
2. **Parallelization**. The per-candidate evaluations in
   `findBestNodeToRemove` are independent and could be parallelized with OpenMP.
   But the current evaluation **mutates** shared graph state via
   `removeNode`/`addNode`, so it would first require a *non-mutating* trial
   incremental evaluator (computing only the objective delta without touching
   `intree_`/`treeSize_`), which is a larger refactor.
3. **More aggressive problem-specific defaults**. For `D >= 3` on larger
   instances, use a smaller `population_size=3` together with a lower
   `chns_random_min_idle_steps` (e.g. 10) to push the per-run CHNS cost down
   further in exchange for more generations. This is purely a parameter change,
   no code needed.
4. **Fully incremental objective** (questionable payoff). Update only the
   affected pairs on a removal without rerunning BFS at all. Because removing a
   node lengthens the shortest path of many pairs, the cost of a precise
   incremental update is comparable to the existing partial BFS, so the expected
   payoff is limited — unless combined with an approximation (tolerating a small
   error in the objective value).

## Stopping criteria

PyPDMS ships with three stopping criteria:

| Class | Description |
|-------|-------------|
| `MaxIterations(n)` | Stop after `n` solver iterations |
| `MaxRuntime(s)`    | Stop after `s` seconds of wall-clock time |
| `NoImprovement(n)` | Stop when the best objective has not improved for `n` consecutive iterations |

A stopping criterion can be any callable matching `Callable[[float], bool]`, so
you can define your own (e.g. an objective threshold, or an external signal).

## Solver parameters

Pass a `SolverParams` to tune the search:

```python
from pypdms import Model, MaxIterations, SolverParams

params = SolverParams(
    population_size=10,      # size of each (feasible/infeasible) population
    offspring_count=2,       # offspring generated per population per generation
    transfer_interval=50,    # generations between feasible<->infeasible exchanges
    partial_ratio=0.95,      # infeasible budget = floor(budget * partial_ratio)
    beta=0.9,                # probability RSC crossover keeps shared nodes
    search="CHNS",           # local-search strategy
)

result = model.solve(
    budget=20,
    stopping_criterion=MaxIterations(500),
    params=params,
    seed=42,
)
```

`SolverParams` also provides several `chns_*` fields to override the CHNS
local-search budget (`chns_random_idle_product`, `chns_random_min_idle_steps`,
`chns_random_max_idle_steps`, `chns_random_batch_max`, `chns_theta`,
`chns_max_idle_steps`). They default to `None` (use the native defaults, tuned
for CNP); they matter most for **DCNP** — see the DCNP section above.

### Search strategies

`params.search` accepts:
- `"CHNS"` — randomized batch size and idle steps (default, adaptive)
- `"CHNS-ADAPT"` — idle-step-based adaptive batch growth
- `"CHNS<N>"` — fixed batch size N (e.g. `"CHNS5"`)

## Result

`Model.solve` returns a `Result` containing:
- `best_solution: set[int]`
- `best_obj_value: int`
- `num_iterations: int`
- `runtime: float`
- `best_found_at_time: float`
- `stats: list[dict] | None` — per-iteration trace (on by default)
- `feasible_population: list[tuple[set[int], int]]` — the final population
- `feasible_population_overlap_ratios: list[list[float]]`

## Batch benchmarking

`benchmark.py` runs the solver repeatedly on the same instance with different
seeds and reports the best / average objective, the average time to the best
solution, and the average total runtime. Both problem variants are supported:

```bash
# CNP: Hamilton3000a, budget 300, 10 runs of 30 seconds each
python benchmark.py Instances/CNP/realworld/Hamilton3000a.txt \
    --budget 300 --runs 10 --max-runtime 30

# DCNP: USAir97, k=20, D=2, 5 runs of 120 seconds each (DCNP tuned defaults applied)
python benchmark.py Instances/DCNP/R1/USAir97.txt \
    --problem DCNP --budget 20 --distance 2 --runs 5 --max-runtime 120
```

`--problem` selects the variant (`CNP` / `DCNP`); DCNP requires `--distance`.
The solver knobs (`--population-size`, `--beta`, the `--chns-*` flags, etc.) are
all optional and fall back to the problem-specific defaults when omitted.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]" --no-build-isolation

# Run the tests
pytest tests/

# Recompile the C++ after changes
python buildtools/build_extensions.py --build_type release
```

> **macOS note.** On Apple Silicon, after recompiling and copying the native
> extension (`_pypdms.*.so`), `import` may be killed outright by AMFI with a
> SIGKILL (exit code 137), because the copied Mach-O's signature is invalidated.
> Re-doing an ad-hoc signature fixes it:
> `codesign -f -s - pypdms/_pypdms.cpython-*-darwin.so`.

## License

MIT — see [`LICENSE`](LICENSE).
