<div align="center">

# PyPDMS

**A high-performance solver for Critical Node Problems, with a C++ core and a clean Python API.**

[![PyPI](https://img.shields.io/pypi/v/pypdms.svg)](https://pypi.org/project/pypdms/)
[![Python](https://img.shields.io/pypi/pyversions/pypdms.svg)](https://pypi.org/project/pypdms/)
[![CI](https://github.com/xuebo100/PCMS/actions/workflows/ci.yml/badge.svg)](https://github.com/xuebo100/PCMS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

PyPDMS finds the set of vertices whose removal most fragments a graph. It solves
the problem with **IRMS** (*Iterative Ruin and Memetic Search*), a
population-based dual memetic metaheuristic implemented in C++ and exposed
through pybind11 — so you get near-native speed behind a few lines of Python.

Given a graph and a budget `k`, the **Critical Node Problem (CNP)** asks which
`k` vertices to remove to minimize the residual pairwise connectivity — the sum
of `|C|·(|C|−1)/2` over every remaining connected component `C`.

## Highlights

- **Two problems, one API.** Solve classic CNP and its distance-based variant
  (DCNP) through a single `Model.solve` call.
- **Fast C++ core.** The search kernel is native code; Python only orchestrates.
- **Batteries included.** Adjacency-list and DIMACS readers, three stopping
  criteria, a rich `Result`, and per-iteration statistics.
- **Tunable, with sane defaults.** Every hyperparameter is exposed via
  `SolverParams`; DCNP automatically applies its own tuned defaults.
- **Typed and tested.** Ships `py.typed` stubs and runs its suite on Linux,
  macOS and Windows for Python 3.9–3.13.

## Installation

```bash
pip install pypdms
```

<details>
<summary>Build from source</summary>

PyPDMS builds from source with Meson + Ninja + pybind11:

```bash
pip install -e . --no-build-isolation
```

Recompile just the native extension after editing the C++:

```bash
python buildtools/build_extensions.py --build_type release
```

Requirements: Python ≥ 3.9 and a C++20-capable compiler (clang ≥ 17, gcc ≥ 11,
or MSVC 2022). `meson`, `ninja` and `pybind11` are pulled in automatically.

</details>

## Quick start — CNP

```python
from pypdms import Model, MaxIterations

model = Model()
for u, v in [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]:
    model.add_edge(u, v)

result = model.solve(
    budget=3,                            # remove 3 nodes
    stopping_criterion=MaxIterations(50),
    seed=42,
)

print(f"Best objective: {result.best_obj_value}")
print(f"Removed nodes:  {sorted(result.best_solution)}")
print(f"Runtime:        {result.runtime:.2f}s")
```

Load a graph from a file instead of building it by hand:

```python
import pypdms

problem = pypdms.read("path/to/graph.adj")   # adjacency-list format
model = pypdms.Model.from_data(problem)
```

`pypdms.read()` also auto-detects DIMACS edge-list files (a `p edge n m` line
followed by `e u v` lines), the format used by the bundled `Instances/DCNP`
data.

## Distance-based CNP (DCNP)

DCNP minimizes the number of node pairs still within `D` hops of each other after
removing exactly `budget` nodes. Select it with `problem="DCNP"` and a
`distance`:

```python
import pypdms
from pypdms import MaxRuntime

model = pypdms.Model.from_data(pypdms.read("Instances/DCNP/R1/karate.txt"))

result = model.solve(
    problem="DCNP",
    budget=3,
    distance=2,                          # count pairs within 2 hops
    stopping_criterion=MaxRuntime(10),
    seed=1,
)

print(f"Best D-hop pairs: {result.best_obj_value}")
print(f"Removed nodes:    {sorted(result.best_solution)}")
```

DCNP runs the *same* dual-population memetic search as CNP, but its objective
rebuilds a K-hop tree on every step, so a single local search is far more
expensive. PyPDMS therefore applies a lighter set of defaults automatically when
you leave the relevant knobs untouched — see
[Tuning DCNP](#tuning-dcnp) below.

## API reference

### `Model.solve(...)`

| Argument | Default | Description |
|----------|---------|-------------|
| `budget` | — | Number of nodes to remove (`1 ≤ budget < \|V\|`). Required. |
| `stopping_criterion` | — | Callable stopping the solver when it returns `True`. Required. |
| `problem` | `"CNP"` | `"CNP"` or `"DCNP"`. |
| `distance` | `None` | DCNP distance threshold `D` (required for DCNP, `D ≥ 1`). |
| `seed` | `0` | RNG seed (`0` is valid). |
| `params` | `SolverParams()` | Tunable solver parameters (see below). |
| `population_size`, `offspring_count`, `search` | `None` | Convenience overrides for the matching `params` fields. |
| `display` | `True` | Log progress to the package logger. |
| `collect_stats` | `True` | Record a per-iteration trace in `Result.stats`. |

### Stopping criteria

| Criterion | Stops when |
|-----------|------------|
| `MaxIterations(n)` | `n` solver iterations have run |
| `MaxRuntime(s)` | `s` seconds of wall-clock time have elapsed |
| `NoImprovement(n)` | the best objective hasn't improved for `n` iterations |

Any `Callable[[float], bool]` works, so you can supply your own (an objective
threshold, an external signal, …).

### `SolverParams`

```python
from pypdms import SolverParams

params = SolverParams(
    population_size=10,   # size of each (feasible/infeasible) population
    offspring_count=2,    # offspring per population per generation
    transfer_interval=50, # generations between feasible<->infeasible exchanges
    partial_ratio=0.95,   # infeasible budget = floor(budget * partial_ratio)
    beta=0.9,             # probability RSC crossover keeps shared nodes
    search="CHNS",        # local-search strategy
)
```

`search` accepts `"CHNS"` (randomized, adaptive — the default), `"CHNS-ADAPT"`
(idle-step-based adaptive batch growth), or `"CHNS<N>"` for a fixed batch size
(e.g. `"CHNS5"`). The `chns_*` fields override the CHNS local-search budget and
matter most for DCNP — see below.

### `Result`

`Model.solve` returns a `Result` with:

| Field | Type | Meaning |
|-------|------|---------|
| `best_solution` | `set[int]` | the removed nodes |
| `best_obj_value` | `int` | the best objective found |
| `num_iterations` | `int` | iterations run |
| `runtime` | `float` | total wall-clock seconds |
| `best_found_at_time` | `float` | when the best solution was found |
| `stats` | `list[dict] \| None` | per-iteration trace (on by default) |
| `feasible_population` | `list[tuple[set[int], int]]` | the final population |
| `feasible_population_overlap_ratios` | `list[list[float]]` | pairwise overlap |

## Tuning DCNP

DCNP's per-CHNS cost is high, so it ships with a different set of defaults,
applied automatically **only when you leave these knobs at the library
defaults**:

| Knob | CNP default | DCNP default | Why |
|------|-------------|--------------|-----|
| `population_size` | 6 | **4** | Smaller populations finish initialization and turn over within a time budget. |
| `transfer_interval` | 50 | **5** | Only a few dozen generations run; a large interval means the exchange never fires. |
| `chns_random_idle_product` | 2000 | **100** | Caps the per-run idle budget; ~3× faster CHNS at `D=2` with no quality loss. |
| `chns_random_min_idle_steps` | 40 | **20** | A binding floor for large `D`. |
| `chns_random_max_idle_steps` | 1000 | **80** | A ceiling on the idle budget. |
| `chns_random_batch_max` | 50 | **15** | A smaller ruin batch per step. |

Pass an explicit value (or any `chns_*` field) to override. Tuned on 100–500
node instances (USAir97, Circuit, Ecoli). With the old CNP defaults, DCNP runs
**0** generations in 180 s on USAir97 (`k=30, D=3`); with these it evolves
normally.

```python
from pypdms import MaxRuntime, SolverParams

# make DCNP even lighter for a large-D instance
params = SolverParams(
    population_size=3,
    chns_random_idle_product=60,
    chns_random_min_idle_steps=10,
    chns_random_max_idle_steps=60,
    chns_theta=0.3,
)
result = model.solve(problem="DCNP", budget=30, distance=3,
                     stopping_criterion=MaxRuntime(120), seed=1, params=params)
```

<details>
<summary>Performance notes &amp; roadmap (large <code>D</code>)</summary>

For `D ≥ 3`, cost is dominated by BFS maintenance of the K-hop tree — the
`bfsKTree` hot path, not the objective sum (already O(n)) or the greedy scan.
That path has a precise data-structure optimization (adjacency flattened to CSR,
per-inner-edge hash lookups replaced by a `vector<uint8_t>` flag array) that
speeds up a single CHNS run **~2.7–3.6×** with byte-identical objectives
(USAir97 `k=30, D=3`: ~29 s → ~11 s; `k=20, D=2`: ~11 s → ~3 s).

Not yet implemented, ranked by cost/benefit:

1. **Candidate list / neighborhood restriction.** Pre-filter the top-M removal
   candidates with a cheap proxy (degree, `treeSize_`, betweenness) and evaluate
   only those precisely, cutting O(n) precise evaluations to O(M). Introduces
   small quality variance; not the current bottleneck, so limited gain.
2. **Parallelization.** Per-candidate evaluations are independent, but the
   current evaluator mutates shared graph state via `removeNode`/`addNode`; a
   non-mutating trial evaluator (objective delta only) would be needed first.
3. **More aggressive defaults.** For large instances at `D ≥ 3`, a smaller
   `population_size=3` and lower `chns_random_min_idle_steps` trade per-run cost
   for more generations. Pure parameter change.
4. **Fully incremental objective.** Update only affected pairs on a removal
   without rerunning BFS. Because a removal lengthens many pairs' shortest paths,
   an exact incremental update costs about the same as the partial BFS — limited
   payoff unless combined with an approximation.

</details>

## Development

```bash
pip install -e ".[dev]" --no-build-isolation   # dev dependencies
pytest tests/                                    # run the tests
python buildtools/build_extensions.py --build_type release   # recompile C++
```

> **macOS note.** On Apple Silicon, after recompiling and copying the native
> extension (`_pypdms.*.so`), `import` may be killed by AMFI with SIGKILL (exit
> code 137) because the copied Mach-O's signature is invalidated. Re-sign it
> ad-hoc to fix it:
> `codesign -f -s - pypdms/_pypdms.cpython-*-darwin.so`.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
