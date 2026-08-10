<div align="center">

# PyPDMS

**A high-performance solver for Critical Node Problems, with a C++ core and a clean Python API.**

[![PyPI](https://img.shields.io/pypi/v/pypdms.svg)](https://pypi.org/project/pypdms/)
[![Python](https://img.shields.io/pypi/pyversions/pypdms.svg)](https://pypi.org/project/pypdms/)
[![CI](https://github.com/xuebo100/PCMS/actions/workflows/ci.yml/badge.svg)](https://github.com/xuebo100/PCMS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

PyPDMS finds the set of vertices whose removal most fragments a graph, using
**IRMS** (*Iterative Ruin and Memetic Search*) — a population-based memetic
metaheuristic implemented in C++ and exposed through pybind11.

Given a graph and a budget `k`, the **Critical Node Problem (CNP)** asks which
`k` vertices to remove to minimize the residual pairwise connectivity — the sum
of `|C|·(|C|−1)/2` over every remaining connected component `C`. The
**distance-based variant (DCNP)** instead minimizes the number of node pairs
that stay within `D` hops of each other.

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
```

Load a graph from a file instead of building it by hand:

```python
import pypdms

model = pypdms.Model.from_data(pypdms.read("path/to/graph.adj"))
```

`pypdms.read()` handles adjacency-list files and auto-detects DIMACS edge-list
files (a `p edge n m` line followed by `e u v` lines).

## Distance-based CNP (DCNP)

Select DCNP with `problem="DCNP"` and a `distance`:

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

DCNP runs the same dual-population search as CNP, but its objective is far more
expensive per step, so PyPDMS applies a lighter set of defaults automatically —
see [Tuning DCNP](#tuning-dcnp).

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
| `display` | `True` | Log progress to the package logger. |
| `collect_stats` | `True` | Record a per-iteration trace in `Result.stats`. |

### Stopping criteria

| Criterion | Stops when |
|-----------|------------|
| `MaxIterations(n)` | `n` solver iterations have run |
| `MaxRuntime(s)` | `s` seconds of wall-clock time have elapsed |
| `NoImprovement(n)` | the best objective hasn't improved for `n` iterations |

Any `Callable[[float], bool]` works, so you can supply your own.

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

`search` accepts `"CHNS"` (the default), `"CHNS-ADAPT"`, or `"CHNS<N>"` for a
fixed batch size (e.g. `"CHNS5"`). The `chns_*` fields override the CHNS
local-search budget and matter most for DCNP.

### `Result`

`Model.solve` returns a `Result` with `best_solution`, `best_obj_value`,
`num_iterations`, `runtime`, `best_found_at_time`, an optional per-iteration
`stats` list, and the final `feasible_population`.

## Tuning DCNP

DCNP's per-step cost is high, so it ships with a different set of defaults,
applied automatically **only when you leave these knobs at the library
defaults**:

| Knob | CNP default | DCNP default |
|------|-------------|--------------|
| `population_size` | 6 | **4** |
| `transfer_interval` | 50 | **5** |
| `chns_random_idle_product` | 2000 | **100** |
| `chns_random_min_idle_steps` | 40 | **20** |
| `chns_random_max_idle_steps` | 1000 | **80** |
| `chns_random_batch_max` | 50 | **15** |

Pass an explicit value (or any `chns_*` field) to override. With the old CNP
defaults, DCNP can fail to complete even a single generation within a time
budget on medium instances.

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
