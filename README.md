<div align="center">

# CNDetector

**A high-performance solver for Critical Node Problems, with a C++ core and a clean Python API.**

[![PyPI](https://img.shields.io/pypi/v/cndetector.svg)](https://pypi.org/project/cndetector/)
[![Python](https://img.shields.io/pypi/pyversions/cndetector.svg)](https://pypi.org/project/cndetector/)
[![CI](https://github.com/xuebo100/CNDetector/actions/workflows/ci.yml/badge.svg)](https://github.com/xuebo100/CNDetector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

CNDetector finds the set of vertices whose removal most fragments a graph, using
a **parallel co-evolutionary memetic search** — two complementary populations
evolved in parallel by an RSC crossover, an LCC-oriented large neighbourhood
search (L2NS) and periodic heterogeneous population cooperation, implemented in
C++ and exposed through pybind11.

Given a graph and a budget `k`, the **Critical Node Problem (CNP)** asks which
`k` vertices to remove to minimize the residual pairwise connectivity — the sum
of `|C|·(|C|−1)/2` over every remaining connected component `C`. The
**distance-based variant (DCNP)** instead minimizes the number of node pairs
that stay within `D` hops of each other.

## Installation

```bash
pip install cndetector
```

<details>
<summary>Build from source</summary>

CNDetector builds from source with Meson + Ninja + pybind11:

```bash
pip install -e . --no-build-isolation
```

Requirements: Python ≥ 3.9 and a C++20-capable compiler (clang ≥ 17, gcc ≥ 11,
or MSVC 2022). `meson`, `ninja` and `pybind11` are pulled in automatically.

</details>

## Quick start — CNP

```python
from cndetector import Model, MaxGenerations

model = Model()
for u, v in [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]:
    model.add_edge(u, v)

result = model.solve(
    budget=3,                            # remove 3 nodes
    stopping_criterion=MaxGenerations(50),
    seed=42,
)

print(f"Best objective: {result.best_obj_value}")
print(f"Removed nodes:  {sorted(result.best_solution)}")
```

Load a graph from a file instead of building it by hand:

```python
import cndetector

model = cndetector.Model.from_data(cndetector.read("path/to/graph.adj"))
```

`cndetector.read()` handles adjacency-list files and auto-detects DIMACS edge-list
files (a `p edge n m` line followed by `e u v` lines).

## Distance-based CNP (DCNP)

Select DCNP with `problem="DCNP"` and a `distance`:

```python
import cndetector
from cndetector import MaxRuntime

model = cndetector.Model.from_data(
    cndetector.read("Instances/DCNP/R2/yeast1.txt")
)

result = model.solve(
    problem="DCNP",
    budget=int(0.05 * len(model.nodes)),
    distance=3,                          # count pairs within 3 hops
    stopping_criterion=MaxRuntime(60),
    seed=1,
)

print(f"Best objective: {result.best_obj_value}")
print(f"Removed nodes:    {sorted(result.best_solution)}")
```

DCNP runs the same dual-population search as CNP. Its objective rebuilds a
b-hop tree on every step, so a single local search is far more expensive, and
CNDetector therefore applies a lighter parameter set automatically — see
[Tuning DCNP](#tuning-dcnp).

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
| `collect_stats` | `True` | Record a per-generation trace in `Result.stats`. |

### Stopping criteria

| Criterion | Stops when |
|-----------|------------|
| `MaxGenerations(n)` | `n` generations have been completed |
| `MaxRuntime(s)` | `s` seconds of wall-clock time have elapsed |
| `NoImprovement(n)` | the best objective hasn't improved for `n` generations |

Any `Callable[[float], bool]` works, so you can supply your own.

### `SolverParams`

```python
from cndetector import SolverParams

params = SolverParams(
    population_size=10,             # theta: size of each population
    thread_count=2,                 # kappa: offspring per population per generation
    interaction_period=20,          # beta: generations between two HPC calls
    relaxation_coefficient=0.05,    # alpha: auxiliary budget = floor(k * (1 - alpha))
    allowable_idle_iterations=1000, # gamma: idle-iteration budget of one L2NS run
    stagnation_threshold=500,       # delta: idle generations before reconstruction
    search="L2NS",                  # local search: "L2NS" or "CHNS"
    crossover="RSC",                # crossover: "RSC" or "DBX"
)
```

The values above are the irace-tuned defaults reported in the paper for CNP.
`population_size`, `interaction_period` and `allowable_idle_iterations` default
to `None`, which selects the tuned value of the problem being solved — see
[Tuning DCNP](#tuning-dcnp); any value you pass is used as given.
`relaxation_coefficient=0` makes the two populations homogeneous.

Each L2NS run draws a destroy size `lambda` from `[1, 50]` and stops after
`min(500, gamma / lambda)` idle iterations. The `l2ns_*` fields override that
schedule directly: `l2ns_min_destroy_size`, `l2ns_max_destroy_size`,
`l2ns_idle_iteration_floor`, `l2ns_idle_iteration_cap` and
`l2ns_impact_selection_rate`. `backbone_rate` is the probability that the RSC
crossover keeps a node of the backbone shared by the two parents.

`search="CHNS"` and `crossover="DBX"` swap in the component-based hybrid
neighborhood search and the double backbone-based crossover, the operators that
L2NS and RSC replace. They are kept so the variants of the paper's ablation
study can be rerun, and support CNP only.

### Parallelism

`thread_count` (`kappa`) is both the number of offspring each population
produces per generation and the number of worker threads they are computed on,
so changing it changes the search itself. How many threads actually run it does
not: for a fixed `kappa`, seed and generation budget the solver returns
identical solutions on any machine, whatever its core count.
`set_max_threads(n)` caps the worker threads below `kappa` (for instance to
share a machine) without affecting the result; `get_max_threads()` reports the
number currently in use.

### `Result`

`Model.solve` returns a `Result` with `best_solution`, `best_obj_value`,
`num_generations`, `runtime`, `best_found_at_time`, an optional per-generation
`stats` list, and the final main population `main_population`.

## Tuning DCNP

Evaluating the DCNP objective rebuilds the b-hop trees on every step, which
makes one local search one to two orders of magnitude more expensive than for
CNP. DCNP therefore uses a lighter parameter set, applied automatically to
**every knob you leave unset**:

| Knob | CNP default | DCNP default |
|------|-------------|--------------|
| `population_size` (theta) | 10 | **4** |
| `interaction_period` (beta) | 20 | **5** |
| `allowable_idle_iterations` (gamma) | 1000 | **100** |
| `l2ns_max_destroy_size` | 50 | **15** |
| `l2ns_idle_iteration_floor` | 1 | **20** |
| `l2ns_idle_iteration_cap` | 500 | **80** |

Any value you pass explicitly is used as given, for both problems.


## License

Released under the MIT License — see [`LICENSE`](LICENSE).
