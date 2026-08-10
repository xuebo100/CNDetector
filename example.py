"""
Usage example for solving Critical Node Problems with PyPDMS.

This example deliberately spells out *every* configurable option so you can see
at a glance what is tunable when running the IRMS (Iterative Ruin and Memetic
Search) algorithm. Unless a comment says otherwise, every value below is the
default.
"""

import time

import pypdms
from pypdms import (
    MaxIterations,  # stop after N generations
    MaxRuntime,  # stop after S seconds of wall-clock time (precise)
    Model,
    NoImprovement,  # stop after N generations without improvement
    SolverParams,
)


def build_graph_manually() -> Model:
    """Build a small graph edge by edge."""
    model = Model()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    for u, v in edges:
        model.add_edge(u, v)
    return model


def main():
    # ------------------------------------------------------------------ #
    # 1. Build the model: manually (see above) or read it from a file.
    # ------------------------------------------------------------------ #
    # model = build_graph_manually()
    model = Model.from_data(
        pypdms.read("Instances/CNP/realworld/condmat.txt")
    )

    # ------------------------------------------------------------------ #
    # 2. Pick a stopping criterion (choose exactly one).
    #    All three are listed; only `stopping_criterion` below is used.
    # ------------------------------------------------------------------ #
    stop_by_iterations = MaxIterations(500)      # stop after 500 generations
    stop_by_runtime = MaxRuntime(3600)             # stop after 3600 seconds
    stop_by_no_improve = NoImprovement(200)      # 200 idle generations -> stop
    # You can also pass any custom Callable[[float], bool].
    _ = (stop_by_iterations, stop_by_no_improve)  # silence "unused" lint warnings
    stopping_criterion = stop_by_runtime

    # ------------------------------------------------------------------ #
    # 3. Tunable solver parameters. Each field is shown with its default.
    #    (Constraints: population_size>=2, offspring_count>=1,
    #     transfer_interval>=1, 0<partial_ratio<1, 0<=beta<=1,
    #     display_interval>0.)
    # ------------------------------------------------------------------ #
    params = SolverParams(
        population_size=6,        # size of each (feasible/infeasible) population
        offspring_count=1,        # offspring generated per population per gen
        transfer_interval=50,     # generations between feasible<->infeasible swaps
        partial_ratio=0.95,       # infeasible budget = floor(budget*partial_ratio)
        beta=0.9,                 # RSC crossover: probability of keeping shared nodes
        display_interval=1.0,     # minimum seconds between progress log lines
        search="CHNS",            # local-search strategy (see options below)
        # CHNS budget overrides; None -> native default (tuned for CNP). These
        # mostly matter for DCNP (see dcnp() below), where each CHNS run is
        # expensive; keep them None for CNP.
        chns_random_idle_product=None,
        chns_random_min_idle_steps=None,
        chns_random_max_idle_steps=None,
        chns_random_batch_max=None,
        chns_theta=None,
    )
    # `search` accepts:
    #   "CHNS"        -> randomized batch size and idle steps (default, adaptive)
    #   "CHNS-ADAPT"  -> idle-step-based adaptive batch growth
    #   "CHNS<N>"     -> fixed batch size N, e.g. "CHNS5"

    # ------------------------------------------------------------------ #
    # 4. Run the solver. Every optional argument of solve() is listed.
    #    population_size / offspring_count / search passed here **override**
    #    the matching fields in `params` (handy for quick experiments); leave
    #    them None to use `params` directly.
    # ------------------------------------------------------------------ #
    result = model.solve(
        budget=2313,                            # required: number of nodes to remove
        stopping_criterion=stopping_criterion, # required: when to stop
        seed=42,                               # random seed (0 is valid)
        params=params,                         # the SolverParams above
        population_size=None,                  # override params.population_size
        offspring_count=None,                  # override params.offspring_count
        search=None,                           # override params.search
        display_interval=None,                 # override params.display_interval
        display=True,                          # log progress to the logger
        collect_stats=True,                    # record the per-generation trace
    )

    # ------------------------------------------------------------------ #
    # 5. Inspect the result. Every field of Result is listed.
    # ------------------------------------------------------------------ #
    print(f"Best objective value : {result.best_obj_value}")
    # print(f"Removed nodes        : {sorted(result.best_solution)}")
    print(f"Iterations           : {result.num_iterations}")
    print(f"Runtime              : {result.runtime:.2f}s")
    print(f"Best found at        : {result.best_found_at_time:.2f}s")
    print(f"Final population size: {len(result.feasible_population)}")
    if result.stats:
        print(f"Trace entries        : {len(result.stats)}")


def dcnp():
    """Solve the distance-based CNP (DCNP) on the small `karate` instance.

    DCNP fixes a removal budget k and a distance threshold D, and minimizes the
    number of unordered node pairs whose shortest-path distance is at most D in
    the residual graph.
    """
    # ------------------------------------------------------------------ #
    # 1. Build the model from a DCNP instance in DIMACS edge-list format.
    #    pypdms.read() auto-detects the `p edge n m` / `e u v` format.
    # ------------------------------------------------------------------ #
    budget = 30
    distance = 3
    seed = 1

    model = Model.from_data(
        pypdms.read("Instances/DCNP/S1/gnm3(9).txt")
    )

    # ------------------------------------------------------------------ #
    # 2. Pick a stopping criterion (choose exactly one).
    # ------------------------------------------------------------------ #
    # stop_by_iterations = MaxIterations(20)
    stop_by_runtime = MaxRuntime(3600)
    # stop_by_no_improve = NoImprovement(10)
    # _ = (stop_by_runtime, stop_by_no_improve)
    stopping_criterion = stop_by_runtime

    # ------------------------------------------------------------------ #
    # 3. Tunable solver parameters. These are the same common parameters used
    #    for CNP; for DCNP they control the population, crossover and CHNS
    #    local-search behavior.
    #
    #    Every field below is set to its DCNP default, so this SolverParams is
    #    equivalent to passing no `params` at all. DCNP overrides three CNP
    #    defaults (population_size 6->4, transfer_interval 50->5, and a lighter
    #    CHNS idle budget) because DCNP's objective rebuilds K-hop trees every
    #    step, making each local search far more expensive; the lighter budget
    #    lets the population actually turn over in time. The population_size /
    #    transfer_interval substitution only applies when you leave them at the
    #    CNP library defaults (6 / 50), so the effective DCNP values are spelled
    #    out here explicitly.
    # ------------------------------------------------------------------ #
    params = SolverParams(
        population_size=4,             # DCNP default (CNP default is 6)
        offspring_count=1,             # offspring generated per generation
        transfer_interval=5,           # DCNP default (CNP default is 50)
        partial_ratio=0.95,            # infeasible budget = floor(budget*ratio)
        beta=0.9,                      # probability of keeping shared nodes
        display_interval=1.0,          # minimum seconds between progress log lines
        search="CHNS",                 # local-search strategy
        # CHNS budget overrides (None -> native default). These values are the
        # DCNP defaults applied automatically; spelled out here for tuning.
        chns_random_idle_product=100,  # scale factor for the per-run idle budget
        chns_random_min_idle_steps=20, # idle-step floor (binds for large D)
        chns_random_max_idle_steps=80,  # idle-step ceiling
        chns_random_batch_max=15,      # max ruin batch size per step
        chns_theta=0.3,                # probability of greedy (vs random) selection
    )

    # ------------------------------------------------------------------ #
    # 4. Run the DCNP solver. `budget` and `distance` are required for DCNP.
    #    population_size / offspring_count / search passed here **override**
    #    the matching fields in `params`; leave them None to use `params`.
    # ------------------------------------------------------------------ #
    start = time.perf_counter()
    result = model.solve(
        problem="DCNP",
        budget=budget,                         # required: number of nodes to remove
        distance=distance,                     # required: D-hop threshold
        stopping_criterion=stopping_criterion, # required: when to stop
        seed=seed,                             # random seed (0 is valid)
        params=params,                         # the SolverParams above
        population_size=None,                  # override params.population_size
        offspring_count=None,                  # override params.offspring_count
        search=None,                           # override params.search
        display_interval=None,                 # override params.display_interval
        display=True,                          # log progress to the logger
        collect_stats=True,                    # record the per-generation trace
    )
    wall = time.perf_counter() - start

    # ------------------------------------------------------------------ #
    # 5. Inspect the result.
    # ------------------------------------------------------------------ #
    print("\n=== DCNP ===")
    print(f"Nodes / budget k / D : {len(model.nodes)} / {budget} / {distance}")
    print(f"Best D-hop pairs     : {result.best_obj_value}")
    print(f"Removed nodes        : {sorted(result.best_solution)}")
    print(f"Iterations           : {result.num_iterations}")
    print(f"Runtime              : {result.runtime:.2f}s")
    print(f"Best found at        : {result.best_found_at_time:.2f}s")
    print(f"Final population size: {len(result.feasible_population)}")
    if result.stats:
        print(f"Trace entries        : {len(result.stats)}")
    print(f"Wall time            : {wall:.2f}s")


if __name__ == "__main__":
    main()
    # dcnp()
