"""
Repeated-run benchmark for PyPDMS.

Runs the IRMS solver ``n`` times with different seeds on the same instance and
reports a summary:

  * the best objective value found,
  * the average objective value,
  * the average time at which each run found its best solution
    (``Result.best_found_at_time``), and
  * the average total runtime.

Every knob is passed on the command line. Examples
-------------------------------------------------
    # 10 runs, 30 seconds each, Hamilton3000a, budget 300
    python benchmark.py Instances/CNP/realworld/Hamilton3000a.txt \
        --budget 300 --runs 10 --max-runtime 30

    # 5 iteration-bounded runs with custom solver knobs
    python benchmark.py Instances/CNP/realworld/Bovine.txt \
        --budget 3 --runs 5 --max-iterations 200 \
        --population-size 12 --offspring-count 2 --search CHNS5

    # DCNP: minimize the number of node pairs within distance D=2 after removing
    # k=20 nodes, 5 runs of 120 seconds each (DCNP tuned defaults applied)
    python benchmark.py Instances/DCNP/R1/USAir97.txt \
        --problem DCNP --budget 20 --distance 2 --runs 5 --max-runtime 120

    # DCNP: use a lighter CHNS budget for a large-D instance
    python benchmark.py Instances/DCNP/R1/USAir97.txt \
        --problem DCNP --budget 30 --distance 3 --runs 3 --max-runtime 120 \
        --population-size 3 --chns-random-idle-product 60 \
        --chns-random-min-idle-steps 10
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import pypdms
from pypdms import (
    MaxIterations,
    MaxRuntime,
    Model,
    NoImprovement,
    SolverParams,
)
from pypdms.stop import StoppingCriterion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyPDMS solver n times and report summary statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "instance",
        help="Path to the graph file in adjacency-list format.",
    )
    parser.add_argument(
        "--problem", type=str, default="CNP", choices=["CNP", "DCNP"],
        help="Problem variant to solve.",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="Number of nodes to remove (k); required, must be < |V|.",
    )
    parser.add_argument(
        "--distance", type=int, default=None,
        help="DCNP distance threshold D (required for --problem DCNP, D >= 1).",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=10,
        help="Number of consecutive runs.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed; run i uses seed + i.",
    )

    # Stopping criteria (mutually exclusive). Defaults to 30 seconds of runtime.
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument(
        "--max-runtime", type=float, metavar="SECONDS",
        help="Stop each run after this many seconds (wall clock).",
    )
    stop.add_argument(
        "--max-iterations", type=int, metavar="N",
        help="Stop each run after this many generations.",
    )
    stop.add_argument(
        "--no-improvement", type=int, metavar="N",
        help="Stop each run after N generations without improvement.",
    )

    # Optional solver knobs (left as None -> use the SolverParams defaults).
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--offspring-count", type=int, default=None)
    parser.add_argument(
        "--search", type=str, default=None,
        help='Local-search strategy: "CHNS", "CHNS-ADAPT" or "CHNS<N>".',
    )
    parser.add_argument(
        "--transfer-interval", type=int, default=None,
        help="Generations between feasible<->infeasible exchanges.",
    )
    parser.add_argument("--partial-ratio", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)

    # CHNS local-search budget overrides (None -> use the problem-specific
    # default). These are the main tuning knobs for DCNP -- its objective
    # rebuilds K-hop trees every step, and DCNP already applies lighter defaults.
    parser.add_argument("--chns-theta", type=float, default=None)
    parser.add_argument("--chns-random-batch-max", type=int, default=None)
    parser.add_argument("--chns-random-idle-product", type=int, default=None)
    parser.add_argument("--chns-random-min-idle-steps", type=int, default=None)
    parser.add_argument("--chns-random-max-idle-steps", type=int, default=None)

    parser.add_argument(
        "--show-solver-log", action="store_true",
        help="Show the solver progress log for each run (off by default).",
    )
    return parser.parse_args()


def make_criterion_factory(args: argparse.Namespace) -> Callable[[], StoppingCriterion]:
    """Return a factory that produces a *fresh* stopping criterion per run.

    A fresh instance per run matters: MaxRuntime resets its clock, and
    MaxIterations / NoImprovement carry mutable counters that must never leak
    across runs.
    """
    if args.max_iterations is not None:
        return lambda: MaxIterations(args.max_iterations)
    if args.no_improvement is not None:
        return lambda: NoImprovement(args.no_improvement)
    # Default: runtime-based, 30 seconds when unspecified.
    seconds = args.max_runtime if args.max_runtime is not None else 30.0
    return lambda: MaxRuntime(seconds)


def build_params(args: argparse.Namespace) -> SolverParams:
    """Build SolverParams, overriding only the fields the user supplied.

    Unset fields stay ``None`` so the solver's problem-specific defaults apply
    (in particular DCNP's lighter population and CHNS budget).
    """
    overrides = {
        key: value
        for key, value in (
            ("population_size", args.population_size),
            ("offspring_count", args.offspring_count),
            ("search", args.search),
            ("transfer_interval", args.transfer_interval),
            ("partial_ratio", args.partial_ratio),
            ("beta", args.beta),
            ("chns_theta", args.chns_theta),
            ("chns_random_batch_max", args.chns_random_batch_max),
            ("chns_random_idle_product", args.chns_random_idle_product),
            ("chns_random_min_idle_steps", args.chns_random_min_idle_steps),
            ("chns_random_max_idle_steps", args.chns_random_max_idle_steps),
        )
        if value is not None
    }
    return SolverParams(**overrides)


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    # Per-variant argument requirements.
    if args.problem == "DCNP" and args.distance is None:
        raise SystemExit("--distance is required for --problem DCNP")
    if args.problem != "DCNP" and args.distance is not None:
        raise SystemExit("--distance only applies to --problem DCNP")
    if args.budget is None:
        raise SystemExit(f"--budget is required for --problem {args.problem}")

    model = Model.from_data(pypdms.read(args.instance))
    params = build_params(args)
    make_criterion = make_criterion_factory(args)

    # Assemble solve() keyword arguments per variant. distance is DCNP-only.
    solve_kwargs: dict = {"problem": args.problem, "budget": args.budget}
    if args.problem == "DCNP":
        solve_kwargs["distance"] = args.distance

    objectives: list[float] = []
    best_found_times: list[float] = []
    runtimes: list[float] = []

    # For DCNP, if population_size / transfer_interval are left at the CNP
    # library defaults, the solver internally substitutes DCNP-tuned values; so
    # show "auto" rather than the misleading library defaults the user did not
    # actually choose.
    if args.problem == "DCNP" and args.population_size is None:
        pop_display = "auto(DCNP)"
    else:
        pop_display = str(params.population_size)

    # Primary constraint: budget k (CNP/DCNP).
    constraint_line = f"Budget   : {args.budget}"
    objective_label = "objective"

    print(f"Instance : {args.instance}")
    print(f"Problem  : {args.problem}"
          + (f"  distance D={args.distance}" if args.problem == "DCNP" else ""))
    print(constraint_line)
    print(f"Runs     : {args.runs}")
    print(f"Search   : {params.search}  pop={pop_display}  "
          f"offspring={params.offspring_count}")
    print("-" * 64)
    print(f"{'run':>4} {'seed':>6} {objective_label:>14} "
          f"{'best@(s)':>10} {'runtime(s)':>11}")
    print("-" * 64)

    for i in range(args.runs):
        seed = args.seed + i
        result = model.solve(
            stopping_criterion=make_criterion(),
            seed=seed,
            params=params,
            display=args.show_solver_log,
            collect_stats=False,
            **solve_kwargs,
        )
        objectives.append(result.best_obj_value)
        best_found_times.append(result.best_found_at_time)
        runtimes.append(result.runtime)
        print(f"{i:>4} {seed:>6} {result.best_obj_value:>14} "
              f"{result.best_found_at_time:>10.2f} {result.runtime:>11.2f}")

    print("-" * 64)
    print("Summary over", args.runs, "run(s):")
    print(f"  Best objective        : {min(objectives)}")
    print(f"  Worst objective       : {max(objectives)}")
    print(f"  Average objective     : {statistics.mean(objectives):.2f}")
    if args.runs > 1:
        print(f"  Std dev objective     : {statistics.stdev(objectives):.2f}")
    print(f"  Avg time to best (s)  : {statistics.mean(best_found_times):.2f}")
    print(f"  Avg total runtime (s) : {statistics.mean(runtimes):.2f}")


if __name__ == "__main__":
    main()
