"""Model entry point for building a graph and running the IRMS solver."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from .ProgressPrinter import ProgressPrinter
from .Result import Result
from .constants import (
    CNP,
    DCNP,
    DEFAULT_INTERACTION_PERIOD,
    DEFAULT_POPULATION_SIZE,
    PACKAGE_LOGGER_NAME,
)
from .params import SolverParams

if TYPE_CHECKING:
    from ._cndetector import ProblemData, SolverConfig
    from .stop import StoppingCriterion

logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# DCNP-specific L2NS budget. DCNP's objective rebuilds K-hop trees on every
# step, so a single L2NS run is far more expensive than in CNP. With the CNP
# budget (xi = 1000, capped at 500 idle iterations) one local search takes tens
# of seconds on 300+ node instances and the population never turns over. These
# lighter caps keep each L2NS run cheap enough for the dual population to evolve
# many generations within a few-minute budget. Tuned on USAir97 / Circuit /
# Ecoli (100-500 node instances). Any field left out here keeps the CNP value.
_DCNP_L2NS_DEFAULTS: dict[str, "int | float"] = {
    "random_idle_product": 100,
    "random_min_idle_steps": 20,
    "random_max_idle_steps": 80,
    "random_batch_max": 15,
    "theta": 0.3,
}

# DCNP-friendly dual-population defaults. DCNP's per-L2NS cost is high, so the
# CNP population size and exchange interval (theta = 10, beta = 20) leave the
# population barely past initialization within a time budget, and the
# feasible<->infeasible exchange rarely fires. A smaller population turns over
# faster and a short exchange interval lets the two populations actually mix.
# Applied only when the caller left these at the library defaults.
_DCNP_POPULATION_SIZE = 4
_DCNP_INTERACTION_PERIOD = 5


def _apply_l2ns_overrides(
    config: "SolverConfig",
    params: SolverParams,
    dcnp_defaults: Optional[dict[str, "int | float"]] = None,
) -> None:
    """Write the L2NS local-search budget into ``config.l2ns``.

    Priority: ``params.l2ns_*`` (explicitly set by the user) > ``dcnp_defaults``
    (problem-specific) > native C++ defaults, which implement the local-search
    budget of the paper.

    Args:
        config: native SolverConfig whose ``l2ns`` sub-config is mutated in place
        params: solver parameters, read for its ``l2ns_*`` override fields
        dcnp_defaults: problem-specific L2NS defaults (keys are l2ns field names,
            without the prefix)
    """
    # l2ns field name -> the corresponding override attribute on params
    field_to_param = {
        "max_idle_steps": "l2ns_max_idle_steps",
        "theta": "l2ns_theta",
        "random_batch_max": "l2ns_random_batch_max",
        "random_idle_product": "l2ns_random_idle_product",
        "random_min_idle_steps": "l2ns_random_min_idle_steps",
        "random_max_idle_steps": "l2ns_random_max_idle_steps",
    }
    defaults = dcnp_defaults or {}
    for l2ns_field, param_attr in field_to_param.items():
        override = getattr(params, param_attr, None)
        if override is not None:
            setattr(config.l2ns, l2ns_field, override)
        elif l2ns_field in defaults:
            setattr(config.l2ns, l2ns_field, defaults[l2ns_field])


def _normalize_feasible_population(
    feasible_population: list[tuple[set[int], int]],
) -> list[tuple[set[int], int]]:
    """Normalize the feasible population so solutions and objectives share types.

    Args:
        feasible_population: feasible population, each element a
            (solution set, objective value) tuple

    Returns:
        The normalized population, sorted by objective value, solution size and
        solution contents.
    """
    # Coerce each solution to a set and each objective to an int for consistency.
    normalized = [
        (set(solution), int(obj_value))
        for solution, obj_value in feasible_population
    ]
    # Sort by objective value, solution size and contents so results are stable.
    normalized.sort(key=lambda item: (item[1], len(item[0]), tuple(sorted(item[0]))))
    return normalized


def _compute_overlap_ratio_matrix(
    feasible_population: list[tuple[set[int], int]],
) -> list[list[float]]:
    """Compute the pairwise overlap-ratio matrix of the feasible population.

    The overlap ratio is the size of the intersection of two solutions divided
    by the size of the first solution.

    Args:
        feasible_population: feasible population, each element a
            (solution set, objective value) tuple

    Returns:
        The overlap-ratio matrix, where ``matrix[i][j]`` is the overlap ratio of
        solution ``i`` against solution ``j``.
    """
    overlap_ratios: list[list[float]] = []
    for solution, _ in feasible_population:
        denominator = len(solution)
        row: list[float] = []
        for other_solution, _ in feasible_population:
            if denominator == 0:
                # An empty solution has an overlap ratio of 0.
                row.append(0.0)
            else:
                # Fraction of the solution covered by the intersection.
                row.append(len(solution & other_solution) / denominator)
        overlap_ratios.append(row)
    return overlap_ratios


class Model:
    """Graph modeling and solving class for Critical Node Problems."""

    def __init__(self) -> None:
        """Initialize the model instance."""
        # Set of nodes.
        self.nodes: set[int] = set()
        # Graph structure as an adjacency list.
        self.adj_list: list[set[int]] = []
        # Cached problem-data object.
        self._problem_data: Optional[ProblemData] = None

    def add_node(self, node: int) -> None:
        """Add a node to the graph.

        Args:
            node: node ID, must be a non-negative integer

        Raises:
            ValueError: if the node ID is not a non-negative integer
        """
        if not isinstance(node, int) or node < 0:
            raise ValueError("Node ID must be a non-negative integer.")
        self.nodes.add(node)
        # Extend the adjacency list to accommodate the new node.
        while len(self.adj_list) <= node:
            self.adj_list.append(set())
        # The problem data must be rebuilt after adding a node.
        self._problem_data = None

    def add_edge(self, u: int, v: int) -> None:
        """Add an edge to the graph.

        Missing endpoints are created automatically.

        Args:
            u: first endpoint of the edge
            v: second endpoint of the edge
        """
        # Make sure both endpoints exist.
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)
        # Extend the adjacency list.
        while len(self.adj_list) <= max(u, v):
            self.adj_list.append(set())
        # Add the undirected edge (both directions).
        self.adj_list[u].add(v)
        self.adj_list[v].add(u)
        # The problem data must be rebuilt after adding an edge.
        self._problem_data = None

    @staticmethod
    def from_data(problem_data: "ProblemData") -> "Model":
        """Create a Model instance from a ProblemData object.

        Args:
            problem_data: problem-data object

        Returns:
            The created Model instance.
        """
        model = Model()
        model.nodes = set(problem_data.get_nodes_set())
        model.adj_list = [set(neighbors) for neighbors in problem_data.get_adj_list()]
        model._problem_data = problem_data
        return model

    @property
    def problem_data(self) -> "ProblemData":
        """Return the problem-data object, creating it if it does not exist."""
        if self._problem_data is None:
            self._problem_data = self._create_problem_data()
        return self._problem_data

    def _create_problem_data(self) -> "ProblemData":
        """Create the ProblemData object.

        Returns:
            The created ProblemData object.
        """
        from ._cndetector import ProblemData

        # Determine the largest node ID.
        max_node_id = max(self.nodes) if self.nodes else 0
        # Create the problem data with size max node ID + 1.
        problem_data = ProblemData(max_node_id + 1)
        # Add all nodes.
        for node in self.nodes:
            problem_data.add_node(node)
        # Add every edge once (u < v avoids duplicates).
        for u in range(len(self.adj_list)):
            for v in self.adj_list[u]:
                if u < v:
                    problem_data.add_edge(u, v)
        return problem_data

    def solve(
        self,
        budget: Optional[int] = None,
        stopping_criterion: Optional["StoppingCriterion"] = None,
        seed: int = 0,
        params: Optional[SolverParams] = None,
        population_size: Optional[int] = None,
        thread_count: Optional[int] = None,
        search: Optional[str] = None,
        display_interval: Optional[float] = None,
        display: bool = True,
        collect_stats: bool = True,
        problem: str = CNP,
        distance: Optional[int] = None,
    ) -> Result:
        """Solve a Critical Node Problem on the graph.

        Two problem variants are supported via the ``problem`` argument:

        - ``"CNP"`` (default): budget-constrained CNP. Given ``budget`` ``k``,
          minimize the residual pairwise connectivity ``Σ |C|(|C|-1)/2``.
        - ``"DCNP"``: distance-based Critical Node Problem. Given ``budget``
          ``k`` and distance ``distance`` ``D``, minimize the number of unordered
          node pairs at distance at most ``D`` in the residual graph.

        Parameters
        ----------
        budget
            Number of nodes to remove (required, ``1 <= budget < |V|``).
        stopping_criterion
            Callable that stops the solver when it returns ``True``.
        seed
            Random number generator seed. ``0`` is a valid seed.
        params
            Tunable solver parameters. Defaults to :class:`SolverParams()`.
        population_size, thread_count, search
            Convenience overrides for the corresponding ``params`` fields.
        display_interval
            Minimum seconds between iteration log lines. Defaults to
            ``params.display_interval``.
        display
            If ``True``, log progress to the package logger.
        collect_stats
            If ``True``, record a per-iteration trace in :attr:`Result.stats`.
        problem
            ``"CNP"`` (default) or ``"DCNP"``.
        distance
            DCNP distance threshold ``D`` (required for DCNP, ``D >= 1``).
        """
        if stopping_criterion is None:
            raise ValueError("stopping_criterion is required")

        # Resolve parameters, applying the direct keyword overrides.
        params = self._resolve_params(
            params, population_size, thread_count, search
        )
        effective_display_interval = (
            display_interval if display_interval is not None
            else params.display_interval
        )

        # Normalize the problem name.
        normalized_problem = str(problem).upper().replace("_", "-")
        if normalized_problem == DCNP:
            if not isinstance(budget, int) or budget < 1:
                raise ValueError("budget must be a positive integer")
            if budget >= len(self.nodes):
                raise ValueError(
                    f"budget ({budget}) must be smaller than the number of "
                    f"nodes ({len(self.nodes)})"
                )
            if not isinstance(distance, int) or distance < 1:
                raise ValueError("distance must be a positive integer for DCNP")
            return self._solve_dcnp(
                budget, distance, stopping_criterion, seed, params,
                effective_display_interval, display, collect_stats,
            )

        if normalized_problem != CNP:
            raise ValueError(f"unknown problem {problem!r}; expected 'CNP' or 'DCNP'")

        # Validate the budget argument.
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        if budget >= len(self.nodes):
            raise ValueError(
                f"budget ({budget}) must be smaller than the number of "
                f"nodes ({len(self.nodes)})"
            )

        max_runtime = getattr(stopping_criterion, "max_runtime", None)
        return self._solve_fixed_budget(
            budget, stopping_criterion, seed, params,
            effective_display_interval, display, collect_stats,
            max_runtime if isinstance(max_runtime, (int, float)) else None,
        )

    @staticmethod
    def _resolve_params(
        params: Optional[SolverParams],
        population_size: Optional[int],
        thread_count: Optional[int],
        search: Optional[str],
    ) -> SolverParams:
        """Apply direct keyword overrides on top of ``SolverParams``.

        Uses :func:`dataclasses.replace` so overrides are re-validated and the
        caller's ``params`` instance is never mutated.

        Args:
            params: solver parameters object
            population_size: population-size override
            thread_count: thread-count (kappa) override
            search: search-strategy override

        Returns:
            The resolved solver parameters.
        """
        from dataclasses import replace

        if params is None:
            params = SolverParams()

        # Build the override dict with the non-None values only.
        overrides: dict[str, Any] = {
            key: value
            for key, value in (
                ("population_size", population_size),
                ("thread_count", thread_count),
                ("search", search),
            )
            if value is not None
        }
        return replace(params, **overrides) if overrides else params

    # Solve the CNP problem by calling _run_solver.
    def _solve_fixed_budget(
        self,
        budget: int,
        stopping_criterion: Callable[[float], bool],
        seed: int,
        params: SolverParams,
        effective_display_interval: float,
        display: bool,
        collect_stats: bool,
        max_runtime: Optional[float],
    ) -> Result:
        """Run one IRMS search at a fixed budget, minimizing pairwise
        connectivity (CNP1).

        When ``max_runtime`` (seconds) is positive it is pushed to the native
        solver as a hard wall-clock deadline, so it can stop during population
        initialization and within a single generation.

        Args:
            budget: budget of nodes to remove
            stopping_criterion: stopping criterion
            seed: random seed
            params: solver parameters
            effective_display_interval: effective display interval
            display: whether to display progress
            collect_stats: whether to collect statistics
            max_runtime: maximum runtime

        Returns:
            The solve result.
        """
        from ._cndetector import SolverConfig

        # Create the original graph.
        original_graph = self.problem_data.create_original_graph(
            budget, seed
        )

        # Configure the solver.
        config = SolverConfig()
        config.population_size = params.population_size
        config.thread_count = params.thread_count
        config.stagnation_threshold = params.stagnation_threshold
        config.interaction_period = params.interaction_period
        config.seed = seed
        config.relaxation_coefficient = params.relaxation_coefficient
        config.beta = params.beta
        config.display_interval = effective_display_interval
        config.search = params.search
        if isinstance(max_runtime, (int, float)) and max_runtime > 0:
            config.max_runtime = float(max_runtime)

        return self._run_solver(
            original_graph, budget, config,
            stopping_criterion, display, effective_display_interval,
            collect_stats,
        )

    def _solve_dcnp(
        self,
        budget: int,
        distance: int,
        stopping_criterion: Callable[[float], bool],
        seed: int,
        params: SolverParams,
        effective_display_interval: float,
        display: bool,
        collect_stats: bool,
    ) -> Result:
        """Solve fixed-budget DCNP with the same dual-population IRMS flow as CNP.

        The objective is the number of unordered node pairs at distance at most
        ``distance`` in the graph after removing ``budget`` nodes; lower is
        better.

        The process is identical to CNP: maintain a feasible (budget ``k``) and
        an infeasible (partial budget ``floor(k * (1 - alpha))``) population, each
        generation producing offspring via RSC crossover plus L2NS local search
        and pruning the population by cost + diversity ranking; every
        ``interaction_period`` generations the best infeasible solution is
        completed to the full budget and injected into the feasible population.
        """
        from ._cndetector import SolverConfig

        # Create the original DCNP graph.
        original_graph = self.problem_data.create_original_dcnp_graph(
            budget, distance, seed
        )

        # A single L2NS run is expensive for DCNP; the CNP population size and
        # exchange interval leave the population barely past initialization
        # within a time budget and the exchange rarely fires. If the caller left
        # these at the library defaults, replace them with a small population
        # and a short exchange interval better suited to DCNP.
        dcnp_population_size = (
            _DCNP_POPULATION_SIZE
            if params.population_size == DEFAULT_POPULATION_SIZE
            else params.population_size
        )
        dcnp_interaction_period = (
            _DCNP_INTERACTION_PERIOD
            if params.interaction_period == DEFAULT_INTERACTION_PERIOD
            else params.interaction_period
        )

        # Configure the solver (same as the CNP path).
        config = SolverConfig()
        config.population_size = dcnp_population_size
        config.thread_count = params.thread_count
        config.stagnation_threshold = params.stagnation_threshold
        config.interaction_period = dcnp_interaction_period
        config.seed = seed
        config.relaxation_coefficient = params.relaxation_coefficient
        config.beta = params.beta
        config.display_interval = effective_display_interval
        config.search = params.search

        # Lighter L2NS budget for DCNP; explicit params.l2ns_* still override it.
        _apply_l2ns_overrides(config, params, dcnp_defaults=_DCNP_L2NS_DEFAULTS)

        max_runtime = getattr(stopping_criterion, "max_runtime", None)
        if isinstance(max_runtime, (int, float)) and max_runtime > 0:
            config.max_runtime = float(max_runtime)

        return self._run_solver(
            original_graph, budget, config,
            stopping_criterion, display, effective_display_interval,
            collect_stats, dcnp=True,
        )

    def _run_solver(
        self,
        original_graph,
        budget: int,
        config,
        stopping_criterion: Callable[[float], bool],
        display: bool,
        display_interval: float,
        collect_stats: bool,
        dcnp: bool = False,
    ) -> Result:
        """Run the dual-population solver (shared by CNP and DCNP).

        Args:
            original_graph: the original graph object (CNP_Graph or DCNP_Graph)
            budget: budget
            config: solver configuration
            stopping_criterion: stopping criterion
            display: whether to display progress
            display_interval: display interval
            collect_stats: whether to collect statistics
            dcnp: whether this is a DCNP problem (selects the dual-population
                implementation)

        Returns:
            The solve result.
        """
        from ._cndetector import DCNPDualPopulation, DualPopulation, set_max_threads

        population_cls = DCNPDualPopulation if dcnp else DualPopulation

        # kappa is the thread count: one worker per offspring of a generation.
        set_max_threads(config.thread_count)

        start_time = time.perf_counter()

        # Align the time-based criterion clock with the actual solver start so
        # the reported runtime and the native deadline share one reference.
        # Otherwise the criterion's clock starts at construction time (before
        # the graph is set up), shrinking the effective budget.
        if hasattr(stopping_criterion, "start_time"):
            stopping_criterion.start_time = start_time

        # The infeasible solution removes fewer nodes than the budget allows,
        # capped at budget-1 so it stays strictly infeasible (and >=1 so the
        # population has something to evolve). When budget=1 we fall back to 1,
        # which equals the feasible budget.
        infeasible_budget = max(1, min(
            math.floor(budget * (1.0 - config.relaxation_coefficient)),
            budget - 1,
        ))

        population = population_cls(
            original_graph, budget, infeasible_budget, config,
        )

        printer = ProgressPrinter(
            should_print=display,
            logger=logger,
            display_interval=display_interval,
        )
        printer.start(budget, config.seed)
        printer.initializing_population_message()

        init_result = population.initialize()
        best_solution = set(init_result[0])
        best_obj_value = init_result[1]
        best_found_at_time = time.perf_counter() - start_time

        printer.print_iterations_header()

        iterations = 0
        idle_generations = 0
        stats: list[dict] = []

        # Main solve loop.
        while not stopping_criterion(best_obj_value):
            population.advance_one_generation()
            iterations += 1

            # Handle exchange events.
            for event in population.drain_exchange_events():
                report = event.report
                printer.exchange(event.iteration, {
                    "exchange_triggered": report.exchange_triggered,
                    "first_population_candidate_obj":
                        report.first_population_candidate_obj,
                    "first_population_improved_best":
                        report.first_population_improved_best,
                })

            # Handle iteration events.
            for iter_event in population.drain_iteration_events():
                elapsed = time.perf_counter() - start_time
                if iter_event.best_objective < best_obj_value:
                    best_obj_value = iter_event.best_objective
                    best_found_at_time = elapsed
                    idle_generations = 0
                else:
                    idle_generations += 1

                if collect_stats:
                    stats.append({
                        "iteration": iter_event.iteration,
                        "elapsed": elapsed,
                        "best_obj_value": best_obj_value,
                        "idle_generations": idle_generations,
                        "population_size": iter_event.population_size,
                    })

                printer.iteration(
                    iter_event.iteration, elapsed, best_obj_value,
                    idle_generations, iter_event.population_size,
                )

        # Fetch the final solution.
        final_sol, final_obj = population.get_best_feasible_solution()
        best_solution = set(final_sol)
        best_obj_value = min(best_obj_value, final_obj)
        runtime = time.perf_counter() - start_time

        # Normalize the feasible population.
        raw_pop = population.get_feasible_population()
        feasible_population = _normalize_feasible_population(
            [(set(s), v) for s, v in raw_pop]
        )
        overlap_ratios = _compute_overlap_ratio_matrix(feasible_population)

        result = Result(
            best_solution=best_solution,
            best_obj_value=best_obj_value,
            num_iterations=iterations,
            runtime=runtime,
            best_found_at_time=best_found_at_time,
            stats=stats if collect_stats else None,
            feasible_population=feasible_population,
            feasible_population_overlap_ratios=overlap_ratios,
        )

        printer.end(result)
        return result
