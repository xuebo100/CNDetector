from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import (
    DEFAULT_ALLOWABLE_IDLE_ITERATIONS,
    DEFAULT_BACKBONE_RATE,
    DEFAULT_DISPLAY_INTERVAL,
    DEFAULT_INTERACTION_PERIOD,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RELAXATION_COEFFICIENT,
    DEFAULT_STAGNATION_THRESHOLD,
    DEFAULT_THREAD_COUNT,
)


@dataclass
class SolverParams:
    """Parameters of the parallel co-evolutionary memetic search of CNDetector.

    The first block holds the main algorithmic parameters; their defaults are
    the tuned values for CNP. The ``l2ns_*`` fields are fine-grained overrides
    of the local search's destroy-size and idle-budget schedule; they default
    to ``None``, meaning "keep the native value". See :meth:`Model.solve` for
    the DCNP-specific defaults.
    """

    # theta: size of each of the two populations.
    population_size: int = DEFAULT_POPULATION_SIZE
    # kappa: thread count; also the number of offspring per population per
    # generation, since each thread produces exactly one offspring.
    thread_count: int = DEFAULT_THREAD_COUNT
    # beta: generations between two heterogeneous population cooperations.
    interaction_period: int = DEFAULT_INTERACTION_PERIOD
    # alpha: relaxation coefficient. The auxiliary population carries
    # floor(k * (1 - alpha)) nodes per solution; alpha = 0 makes it
    # identical to the main population.
    relaxation_coefficient: float = DEFAULT_RELAXATION_COEFFICIENT
    # gamma: allowable number of idle iterations of one local-search run.
    allowable_idle_iterations: int = DEFAULT_ALLOWABLE_IDLE_ITERATIONS
    # delta: non-improving generations that trigger population reconstruction.
    stagnation_threshold: int = DEFAULT_STAGNATION_THRESHOLD

    # Probability with which the RSC crossover keeps a node of the backbone
    # shared by the two parents. Not one of the tuned parameters above.
    backbone_rate: float = DEFAULT_BACKBONE_RATE
    display_interval: float = DEFAULT_DISPLAY_INTERVAL
    search: str = "L2NS"

    # L2NS schedule overrides (None -> keep the native value). lambda is drawn
    # from [l2ns_min_destroy_size, l2ns_max_destroy_size] and the per-run idle
    # budget is clamped to [l2ns_idle_iteration_floor, l2ns_idle_iteration_cap].
    l2ns_min_destroy_size: Optional[int] = None
    l2ns_max_destroy_size: Optional[int] = None
    l2ns_idle_iteration_floor: Optional[int] = None
    l2ns_idle_iteration_cap: Optional[int] = None
    l2ns_impact_selection_rate: Optional[float] = None
    l2ns_adaptive_max_idle_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if self.thread_count < 1:
            raise ValueError("thread_count must be >= 1.")
        if self.interaction_period < 1:
            raise ValueError("interaction_period must be >= 1.")
        if not 0 <= self.relaxation_coefficient < 1:
            raise ValueError("relaxation_coefficient must be in [0, 1).")
        if self.allowable_idle_iterations < 1:
            raise ValueError("allowable_idle_iterations must be >= 1.")
        if self.stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be >= 1.")
        if not 0 <= self.backbone_rate <= 1:
            raise ValueError("backbone_rate must be in [0, 1].")
        if self.display_interval <= 0:
            raise ValueError("display_interval must be positive.")

        for field_name in (
            "l2ns_min_destroy_size",
            "l2ns_max_destroy_size",
            "l2ns_idle_iteration_floor",
            "l2ns_idle_iteration_cap",
            "l2ns_adaptive_max_idle_iterations",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be >= 1.")

        if (self.l2ns_impact_selection_rate is not None
                and not 0 <= self.l2ns_impact_selection_rate <= 1):
            raise ValueError("l2ns_impact_selection_rate must be in [0, 1].")
        if (self.l2ns_min_destroy_size is not None
                and self.l2ns_max_destroy_size is not None
                and self.l2ns_min_destroy_size > self.l2ns_max_destroy_size):
            raise ValueError(
                "l2ns_min_destroy_size must be <= l2ns_max_destroy_size."
            )
        if (self.l2ns_idle_iteration_floor is not None
                and self.l2ns_idle_iteration_cap is not None
                and self.l2ns_idle_iteration_floor > self.l2ns_idle_iteration_cap):
            raise ValueError(
                "l2ns_idle_iteration_floor must be <= l2ns_idle_iteration_cap."
            )
