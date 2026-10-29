from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import (
    CHNS,
    CLNS,
    DBX,
    DEFAULT_BACKBONE_RATE,
    DEFAULT_DISPLAY_INTERVAL,
    DEFAULT_RELAXATION_COEFFICIENT,
    DEFAULT_STAGNATION_THRESHOLD,
    DEFAULT_THREAD_COUNT,
    RSC,
)


@dataclass
class SolverParams:
    """Parameters of the parallel co-evolutionary memetic search of CNDetector.

    The first block holds the main algorithmic parameters. ``population_size``,
    ``interaction_period`` and ``allowable_idle_iterations`` are tuned
    separately for CNP and DCNP; leaving them at ``None`` selects the tuned
    value of the problem being solved, while an explicit value is always used
    as given. The ``clns_*`` fields are fine-grained overrides of the local
    search's destroy-size and idle-budget schedule; they also default to
    ``None``, meaning "keep the problem's default".
    """

    # theta: size of each of the two populations (None -> 10 for CNP, 4 for
    # DCNP).
    population_size: Optional[int] = None
    # kappa: thread count. Each population produces this many offspring per
    # generation, and the search runs on up to this many worker threads.
    thread_count: int = DEFAULT_THREAD_COUNT
    # beta: generations between two heterogeneous population cooperations
    # (None -> 20 for CNP, 5 for DCNP).
    interaction_period: Optional[int] = None
    # alpha: relaxation coefficient. The auxiliary population carries
    # floor(k * (1 - alpha)) nodes per solution; alpha = 0 makes it
    # identical to the main population.
    relaxation_coefficient: float = DEFAULT_RELAXATION_COEFFICIENT
    # gamma: allowable number of idle iterations of one local-search run
    # (None -> 1000 for CNP, 100 for DCNP).
    allowable_idle_iterations: Optional[int] = None
    # delta: non-improving generations that trigger population reconstruction.
    stagnation_threshold: int = DEFAULT_STAGNATION_THRESHOLD

    # Probability with which the RSC crossover keeps a node of the backbone
    # shared by the two parents. Not one of the tuned parameters above.
    backbone_rate: float = DEFAULT_BACKBONE_RATE
    display_interval: float = DEFAULT_DISPLAY_INTERVAL
    # Local search refining every new solution: "CLNS", or "CHNS", the
    # component-based hybrid neighborhood search it replaces (CNP only).
    search: str = CLNS
    # Crossover producing the offspring: "RSC", or "DBX", the double
    # backbone-based crossover it replaces (CNP only).
    crossover: str = RSC

    # CLNS schedule overrides (None -> keep the problem's default). lambda is
    # drawn from [clns_min_destroy_size, clns_max_destroy_size] and the per-run
    # idle budget is clamped to [clns_idle_iteration_floor,
    # clns_idle_iteration_cap].
    clns_min_destroy_size: Optional[int] = None
    clns_max_destroy_size: Optional[int] = None
    clns_idle_iteration_floor: Optional[int] = None
    clns_idle_iteration_cap: Optional[int] = None
    clns_impact_selection_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.population_size is not None and self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if self.thread_count < 1:
            raise ValueError("thread_count must be >= 1.")
        if self.interaction_period is not None and self.interaction_period < 1:
            raise ValueError("interaction_period must be >= 1.")
        if not 0 <= self.relaxation_coefficient < 1:
            raise ValueError("relaxation_coefficient must be in [0, 1).")
        if (self.allowable_idle_iterations is not None
                and self.allowable_idle_iterations < 1):
            raise ValueError("allowable_idle_iterations must be >= 1.")
        if self.stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be >= 1.")
        if not 0 <= self.backbone_rate <= 1:
            raise ValueError("backbone_rate must be in [0, 1].")
        if self.display_interval <= 0:
            raise ValueError("display_interval must be positive.")
        if self.search not in (CLNS, CHNS):
            raise ValueError(f"search must be {CLNS!r} or {CHNS!r}.")
        if self.crossover not in (RSC, DBX):
            raise ValueError(f"crossover must be {RSC!r} or {DBX!r}.")

        for field_name in (
            "clns_min_destroy_size",
            "clns_max_destroy_size",
            "clns_idle_iteration_floor",
            "clns_idle_iteration_cap",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be >= 1.")

        if (self.clns_impact_selection_rate is not None
                and not 0 <= self.clns_impact_selection_rate <= 1):
            raise ValueError("clns_impact_selection_rate must be in [0, 1].")
        if (self.clns_min_destroy_size is not None
                and self.clns_max_destroy_size is not None
                and self.clns_min_destroy_size > self.clns_max_destroy_size):
            raise ValueError(
                "clns_min_destroy_size must be <= clns_max_destroy_size."
            )
        if (self.clns_idle_iteration_floor is not None
                and self.clns_idle_iteration_cap is not None
                and self.clns_idle_iteration_floor > self.clns_idle_iteration_cap):
            raise ValueError(
                "clns_idle_iteration_floor must be <= clns_idle_iteration_cap."
            )
