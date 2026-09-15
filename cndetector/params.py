from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import (
    DEFAULT_BETA,
    DEFAULT_DISPLAY_INTERVAL,
    DEFAULT_INTERACTION_PERIOD,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RELAXATION_COEFFICIENT,
    DEFAULT_STAGNATION_THRESHOLD,
    DEFAULT_THREAD_COUNT,
)


@dataclass
class SolverParams:
    """Configuration for the IRMS dual-population solver.

    The ``l2ns_*`` fields override the L2NS local-search budget. They default
    to ``None`` (use the native defaults, tuned for CNP). They matter most for
    DCNP, whose objective rebuilds K-hop trees every step, making each L2NS run
    far more expensive than in CNP; capping the idle-step budget keeps a single
    local search cheap enough for the population to actually turn over within a
    time budget. See :meth:`Model.solve` for the DCNP-specific defaults.
    """

    # theta: size of each of the two populations.
    population_size: int = DEFAULT_POPULATION_SIZE
    # kappa: thread count; also the number of offspring per population per
    # generation, since Algorithm 2 runs one offspring per thread.
    thread_count: int = DEFAULT_THREAD_COUNT
    # beta: generations between two heterogeneous population cooperations.
    interaction_period: int = DEFAULT_INTERACTION_PERIOD
    # alpha: relaxation coefficient. The auxiliary population carries
    # floor(k * (1 - alpha)) nodes per solution; alpha = 0 makes it
    # identical to the main population.
    relaxation_coefficient: float = DEFAULT_RELAXATION_COEFFICIENT
    # delta: non-improving generations that trigger population reconstruction.
    stagnation_threshold: int = DEFAULT_STAGNATION_THRESHOLD
    beta: float = DEFAULT_BETA
    display_interval: float = DEFAULT_DISPLAY_INTERVAL
    search: str = "L2NS"

    # L2NS local-search budget overrides (None -> native default). See the
    # class docstring; these are the dominant cost knobs for DCNP.
    l2ns_max_idle_steps: Optional[int] = None
    l2ns_theta: Optional[float] = None
    l2ns_random_batch_max: Optional[int] = None
    l2ns_random_idle_product: Optional[int] = None
    l2ns_random_min_idle_steps: Optional[int] = None
    l2ns_random_max_idle_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if self.thread_count < 1:
            raise ValueError("thread_count must be >= 1.")
        if self.stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be >= 1.")
        if self.interaction_period < 1:
            raise ValueError("interaction_period must be >= 1.")
        if not 0 <= self.relaxation_coefficient < 1:
            raise ValueError("relaxation_coefficient must be in [0, 1).")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1].")
        if self.display_interval <= 0:
            raise ValueError("display_interval must be positive.")
        if self.l2ns_max_idle_steps is not None and self.l2ns_max_idle_steps < 1:
            raise ValueError("l2ns_max_idle_steps must be >= 1.")
        if self.l2ns_theta is not None and not 0 <= self.l2ns_theta <= 1:
            raise ValueError("l2ns_theta must be in [0, 1].")
        if self.l2ns_random_batch_max is not None and self.l2ns_random_batch_max < 1:
            raise ValueError("l2ns_random_batch_max must be >= 1.")
        if (self.l2ns_random_idle_product is not None
                and self.l2ns_random_idle_product < 1):
            raise ValueError("l2ns_random_idle_product must be >= 1.")
        if (self.l2ns_random_min_idle_steps is not None
                and self.l2ns_random_min_idle_steps < 1):
            raise ValueError("l2ns_random_min_idle_steps must be >= 1.")
        if (self.l2ns_random_max_idle_steps is not None
                and self.l2ns_random_max_idle_steps < 1):
            raise ValueError("l2ns_random_max_idle_steps must be >= 1.")
        if (self.l2ns_random_min_idle_steps is not None
                and self.l2ns_random_max_idle_steps is not None
                and self.l2ns_random_min_idle_steps > self.l2ns_random_max_idle_steps):
            raise ValueError(
                "l2ns_random_min_idle_steps must be <= l2ns_random_max_idle_steps."
            )
