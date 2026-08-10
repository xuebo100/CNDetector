from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import (
    DEFAULT_BETA,
    DEFAULT_DISPLAY_INTERVAL,
    DEFAULT_OFFSPRING_COUNT,
    DEFAULT_PARTIAL_RATIO,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_TRANSFER_INTERVAL,
)


@dataclass
class SolverParams:
    """Configuration for the IRMS dual-population solver.

    The ``chns_*`` fields override the CHNS local-search budget. They default
    to ``None`` (use the native defaults, tuned for CNP). They matter most for
    DCNP, whose objective rebuilds K-hop trees every step, making each CHNS run
    far more expensive than in CNP; capping the idle-step budget keeps a single
    local search cheap enough for the population to actually turn over within a
    time budget. See :meth:`Model.solve` for the DCNP-specific defaults.
    """

    population_size: int = DEFAULT_POPULATION_SIZE
    offspring_count: int = DEFAULT_OFFSPRING_COUNT
    transfer_interval: int = DEFAULT_TRANSFER_INTERVAL
    partial_ratio: float = DEFAULT_PARTIAL_RATIO
    beta: float = DEFAULT_BETA
    display_interval: float = DEFAULT_DISPLAY_INTERVAL
    search: str = "CHNS"

    # CHNS local-search budget overrides (None -> native default). See the
    # class docstring; these are the dominant cost knobs for DCNP.
    chns_max_idle_steps: Optional[int] = None
    chns_theta: Optional[float] = None
    chns_random_batch_max: Optional[int] = None
    chns_random_idle_product: Optional[int] = None
    chns_random_min_idle_steps: Optional[int] = None
    chns_random_max_idle_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if self.offspring_count < 1:
            raise ValueError("offspring_count must be >= 1.")
        if self.transfer_interval < 1:
            raise ValueError("transfer_interval must be >= 1.")
        if not 0 < self.partial_ratio < 1:
            raise ValueError("partial_ratio must be in (0, 1).")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1].")
        if self.display_interval <= 0:
            raise ValueError("display_interval must be positive.")
        if self.chns_max_idle_steps is not None and self.chns_max_idle_steps < 1:
            raise ValueError("chns_max_idle_steps must be >= 1.")
        if self.chns_theta is not None and not 0 <= self.chns_theta <= 1:
            raise ValueError("chns_theta must be in [0, 1].")
        if self.chns_random_batch_max is not None and self.chns_random_batch_max < 1:
            raise ValueError("chns_random_batch_max must be >= 1.")
        if (self.chns_random_idle_product is not None
                and self.chns_random_idle_product < 1):
            raise ValueError("chns_random_idle_product must be >= 1.")
        if (self.chns_random_min_idle_steps is not None
                and self.chns_random_min_idle_steps < 1):
            raise ValueError("chns_random_min_idle_steps must be >= 1.")
        if (self.chns_random_max_idle_steps is not None
                and self.chns_random_max_idle_steps < 1):
            raise ValueError("chns_random_max_idle_steps must be >= 1.")
        if (self.chns_random_min_idle_steps is not None
                and self.chns_random_max_idle_steps is not None
                and self.chns_random_min_idle_steps > self.chns_random_max_idle_steps):
            raise ValueError(
                "chns_random_min_idle_steps must be <= chns_random_max_idle_steps."
            )
