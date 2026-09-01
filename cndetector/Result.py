from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class Result:
    """Container for solver run results."""

    best_solution: Set[int] = field(default_factory=set)
    best_obj_value: float = math.inf
    num_iterations: int = 0
    runtime: float = 0.0
    best_found_at_time: float = 0.0
    stats: Optional[list[dict]] = None
    feasible_population: list[tuple[Set[int], int]] = field(default_factory=list)
    feasible_population_overlap_ratios: list[list[float]] = field(
        default_factory=list
    )
