from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class Result:
    """Container for the outcome of one CNDetector run.

    ``main_population`` is the final main population of the search, i.e. the
    pool of solutions that remove exactly ``budget`` nodes.
    """

    best_solution: Set[int] = field(default_factory=set)
    best_obj_value: float = math.inf
    num_iterations: int = 0
    runtime: float = 0.0
    best_found_at_time: float = 0.0
    stats: Optional[list[dict]] = None
    main_population: list[tuple[Set[int], int]] = field(default_factory=list)
    main_population_overlap_ratios: list[list[float]] = field(
        default_factory=list
    )
