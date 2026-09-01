from __future__ import annotations

import time

from .StoppingCriterion import StoppingCriterion


class MaxRuntime(StoppingCriterion):
    def __init__(self, max_runtime_in_sec: float):
        if not isinstance(max_runtime_in_sec, (int, float)) or max_runtime_in_sec <= 0:
            raise ValueError("max_runtime_in_sec must be a positive number.")
        self.max_runtime = max_runtime_in_sec
        self.start_time = time.perf_counter()

    def __call__(self, best_obj_value: float) -> bool:
        return time.perf_counter() - self.start_time >= self.max_runtime

    def get_name(self) -> str:
        return "MaxRuntime"
