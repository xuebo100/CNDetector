from __future__ import annotations

from .StoppingCriterion import StoppingCriterion


class NoImprovement(StoppingCriterion):
    def __init__(self, max_idle_iterations: int):
        if not isinstance(max_idle_iterations, int) or max_idle_iterations <= 0:
            raise ValueError("max_idle_iterations must be a positive integer.")
        self.max_idle_iterations = max_idle_iterations
        self.idle_iterations = 0
        self.last_best_obj_value = float("inf")

    def __call__(self, best_obj_value: float) -> bool:
        if best_obj_value < self.last_best_obj_value:
            self.last_best_obj_value = best_obj_value
            self.idle_iterations = 0
        else:
            self.idle_iterations += 1
        return self.idle_iterations >= self.max_idle_iterations

    def get_name(self) -> str:
        return "NoImprovement"
