from __future__ import annotations

from .StoppingCriterion import StoppingCriterion


class MaxIterations(StoppingCriterion):
    def __init__(self, max_iterations: int):
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        self.max_iterations = max_iterations
        self.iterations = 0

    def __call__(self, best_obj_value: float) -> bool:
        if self.iterations >= self.max_iterations:
            return True
        self.iterations += 1
        return False

    def get_name(self) -> str:
        return "MaxIterations"
