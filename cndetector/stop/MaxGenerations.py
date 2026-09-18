from __future__ import annotations

from .StoppingCriterion import StoppingCriterion


class MaxGenerations(StoppingCriterion):
    def __init__(self, max_generations: int):
        if not isinstance(max_generations, int) or max_generations <= 0:
            raise ValueError("max_generations must be a positive integer.")
        self.max_generations = max_generations
        self.generations = 0

    def __call__(self, best_obj_value: float) -> bool:
        if self.generations >= self.max_generations:
            return True
        self.generations += 1
        return False

    def get_name(self) -> str:
        return "MaxGenerations"
