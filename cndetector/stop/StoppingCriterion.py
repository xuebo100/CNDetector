from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StoppingCriterion(Protocol):
    def __call__(self, best_obj_value: float) -> bool: ...

    def get_name(self) -> str:
        return self.__class__.__name__
