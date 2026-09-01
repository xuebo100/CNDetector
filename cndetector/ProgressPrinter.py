from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Result import Result


class ProgressPrinter:
    def __init__(
        self,
        should_print: bool,
        logger: logging.Logger,
        display_interval: float = 60.0,
    ):
        self._should_print = should_print
        self._logger = logger
        self._display_interval = display_interval
        self._last_print_time = time.perf_counter()
        self._last_printed_iteration = 0
        self._current_best = float("inf")

    def start(self, budget: int, seed: int) -> None:
        if not self._should_print:
            return
        self._logger.info("CNDetector IRMS Solver")
        self._logger.info("  Budget: %d", budget)
        self._logger.info("  Seed:   %d", seed)

    def initializing_population_message(self) -> None:
        if not self._should_print:
            return
        self._logger.info(
            "-----------------------Initializing population-----------------------"
        )
        self._last_print_time = time.perf_counter()

    def print_iterations_header(self) -> None:
        if not self._should_print:
            return
        self._logger.info(
            "--------------------Starting Population Iterations--------------------"
        )
        self._logger.info(
            "  Iter        | Time        | Best        | Idle    | PopSize"
        )
        self._last_print_time = time.perf_counter()

    def iteration(
        self,
        iteration: int,
        elapsed: float,
        best_obj: int,
        idle_gens: int,
        pop_size: int,
        force: bool = False,
    ) -> None:
        if not self._should_print:
            return
        now = time.perf_counter()
        if not force and (now - self._last_print_time) < self._display_interval:
            return

        indicator = "*" if best_obj < self._current_best else " "
        self._current_best = min(self._current_best, best_obj)

        self._logger.info(
            "%sIter %5d | Time: %6.2fs | Best: %-10d | Idle: %4d | PopSize: %3d",
            indicator,
            iteration,
            elapsed,
            best_obj,
            idle_gens,
            pop_size,
        )
        self._last_print_time = now
        self._last_printed_iteration = iteration

    def exchange(self, iteration: int, report: dict) -> None:
        if not self._should_print:
            return
        if not report.get("exchange_triggered", False):
            return
        self._logger.info(
            "[Exchange] Iter %5d | Candidate obj: %10d | Improved best: %s",
            iteration,
            report.get("first_population_candidate_obj", -1),
            "yes" if report.get("first_population_improved_best", False) else "no",
        )

    def end(self, result: "Result") -> None:
        if not self._should_print:
            return
        self._logger.info(
            "----------------------------------------------------------------------"
        )
        self._logger.info("IRMS finished.")
        self._logger.info("  Total iterations: %d", result.num_iterations)
        self._logger.info("  Total runtime: %.2f seconds", result.runtime)
        self._logger.info("  Best objective: %d", result.best_obj_value)
        self._logger.info("  Solution size: %d", len(result.best_solution))
