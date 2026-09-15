from __future__ import annotations

# Solver defaults follow the irace-tuned parameter settings of the paper:
# theta = 10, kappa = 2, beta = 20, alpha = 0.05, xi = 1000, delta = 500.
DEFAULT_POPULATION_SIZE = 10  # theta
DEFAULT_THREAD_COUNT = 2  # kappa
DEFAULT_INTERACTION_PERIOD = 20  # beta
DEFAULT_RELAXATION_COEFFICIENT = 0.05  # alpha
DEFAULT_STAGNATION_THRESHOLD = 500  # delta
DEFAULT_BETA = 0.9
DEFAULT_DISPLAY_INTERVAL = 1.0

PACKAGE_LOGGER_NAME = "cndetector"

CNP = "CNP"
DCNP = "DCNP"
L2NS = "L2NS"
