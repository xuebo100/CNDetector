from __future__ import annotations

# Tuned solver defaults for CNP: theta = 10, kappa = 2, beta = 20,
# alpha = 0.05, gamma = 1000, delta = 500.
DEFAULT_POPULATION_SIZE = 10  # theta
DEFAULT_THREAD_COUNT = 2  # kappa
DEFAULT_INTERACTION_PERIOD = 20  # beta
DEFAULT_RELAXATION_COEFFICIENT = 0.05  # alpha
DEFAULT_ALLOWABLE_IDLE_ITERATIONS = 1000  # gamma
DEFAULT_STAGNATION_THRESHOLD = 500  # delta

# Probability with which the RSC crossover keeps a node of the backbone shared
# by the two parents. Not one of the tuned parameters above.
DEFAULT_BACKBONE_RATE = 0.9
DEFAULT_DISPLAY_INTERVAL = 1.0

PACKAGE_LOGGER_NAME = "cndetector"

CNP = "CNP"
DCNP = "DCNP"
L2NS = "L2NS"
