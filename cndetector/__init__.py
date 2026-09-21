from __future__ import annotations

import logging
import sys
from importlib import metadata

from .Model import Model
from .Result import Result
from ._cndetector import get_max_threads, set_max_threads
from .constants import (
    CNP,
    DCNP,
    DEFAULT_DISPLAY_INTERVAL,
    L2NS,
    PACKAGE_LOGGER_NAME,
)
from .params import SolverParams
from .read import read, read_adjacency_list_format, read_dimacs_edge_format
from .stop import (
    MaxGenerations,
    MaxRuntime,
    NoImprovement,
    StoppingCriterion,
)

_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
if not _logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False

try:
    __version__ = metadata.version("cndetector")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "CNP",
    "DCNP",
    "DEFAULT_DISPLAY_INTERVAL",
    "L2NS",
    "MaxGenerations",
    "MaxRuntime",
    "Model",
    "NoImprovement",
    "Result",
    "SolverParams",
    "StoppingCriterion",
    "get_max_threads",
    "read",
    "read_adjacency_list_format",
    "read_dimacs_edge_format",
    "set_max_threads",
]
