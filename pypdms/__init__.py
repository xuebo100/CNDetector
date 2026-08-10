from __future__ import annotations

import logging
import sys
from importlib import metadata

from .Model import Model
from .Result import Result
from .constants import (
    CHNS,
    CNP,
    DCNP,
    DEFAULT_DISPLAY_INTERVAL,
    PACKAGE_LOGGER_NAME,
)
from .params import SolverParams
from .read import read, read_adjacency_list_format, read_dimacs_edge_format
from .stop import (
    MaxIterations,
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
    __version__ = metadata.version("pypdms")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "CHNS",
    "CNP",
    "DCNP",
    "DEFAULT_DISPLAY_INTERVAL",
    "MaxIterations",
    "MaxRuntime",
    "Model",
    "NoImprovement",
    "Result",
    "SolverParams",
    "StoppingCriterion",
    "read",
    "read_adjacency_list_format",
    "read_dimacs_edge_format",
]
