from __future__ import annotations

import os

from ._pypdms import ProblemData


def read(graph_file: str) -> ProblemData:
    """Read a graph file and return a ProblemData object.

    Supports the package adjacency-list format and DIMACS edge-list files
    (`p edge n m` / `e u v`).
    """
    graph_file = os.path.abspath(os.path.expanduser(graph_file))
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    with open(graph_file, encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("c"):
                continue
            # Normalize tabs to spaces for DIMACS format detection
            normalized = stripped.replace("\t", " ")
            if normalized.startswith("p ") or normalized.startswith("e "):
                return read_dimacs_edge_format(graph_file)
            break
    return read_adjacency_list_format(graph_file)


def read_adjacency_list_format(filename: str) -> ProblemData:
    """Read data from adjacency list format file."""
    return ProblemData.read_from_adjacency_list_file(filename)


def read_dimacs_edge_format(filename: str) -> ProblemData:
    """Read a DIMACS edge-list file (`p edge n m` / `e u v`)."""
    filename = os.path.abspath(os.path.expanduser(filename))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Graph file not found: {filename}")
    return ProblemData.read_from_dimacs_edge_file(filename)
