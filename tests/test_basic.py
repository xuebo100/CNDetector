"""Tests for the CNDetector Python API."""

import pytest

import cndetector
from cndetector import (
    MaxIterations,
    MaxRuntime,
    Model,
    NoImprovement,
    SolverParams,
)
from cndetector._cndetector import ProblemData


def _cycle_with_chords_model() -> Model:
    """Return a small fixed graph used by several tests."""
    model = Model()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    for u, v in edges:
        model.add_edge(u, v)
    return model


def test_version():
    assert cndetector.__version__ is not None


def test_model_add_nodes_and_edges():
    model = Model()
    model.add_node(0)
    model.add_node(1)
    model.add_node(2)
    model.add_edge(0, 1)
    model.add_edge(1, 2)
    assert {0, 1, 2}.issubset(model.nodes)


def test_add_edge_creates_implicit_nodes():
    model = Model()
    model.add_edge(3, 7)
    assert 3 in model.nodes
    assert 7 in model.nodes


def test_add_node_rejects_negative():
    model = Model()
    with pytest.raises(ValueError):
        model.add_node(-1)


def test_solve_small_graph():
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=2,
        stopping_criterion=MaxIterations(5),
        seed=42,
        display=False,
    )
    assert len(result.best_solution) == 2
    assert result.best_obj_value >= 0
    assert result.num_iterations == 5


def test_solve_rejects_invalid_budget():
    model = _cycle_with_chords_model()
    with pytest.raises(ValueError):
        model.solve(
            budget=0,
            stopping_criterion=MaxIterations(1),
            seed=1,
            display=False,
        )
    with pytest.raises(ValueError):
        model.solve(
            budget=10,  # equal to n -> invalid
            stopping_criterion=MaxIterations(1),
            seed=1,
            display=False,
        )


def test_solve_with_seed_zero():
    """seed=0 must be accepted (no silent replacement)."""
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=2,
        stopping_criterion=MaxIterations(3),
        seed=0,
        display=False,
    )
    assert len(result.best_solution) == 2


def test_solve_is_deterministic_for_same_seed():
    model = _cycle_with_chords_model()
    out_a = model.solve(
        budget=3, stopping_criterion=MaxIterations(10),
        seed=7, display=False,
    )
    out_b = model.solve(
        budget=3, stopping_criterion=MaxIterations(10),
        seed=7, display=False,
    )
    assert out_a.best_obj_value == out_b.best_obj_value
    assert out_a.best_solution == out_b.best_solution


def test_max_runtime_criterion():
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=3,
        stopping_criterion=MaxRuntime(0.2),
        seed=1,
        display=False,
    )
    assert result.runtime >= 0.2 - 0.05  # allow small slack
    assert result.num_iterations >= 1


def test_max_runtime_does_not_overshoot():
    """The native deadline must stop the solver close to the limit, even
    when most of the budget is spent inside population initialization."""
    model = _cycle_with_chords_model()
    limit = 0.5
    # A large population makes initialization (one local search per member)
    # the dominant cost; before the deadline was threaded into the C++ core,
    # initialization ran to completion and blew past the limit.
    params = SolverParams(population_size=50)
    result = model.solve(
        budget=3,
        stopping_criterion=MaxRuntime(limit),
        seed=1,
        params=params,
        display=False,
    )
    # Generous upper bound: precise stopping keeps this well under 2x.
    assert result.runtime <= limit + 0.25


def test_no_improvement_criterion_stops():
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=3,
        stopping_criterion=NoImprovement(5),
        seed=1,
        display=False,
    )
    assert result.num_iterations >= 1


def test_solver_params_validation():
    with pytest.raises(ValueError):
        SolverParams(population_size=1)
    with pytest.raises(ValueError):
        SolverParams(thread_count=0)
    with pytest.raises(ValueError):
        SolverParams(interaction_period=0)
    with pytest.raises(ValueError):
        SolverParams(relaxation_coefficient=-0.1)
    with pytest.raises(ValueError):
        SolverParams(relaxation_coefficient=1.0)
    # alpha = 0 is valid: it makes the auxiliary population identical to the
    # main one (see the paper's definition of the relaxation coefficient).
    SolverParams(relaxation_coefficient=0.0)
    with pytest.raises(ValueError):
        SolverParams(allowable_idle_iterations=0)
    with pytest.raises(ValueError):
        SolverParams(stagnation_threshold=0)
    with pytest.raises(ValueError):
        SolverParams(backbone_rate=-0.1)
    with pytest.raises(ValueError):
        SolverParams(backbone_rate=1.1)


def test_custom_solver_params():
    model = _cycle_with_chords_model()
    params = SolverParams(
        population_size=4,
        thread_count=1,
        interaction_period=5,
        relaxation_coefficient=0.5,
        allowable_idle_iterations=200,
        backbone_rate=0.7,
        search="L2NS",
    )
    result = model.solve(
        budget=3,
        stopping_criterion=MaxIterations(10),
        seed=42,
        params=params,
        display=False,
    )
    assert len(result.best_solution) == 3


def test_result_stats_collected_by_default():
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=2,
        stopping_criterion=MaxIterations(5),
        seed=1,
        display=False,
    )
    assert result.stats is not None
    assert len(result.stats) >= 1
    assert {"iteration", "best_obj_value"}.issubset(result.stats[0].keys())


def test_result_stats_disabled():
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=2,
        stopping_criterion=MaxIterations(5),
        seed=1,
        display=False,
        collect_stats=False,
    )
    assert result.stats is None


def test_main_population_normalized():
    model = _cycle_with_chords_model()
    result = model.solve(
        budget=2,
        stopping_criterion=MaxIterations(10),
        seed=1,
        display=False,
    )
    assert len(result.main_population) >= 1
    overlaps = result.main_population_overlap_ratios
    assert len(overlaps) == len(result.main_population)


def test_cpp_graph_add_node_restores_connectivity():
    problem_data = ProblemData(5)
    for i in range(5):
        problem_data.add_node(i)
    for i in range(4):
        problem_data.add_edge(i, i + 1)

    graph = problem_data.create_original_graph(budget=1, seed=42)
    graph = graph.get_random_full_budget_graph(seed=1)

    removed = graph.get_removed_nodes()
    assert len(removed) == 1
    node = next(iter(removed))
    # Removing `node` from the path 0-1-2-3-4 leaves two segments.
    left, right = node, 4 - node
    assert graph.get_objective_value() == (
        left * (left - 1) // 2 + right * (right - 1) // 2
    )

    graph.add_node(node)
    assert graph.get_removed_nodes() == set()
    assert graph.get_objective_value() == 10


def test_cpp_random_feasible_graph_respects_seed():
    pd = ProblemData(10)
    for i in range(10):
        pd.add_node(i)
    for i in range(9):
        pd.add_edge(i, i + 1)

    graph = pd.create_original_graph(budget=3, seed=42)
    g1 = graph.get_random_full_budget_graph(seed=123)
    g2 = graph.get_random_full_budget_graph(seed=123)
    assert g1.get_removed_nodes() == g2.get_removed_nodes()


def _path_model(n: int) -> Model:
    """Return a path graph 0-1-...-(n-1)."""
    model = Model()
    for i in range(n - 1):
        model.add_edge(i, i + 1)
    return model


# --- DCNP (distance-based CNP) -----------------------------------------------


def _distance_pair_count(model: Model, removed: set[int], distance: int) -> int:
    total = 0
    for source in model.nodes:
        if source in removed:
            continue
        seen = {source}
        queue = [(source, 0)]
        head = 0
        while head < len(queue):
            node, depth = queue[head]
            head += 1
            if depth == distance:
                continue
            for neighbor in model.adj_list[node]:
                if neighbor in removed or neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
        total += sum(1 for target in seen if target > source)
    return total


def test_dcnp_graph_objective_counts_d_hop_pairs():
    model = _path_model(4)
    graph = model.problem_data.create_original_dcnp_graph(
        budget=1, distance=2, seed=1,
    )
    # Path 0-1-2-3 has five unordered pairs at distance <= 2.
    assert graph.get_objective_value() == 5
    graph.update_graph_by_removed_nodes({1})
    assert graph.get_objective_value() == 1


def test_dcnp_solve_returns_budget_sized_solution():
    model = _path_model(4)
    result = model.solve(
        problem="DCNP",
        budget=1,
        distance=2,
        stopping_criterion=MaxIterations(2),
        seed=1,
        display=False,
    )
    assert len(result.best_solution) == 1
    assert result.best_obj_value == _distance_pair_count(
        model, result.best_solution, 2,
    )
    assert result.best_obj_value == 1


def test_dcnp_requires_distance():
    model = _path_model(4)
    with pytest.raises(ValueError):
        model.solve(
            problem="DCNP",
            budget=1,
            stopping_criterion=MaxIterations(1),
            seed=1,
            display=False,
        )


def test_read_dimacs_edge_format(tmp_path):
    graph_file = tmp_path / "graph.dimacs"
    graph_file.write_text(
        "c tiny graph\np edge 3 2\ne 0 1\ne 1 2\n",
        encoding="utf-8",
    )
    data = cndetector.read_dimacs_edge_format(str(graph_file))
    assert data.num_nodes() == 3
    assert data.get_adj_list()[1] == {0, 2}
    auto_data = cndetector.read(str(graph_file))
    assert auto_data.num_nodes() == 3
    assert auto_data.get_adj_list()[1] == {0, 2}
