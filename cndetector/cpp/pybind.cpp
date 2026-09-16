#include "Graph/CNP_Graph.h"
#include "Graph/DCNP_Graph.h"
#include "ParallelFor.h"
#include "Population.h"
#include "ProblemData.h"
#include "crossover/reduceSolveCombine.h"
#include "search/L2NSSearch.h"
#include "search/DCNPSearch.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <type_traits>

namespace py = pybind11;

py::set solutionToPyset(const Solution &sol)
{
    py::set pyset;
    for (const auto &node : sol)
    {
        pyset.add(py::int_(node));
    }
    return pyset;
}

Solution pysetToSolution(const py::set &py_set)
{
    Solution sol;
    for (auto item : py_set)
    {
        sol.insert(item.cast<int>());
    }
    return sol;
}

PYBIND11_MODULE(_cndetector, m)
{
    m.doc() = R"doc(
        CNDetector - Python bindings for the parallel co-evolutionary memetic
        search for critical node detection problems.
    )doc";

    // ProblemData
    py::class_<ProblemData>(m, "ProblemData")
        .def(py::init<int>(), py::arg("num_nodes"))
        .def_static("read_from_adjacency_list_file",
                     &ProblemData::readFromAdjacencyListFile,
                     py::arg("filename"))
        .def_static("read_from_dimacs_edge_file",
                     &ProblemData::readFromDimacsEdgeFile,
                     py::arg("filename"))
        .def("create_original_graph",
             &ProblemData::createOriginalGraph,
             py::arg("budget"), py::arg("seed"),
             py::return_value_policy::take_ownership)
        .def("create_original_dcnp_graph",
             &ProblemData::createOriginalDCNPGraph,
             py::arg("budget"), py::arg("distance"), py::arg("seed"),
             py::return_value_policy::take_ownership)
        .def("add_node", &ProblemData::addNode, py::arg("node_id"))
        .def("add_edge", &ProblemData::addEdge, py::arg("u"), py::arg("v"))
        .def("get_nodes_set", &ProblemData::getNodesSet,
             py::return_value_policy::reference_internal)
        .def("get_adj_list", &ProblemData::getAdjList,
             py::return_value_policy::reference_internal)
        .def("num_nodes", &ProblemData::numNodes);

    // CNP_Graph
    py::class_<CNP_Graph, std::unique_ptr<CNP_Graph>>(m, "CNP_Graph")
        .def(py::init<>())
        .def("clone", &CNP_Graph::clone)
        .def("update_graph_by_removed_nodes", &CNP_Graph::updateGraphByRemovedNodes)
        .def("get_reduced_graph_by_removed_nodes", &CNP_Graph::getReducedGraphByRemovedNodes)
        .def("add_node", &CNP_Graph::addNode)
        .def("remove_node", &CNP_Graph::removeNode)
        .def("remove_best_nodes_from_set", &CNP_Graph::removeBestNodesFromSet,
             py::arg("candidate_nodes"))
        .def("is_node_removed", &CNP_Graph::isNodeRemoved)
        .def("get_num_nodes", &CNP_Graph::getNumNodes)
        .def("get_removed_nodes",
             [](const CNP_Graph &self) { return solutionToPyset(self.getRemovedNodes()); })
        .def("set_node_age", &CNP_Graph::setNodeAge)
        .def("get_objective_value", &CNP_Graph::getObjectiveValue)
        .def("get_random_full_budget_graph",
             &CNP_Graph::getRandomFullBudgetGraph,
             py::arg("seed"),
             py::return_value_policy::take_ownership)
        .def("get_random_partial_graph",
             &CNP_Graph::getRandomPartialGraph,
             py::arg("partial_budget"), py::arg("seed"),
             py::return_value_policy::take_ownership)
        .def("initialize_components_and_mapping",
             &CNP_Graph::initializeComponentsAndMapping);

    // DCNP_Graph
    py::class_<DCNP_Graph, std::unique_ptr<DCNP_Graph>>(m, "DCNP_Graph")
        .def(py::init<>())
        .def("clone", &DCNP_Graph::clone)
        .def("update_graph_by_removed_nodes", &DCNP_Graph::updateGraphByRemovedNodes)
        .def("get_reduced_graph_by_removed_nodes", &DCNP_Graph::getReducedGraphByRemovedNodes)
        .def("add_node", &DCNP_Graph::addNode)
        .def("remove_node", &DCNP_Graph::removeNode)
        .def("is_node_removed", &DCNP_Graph::isNodeRemoved)
        .def("get_num_nodes", &DCNP_Graph::getNumNodes)
        .def("get_removed_nodes",
             [](const DCNP_Graph &self) { return solutionToPyset(self.getRemovedNodes()); })
        .def("set_node_age", &DCNP_Graph::setNodeAge)
        .def("get_objective_value", &DCNP_Graph::getObjectiveValue)
        .def("get_random_full_budget_graph",
             &DCNP_Graph::getRandomFullBudgetGraph,
             py::arg("seed"),
             py::return_value_policy::take_ownership)
        .def("get_random_partial_graph",
             &DCNP_Graph::getRandomPartialGraph,
             py::arg("partial_budget"), py::arg("seed"),
             py::return_value_policy::take_ownership)
        .def("build_tree", &DCNP_Graph::buildTree)
        .def("calculate_khop_tree_size", &DCNP_Graph::calculateKhopTreeSize)
        .def("find_best_node_to_remove", &DCNP_Graph::findBestNodeToRemove)
        .def("find_best_node_to_add", &DCNP_Graph::findBestNodeToAdd)
        .def("random_select_node_to_remove", &DCNP_Graph::randomSelectNodeToRemove);

    // DualPopulation (CNP) and DCNPDualPopulation (DCNP) share the same
    // template implementation; register both with identical interfaces.
    auto registerDualPopulation = []<typename GraphT>(
        py::module_ &module, const char *name, std::type_identity<GraphT>)
    {
        using PopulationT = DualPopulationT<GraphT>;
        py::class_<PopulationT>(module, name)
            .def(py::init([](const GraphT &graph, int mainBudget,
                             int auxiliaryBudget, SolverConfig config) {
                     return new PopulationT(
                         graph, mainBudget, auxiliaryBudget,
                         std::move(config), std::chrono::steady_clock::now());
                 }),
                 py::arg("original_graph"),
                 py::arg("main_budget"),
                 py::arg("auxiliary_budget"),
                 py::arg("config"))
            .def("initialize",
                 [](PopulationT &self)
                 {
                     std::pair<Solution, int> result;
                     {
                         py::gil_scoped_release release;
                         result = self.initialize();
                     }
                     return py::make_tuple(solutionToPyset(result.first), result.second);
                 })
            .def("advance_one_generation",
                 [](PopulationT &self)
                 {
                     py::gil_scoped_release release;
                     self.advanceOneGeneration();
                 })
            .def("drain_iteration_events", &PopulationT::drainIterationEvents)
            .def("drain_hpc_events", &PopulationT::drainHPCEvents)
            .def("get_best_solution",
                 [](const PopulationT &self)
                 {
                     auto [solution, objValue] = self.getBestSolution();
                     return py::make_tuple(solutionToPyset(solution), objValue);
                 })
            .def("get_main_population",
                 [](const PopulationT &self)
                 {
                     py::list population;
                     for (const auto &[solution, objValue] : self.getMainPopulation())
                     {
                         population.append(py::make_tuple(solutionToPyset(solution), objValue));
                     }
                     return population;
                 })
            .def("get_main_population_size", &PopulationT::getMainPopulationSize)
            .def("get_main_generation_count", &PopulationT::getMainGenerationCount);
    };
    registerDualPopulation(m, "DualPopulation", std::type_identity<CNP_Graph>{});
    registerDualPopulation(m, "DCNPDualPopulation", std::type_identity<DCNP_Graph>{});

    // SolverConfig
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("population_size", &SolverConfig::populationSize)
        .def_readwrite("thread_count", &SolverConfig::threadCount)
        .def_readwrite("stagnation_threshold", &SolverConfig::stagnationThreshold)
        .def_readwrite("interaction_period", &SolverConfig::interactionPeriod)
        .def_readwrite("seed", &SolverConfig::seed)
        .def_readwrite("relaxation_coefficient", &SolverConfig::relaxationCoefficient)
        .def_readwrite("backbone_rate", &SolverConfig::backboneRate)
        .def_readwrite("display_interval", &SolverConfig::displayInterval)
        .def_readwrite("max_runtime", &SolverConfig::maxRuntime)
        .def_readwrite("search", &SolverConfig::search)
        .def_readwrite("l2ns", &SolverConfig::l2ns);

    // L2NSConfig
    py::class_<L2NSConfig>(m, "L2NSConfig")
        .def(py::init<>())
        .def_readwrite("allowable_idle_iterations", &L2NSConfig::allowableIdleIterations)
        .def_readwrite("min_destroy_size", &L2NSConfig::minDestroySize)
        .def_readwrite("max_destroy_size", &L2NSConfig::maxDestroySize)
        .def_readwrite("idle_iteration_cap", &L2NSConfig::idleIterationCap)
        .def_readwrite("idle_iteration_floor", &L2NSConfig::idleIterationFloor)
        .def_readwrite("impact_selection_rate", &L2NSConfig::impactSelectionRate)
        .def_readwrite("randomize_destroy_size", &L2NSConfig::randomizeDestroySize)
        .def_readwrite("adaptive_max_idle_iterations",
                       &L2NSConfig::adaptiveMaxIdleIterations)
        .def_readwrite("adaptive_min_destroy_size",
                       &L2NSConfig::adaptiveMinDestroySize)
        .def_readwrite("adaptive_max_destroy_size",
                       &L2NSConfig::adaptiveMaxDestroySize)
        .def_readwrite("adaptive_growth_interval",
                       &L2NSConfig::adaptiveGrowthInterval);

    // IterationEvent
    py::class_<IterationEvent>(m, "IterationEvent")
        .def(py::init<>())
        .def_readwrite("iteration", &IterationEvent::iteration)
        .def_readwrite("elapsed", &IterationEvent::elapsed)
        .def_readwrite("best_objective", &IterationEvent::bestObjective)
        .def_readwrite("population_size", &IterationEvent::populationSize);

    // HPCReport
    py::class_<HPCReport>(m, "HPCReport")
        .def(py::init<>())
        .def_readwrite("hpc_triggered", &HPCReport::hpcTriggered)
        .def_readwrite("candidate_obj_value", &HPCReport::candidateObjValue)
        .def_readwrite("improved_main_best", &HPCReport::improvedMainBest);

    // HPCEvent
    py::class_<HPCEvent>(m, "HPCEvent")
        .def(py::init<>())
        .def_readwrite("iteration", &HPCEvent::iteration)
        .def_readwrite("report", &HPCEvent::report);

    // LocalSearchResult
    py::class_<LocalSearchResult>(m, "LocalSearchResult")
        .def(py::init<>())
        .def_property_readonly("solution",
            [](const LocalSearchResult &r) { return solutionToPyset(r.solution); })
        .def_readwrite("obj_value", &LocalSearchResult::objValue);

    // Free functions
    m.def("run_l2ns",
          [](CNP_Graph &graph, int seed, const L2NSConfig &config)
          {
              py::gil_scoped_release release;
              return runL2NS(graph, seed, config);
          },
          py::arg("graph"), py::arg("seed"), py::arg("config"));

    m.def("run_dcnp_l2ns",
          [](DCNP_Graph &graph, int seed, const L2NSConfig &config)
          {
              py::gil_scoped_release release;
              return runDCNPL2NS(graph, seed, config);
          },
          py::arg("graph"), py::arg("seed"), py::arg("config"));

    m.def("resolve_l2ns_config", &resolveL2NSConfig, py::arg("config"));

    m.def("set_max_threads", &cndetector::setMaxThreads, py::arg("count"),
          "Cap the solver's worker threads (0 restores the hardware default). "
          "Results are identical for any thread count.");
    m.def("get_max_threads", &cndetector::maxThreads,
          "Effective worker-thread cap currently in use.");

    m.def("deterministic_seed", &deterministicSeed,
          py::arg("base_seed"), py::arg("stream_id"),
          py::arg("iteration") = 0, py::arg("slot") = 0);
}
