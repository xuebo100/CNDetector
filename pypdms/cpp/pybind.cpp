#include "Graph/CNP_Graph.h"
#include "Graph/DCNP_Graph.h"
#include "Population.h"
#include "ProblemData.h"
#include "crossover/reduceSolveCombine.h"
#include "search/CHNSSearch.h"
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

PYBIND11_MODULE(_pypdms, m)
{
    m.doc() = R"doc(
        PyPDMS - Python bindings for Population-based Dual Memetic Search
        for Critical Node Problems.
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
        .def("get_random_feasible_graph",
             &CNP_Graph::getRandomFeasibleGraph,
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
        .def("get_random_feasible_graph",
             &DCNP_Graph::getRandomFeasibleGraph,
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
            .def(py::init([](const GraphT &graph, int feasibleBudget,
                             int infeasibleBudget, SolverConfig config) {
                     return new PopulationT(
                         graph, feasibleBudget, infeasibleBudget,
                         std::move(config), std::chrono::steady_clock::now());
                 }),
                 py::arg("original_graph"),
                 py::arg("feasible_budget"),
                 py::arg("infeasible_budget"),
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
            .def("drain_exchange_events", &PopulationT::drainExchangeEvents)
            .def("get_best_feasible_solution",
                 [](const PopulationT &self)
                 {
                     auto [solution, objValue] = self.getBestFeasibleSolution();
                     return py::make_tuple(solutionToPyset(solution), objValue);
                 })
            .def("get_feasible_population",
                 [](const PopulationT &self)
                 {
                     py::list population;
                     for (const auto &[solution, objValue] : self.getFeasiblePopulation())
                     {
                         population.append(py::make_tuple(solutionToPyset(solution), objValue));
                     }
                     return population;
                 })
            .def("get_feasible_population_size", &PopulationT::getFeasiblePopulationSize)
            .def("get_feasible_iteration_count", &PopulationT::getFeasibleIterationCount);
    };
    registerDualPopulation(m, "DualPopulation", std::type_identity<CNP_Graph>{});
    registerDualPopulation(m, "DCNPDualPopulation", std::type_identity<DCNP_Graph>{});

    // SolverConfig
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("population_size", &SolverConfig::populationSize)
        .def_readwrite("offspring_count", &SolverConfig::offspringCount)
        .def_readwrite("transfer_interval", &SolverConfig::transferInterval)
        .def_readwrite("seed", &SolverConfig::seed)
        .def_readwrite("partial_ratio", &SolverConfig::partialRatio)
        .def_readwrite("beta", &SolverConfig::beta)
        .def_readwrite("display_interval", &SolverConfig::displayInterval)
        .def_readwrite("max_runtime", &SolverConfig::maxRuntime)
        .def_readwrite("search", &SolverConfig::search)
        .def_readwrite("chns", &SolverConfig::chns);

    // CHNSConfig
    py::class_<CHNSConfig>(m, "CHNSConfig")
        .def(py::init<>())
        .def_readwrite("max_idle_steps", &CHNSConfig::maxIdleSteps)
        .def_readwrite("theta", &CHNSConfig::theta)
        .def_readwrite("min_batch_size", &CHNSConfig::minBatchSize)
        .def_readwrite("max_batch_size", &CHNSConfig::maxBatchSize)
        .def_readwrite("batch_idle_threshold", &CHNSConfig::batchIdleThreshold)
        .def_readwrite("randomize_batch_and_idle", &CHNSConfig::randomizeBatchAndIdle)
        .def_readwrite("random_batch_min", &CHNSConfig::randomBatchMin)
        .def_readwrite("random_batch_max", &CHNSConfig::randomBatchMax)
        .def_readwrite("random_idle_product", &CHNSConfig::randomIdleProduct)
        .def_readwrite("random_min_idle_steps", &CHNSConfig::randomMinIdleSteps)
        .def_readwrite("random_max_idle_steps", &CHNSConfig::randomMaxIdleSteps);

    // IterationEvent
    py::class_<IterationEvent>(m, "IterationEvent")
        .def(py::init<>())
        .def_readwrite("iteration", &IterationEvent::iteration)
        .def_readwrite("elapsed", &IterationEvent::elapsed)
        .def_readwrite("best_objective", &IterationEvent::bestObjective)
        .def_readwrite("population_size", &IterationEvent::populationSize);

    // ExchangeReport
    py::class_<ExchangeReport>(m, "ExchangeReport")
        .def(py::init<>())
        .def_readwrite("exchange_triggered", &ExchangeReport::exchangeTriggered)
        .def_readwrite("first_population_candidate_obj", &ExchangeReport::firstPopulationCandidateObj)
        .def_readwrite("first_population_improved_best", &ExchangeReport::firstPopulationImprovedBest);

    // ExchangeEvent
    py::class_<ExchangeEvent>(m, "ExchangeEvent")
        .def(py::init<>())
        .def_readwrite("iteration", &ExchangeEvent::iteration)
        .def_readwrite("report", &ExchangeEvent::report);

    // LocalSearchResult
    py::class_<LocalSearchResult>(m, "LocalSearchResult")
        .def(py::init<>())
        .def_property_readonly("solution",
            [](const LocalSearchResult &r) { return solutionToPyset(r.solution); })
        .def_readwrite("obj_value", &LocalSearchResult::objValue);

    // Free functions
    m.def("run_chns",
          [](CNP_Graph &graph, int seed, const CHNSConfig &config)
          {
              py::gil_scoped_release release;
              return runCHNS(graph, seed, config);
          },
          py::arg("graph"), py::arg("seed"), py::arg("config"));

    m.def("run_dcnp_chns",
          [](DCNP_Graph &graph, int seed, const CHNSConfig &config)
          {
              py::gil_scoped_release release;
              return runDCNPCHNS(graph, seed, config);
          },
          py::arg("graph"), py::arg("seed"), py::arg("config"));

    m.def("resolve_chns_config", &resolveCHNSConfig, py::arg("config"));

    m.def("deterministic_seed", &deterministicSeed,
          py::arg("base_seed"), py::arg("stream_id"),
          py::arg("iteration") = 0, py::arg("slot") = 0);
}
