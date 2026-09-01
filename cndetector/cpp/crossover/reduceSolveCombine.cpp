#include "reduceSolveCombine.h"
#include "../Graph/DCNP_Graph.h"
#include "../RandomNumberGenerator.h"
#include "../search/LocalSearch.h"
#include <stdexcept>

uint64_t splitmix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int deterministicSeed(
    int baseSeed, uint64_t streamId, uint64_t iteration, uint64_t slot)
{
    uint64_t value = splitmix64(static_cast<uint32_t>(baseSeed));
    value ^= splitmix64(streamId);
    value ^= splitmix64(iteration);
    value ^= splitmix64(slot);

    const int seed = static_cast<int>(value & 0x7fffffff);
    return seed == 0 ? 1 : seed;
}

L2NSConfig resolveL2NSConfig(const SolverConfig &config)
{
    L2NSConfig resolved = config.l2ns;

    if (config.search == "L2NS")
    {
        resolved.randomizeBatchAndIdle = true;
        return resolved;
    }

    if (config.search == "L2NS-ADAPT")
    {
        resolved.randomizeBatchAndIdle = false;
        return resolved;
    }

    if (config.search.rfind("L2NS", 0) == 0 && config.search.size() > 4)
    {
        const int fixedBatchSize = std::stoi(config.search.substr(4));
        if (fixedBatchSize <= 0)
        {
            throw std::runtime_error("L2NS fixed batch size must be positive");
        }
        resolved.randomizeBatchAndIdle = false;
        resolved.minBatchSize = fixedBatchSize;
        resolved.maxBatchSize = fixedBatchSize;
        return resolved;
    }

    throw std::runtime_error(
        "Unsupported search strategy: " + config.search);
}

template <typename GraphT>
std::unique_ptr<GraphT> reduceSolveCombine(
    const GraphT &originalGraph,
    const std::pair<const Solution *, const Solution *> &parents,
    double beta,
    std::optional<int> targetBudget,
    int seed,
    const SolverConfig &config,
    Deadline deadline)
{
    if (beta < 0.0 || beta > 1.0)
    {
        throw std::invalid_argument("beta for RSC crossover must be in [0, 1]");
    }

    RandomNumberGenerator rng;
    rng.setSeed(seed);

    const auto &lhs = *parents.first;
    const auto &rhs = *parents.second;

    Solution nodesToRemove;
    nodesToRemove.reserve(std::min(lhs.size(), rhs.size()));

    const int maxCommonToRemove = targetBudget.has_value()
        ? std::max(0, *targetBudget - 1)
        : static_cast<int>(std::min(lhs.size(), rhs.size()));

    for (Node node : lhs)
    {
        if (static_cast<int>(nodesToRemove.size()) >= maxCommonToRemove)
        {
            break;
        }
        if (rhs.contains(node) && rng.generateProbability() < beta)
        {
            nodesToRemove.insert(node);
        }
    }

    auto workingGraph = originalGraph.clone();
    workingGraph->getReducedGraphByRemovedNodes(nodesToRemove);

    std::unique_ptr<GraphT> reducedGraph;
    if (targetBudget.has_value())
    {
        const int remainingBudget
            = *targetBudget - static_cast<int>(nodesToRemove.size());
        if (remainingBudget <= 0)
        {
            reducedGraph = std::move(workingGraph);
        }
        else
        {
            reducedGraph = workingGraph->getRandomPartialGraph(
                remainingBudget, deterministicSeed(seed, 0x5001u));
        }
    }
    else
    {
        reducedGraph = workingGraph->getRandomFeasibleGraph(
            deterministicSeed(seed, 0x5002u));
    }

    const LocalSearchResult result
        = runLocalSearch(*reducedGraph, deterministicSeed(seed, 0x5003u),
                         config.resolvedL2ns, deadline);

    Solution finalNodes = nodesToRemove;
    finalNodes.insert(result.solution.begin(), result.solution.end());

    auto improvedGraph = originalGraph.clone();
    improvedGraph->updateGraphByRemovedNodes(finalNodes);
    return improvedGraph;
}

template std::unique_ptr<CNP_Graph> reduceSolveCombine<CNP_Graph>(
    const CNP_Graph &,
    const std::pair<const Solution *, const Solution *> &,
    double, std::optional<int>, int, const SolverConfig &, Deadline);

template std::unique_ptr<DCNP_Graph> reduceSolveCombine<DCNP_Graph>(
    const DCNP_Graph &,
    const std::pair<const Solution *, const Solution *> &,
    double, std::optional<int>, int, const SolverConfig &, Deadline);
