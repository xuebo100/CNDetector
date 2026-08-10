#ifndef REDUCE_SOLVE_COMBINE_H
#define REDUCE_SOLVE_COMBINE_H

#include "../Graph/Types.h"
#include "../search/CHNSSearch.h"
#include <memory>
#include <optional>

struct SolverConfig
{
    int populationSize = 6;
    int offspringCount = 1;
    int transferInterval = 50;
    int seed = 0;
    double partialRatio = 0.95;
    double beta = 0.9;
    double displayInterval = 60.0;
    // Hard wall-clock budget in seconds. <= 0 means "no limit".
    double maxRuntime = 0.0;
    std::string search = "CHNS";
    CHNSConfig chns;
    CHNSConfig resolvedChns;
};

uint64_t splitmix64(uint64_t value);

int deterministicSeed(
    int baseSeed, uint64_t streamId, uint64_t iteration = 0, uint64_t slot = 0);

CHNSConfig resolveCHNSConfig(const SolverConfig &config);

// Reduce-Solve-Combine crossover, generic over the graph type (CNP_Graph or
// DCNP_Graph). Explicitly instantiated in reduceSolveCombine.cpp.
template <typename GraphT>
std::unique_ptr<GraphT> reduceSolveCombine(
    const GraphT &originalGraph,
    const std::pair<const Solution *, const Solution *> &parents,
    double beta,
    std::optional<int> targetBudget,
    int seed,
    const SolverConfig &config,
    Deadline deadline = noDeadline());

#endif // REDUCE_SOLVE_COMBINE_H
