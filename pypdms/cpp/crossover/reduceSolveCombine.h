#ifndef REDUCE_SOLVE_COMBINE_H
#define REDUCE_SOLVE_COMBINE_H

#include "../Graph/Types.h"
#include "../search/L2NSSearch.h"
#include <memory>
#include <optional>

// Defaults follow the irace-tuned parameter settings of Table 1:
// theta = 10, beta = 20, alpha = 0.05 (partialRatio = 1 - alpha),
// xi = 1000 (L2NSConfig::randomIdleProduct), delta = 500, kappa = 2.
struct SolverConfig
{
    // theta: size of each of the two populations.
    int populationSize = 10;
    // kappa: thread count. Algorithm 2 runs one offspring per thread, so this
    // is also the number of offspring generated per population per generation.
    int threadCount = 2;
    // beta: generations between two heterogeneous population cooperations.
    int transferInterval = 20;
    int seed = 0;
    // 1 - alpha, so the auxiliary population carries floor(k * partialRatio)
    // nodes per solution.
    double partialRatio = 0.95;
    // delta: consecutive non-improving generations that trigger the
    // reconstruction of the auxiliary population (Algorithm 4, lines 8-11).
    int stagnationThreshold = 500;
    double beta = 0.9;
    double displayInterval = 60.0;
    // Hard wall-clock budget in seconds. <= 0 means "no limit".
    double maxRuntime = 0.0;
    std::string search = "L2NS";
    L2NSConfig l2ns;
    L2NSConfig resolvedL2ns;
};

uint64_t splitmix64(uint64_t value);

int deterministicSeed(
    int baseSeed, uint64_t streamId, uint64_t iteration = 0, uint64_t slot = 0);

L2NSConfig resolveL2NSConfig(const SolverConfig &config);

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
