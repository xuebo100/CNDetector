#ifndef DESTROY_REPAIR_SEARCH_H
#define DESTROY_REPAIR_SEARCH_H

#include "../Graph/Types.h"
#include "../RandomNumberGenerator.h"

#include <algorithm>
#include <chrono>
#include <limits>

struct LocalSearchResult
{
    Solution solution;
    int objValue = std::numeric_limits<int>::max();
};

// A hard wall-clock deadline. The search returns the best solution found so
// far once the deadline is reached. The default (time_point::max()) means
// "no deadline" so existing callers keep running to natural completion.
using Deadline = std::chrono::steady_clock::time_point;
inline Deadline noDeadline() { return Deadline::max(); }

// The destroy-and-repair loop shared by every local search. A run draws its
// destroy size uniformly from [config.minDestroySize, config.maxDestroySize],
// then alternates destroy(graph, destroySize, rng, step) and
// repair(graph, numRemoved, step) until
// clamp(config.allowableIdleIterations / destroySize,
//       config.idleIterationFloor, config.idleIterationCap)
// consecutive iterations bring no improvement: a larger destroy size makes
// each iteration more expensive, so fewer of them are allowed. The incumbent
// solution is carried by the graph's removed-node set, and the best solution
// seen is returned.
template <typename GraphT, typename Config, typename Destroy, typename Repair>
LocalSearchResult runDestroyRepairSearch(
    GraphT &graph, int seed, const Config &config, Deadline deadline,
    Destroy &&destroy, Repair &&repair)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    Solution bestSolution = graph.getRemovedNodes();
    int bestObjValue = graph.getObjectiveValue();

    const int destroySizeRange = config.maxDestroySize - config.minDestroySize + 1;
    const int destroySize = config.minDestroySize + rng.generateIndex(destroySizeRange);
    const int idleLimit = std::clamp(
        config.allowableIdleIterations / destroySize,
        config.idleIterationFloor,
        config.idleIterationCap);

    long numSteps = 0;
    long idleIterations = 0;
    while (idleIterations < idleLimit)
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            break;
        }

        ++numSteps;

        const int activeNodes = graph.getNumNodes()
            - static_cast<int>(graph.getRemovedNodes().size());
        const int stepDestroySize = std::min(destroySize, std::max(0, activeNodes - 1));
        if (stepDestroySize <= 0)
        {
            ++idleIterations;
            continue;
        }

        const int numRemoved = destroy(graph, stepDestroySize, rng, numSteps);
        repair(graph, numRemoved, numSteps);

        const int objValue = graph.getObjectiveValue();
        if (objValue < bestObjValue)
        {
            bestSolution = graph.getRemovedNodes();
            bestObjValue = objValue;
            idleIterations = 0;
        }
        else
        {
            ++idleIterations;
        }
    }

    return {bestSolution, bestObjValue};
}

#endif // DESTROY_REPAIR_SEARCH_H
