#include "DCNPSearch.h"

int dcnpDestroy(
    DCNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    const L2NSConfig &l2ns, long step)
{
    int numRemoved = 0;
    for (int i = 0; i < destroySize; ++i)
    {
        const Node nodeToRemove
            = rng.generateProbability() < l2ns.impactSelectionRate
                ? graph.findBestNodeToRemove()
                : graph.randomSelectNodeToRemove();
        if (nodeToRemove == INVALID_NODE)
        {
            break;
        }
        graph.removeNode(nodeToRemove);
        graph.setNodeAge(nodeToRemove, step);
        ++numRemoved;
    }
    return numRemoved;
}

void dcnpGreedyRepair(DCNP_Graph &graph, int numNodes, long step)
{
    for (int i = 0; i < numNodes; ++i)
    {
        const Node nodeToAdd = graph.findBestNodeToAdd();
        if (nodeToAdd == INVALID_NODE)
        {
            continue;
        }
        graph.addNode(nodeToAdd);
        graph.setNodeAge(nodeToAdd, step);
    }
}

LocalSearchResult runDCNPL2NS(
    DCNP_Graph &graph, int seed, const L2NSConfig &l2ns, Deadline deadline)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    Solution bestSolution = graph.getRemovedNodes();
    int currentObjValue = graph.getObjectiveValue();
    int bestObjValue = currentObjValue;
    long numSteps = 0;
    // Consecutive non-improving iterations, and the limit at which this run
    // gives up.
    long idleIterations = 0;
    int idleLimit = l2ns.adaptiveMaxIdleIterations;
    // Destroy size: how many nodes one destroy step removes.
    int destroySize = l2ns.adaptiveMinDestroySize;

    if (l2ns.randomizeDestroySize)
    {
        // Draw the destroy size at random, then divide the idle budget by it.
        // For DCNP both ranges are much tighter than for CNP, because a single
        // objective evaluation is far more expensive.
        const int destroySizeRange
            = l2ns.maxDestroySize - l2ns.minDestroySize + 1;
        destroySize = l2ns.minDestroySize + rng.generateIndex(destroySizeRange);
        idleLimit = std::clamp(
            l2ns.allowableIdleIterations / destroySize,
            l2ns.idleIterationFloor,
            l2ns.idleIterationCap);
    }

    while (idleIterations < idleLimit)
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            break;
        }

        ++numSteps;

        int stepDestroySize = destroySize;
        if (!l2ns.randomizeDestroySize
            && l2ns.adaptiveMinDestroySize != l2ns.adaptiveMaxDestroySize)
        {
            const int growthLevel
                = static_cast<int>(idleIterations / l2ns.adaptiveGrowthInterval);
            stepDestroySize = std::min(
                l2ns.adaptiveMaxDestroySize,
                l2ns.adaptiveMinDestroySize + 2 * growthLevel);
        }

        const int activeNodes = graph.getNumNodes()
            - static_cast<int>(graph.getRemovedNodes().size());
        stepDestroySize = std::min(stepDestroySize, std::max(0, activeNodes - 1));
        if (stepDestroySize <= 0)
        {
            ++idleIterations;
            continue;
        }

        // Destroy, then repair.
        const int numRemoved
            = dcnpDestroy(graph, stepDestroySize, rng, l2ns, numSteps);
        dcnpGreedyRepair(graph, numRemoved, numSteps);

        // Track the best solution seen and reset the idle counter whenever
        // it improves.
        currentObjValue = graph.getObjectiveValue();
        if (currentObjValue < bestObjValue)
        {
            bestSolution = graph.getRemovedNodes();
            bestObjValue = currentObjValue;
            idleIterations = 0;
        }
        else
        {
            ++idleIterations;
        }
    }

    return {bestSolution, bestObjValue};
}
