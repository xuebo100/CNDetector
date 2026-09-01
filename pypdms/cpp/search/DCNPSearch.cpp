#include "DCNPSearch.h"

LocalSearchResult runDCNPL2NS(
    DCNP_Graph &graph, int seed, const L2NSConfig &l2ns, Deadline deadline)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    Solution bestSolution = graph.getRemovedNodes();
    int currentObjValue = graph.getObjectiveValue();
    int bestObjValue = currentObjValue;
    long numSteps = 0;
    long numIdleSteps = 0;

    int runMaxIdleSteps = l2ns.maxIdleSteps;
    int fixedBatchSize = l2ns.minBatchSize;

    if (l2ns.randomizeBatchAndIdle)
    {
        const int batchRange = l2ns.randomBatchMax - l2ns.randomBatchMin + 1;
        fixedBatchSize = l2ns.randomBatchMin + rng.generateIndex(batchRange);
        runMaxIdleSteps = std::clamp(
            l2ns.randomIdleProduct / fixedBatchSize,
            l2ns.randomMinIdleSteps,
            l2ns.randomMaxIdleSteps);
    }

    while (numIdleSteps < runMaxIdleSteps)
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            break;
        }

        ++numSteps;

        int batchSize = fixedBatchSize;
        if (!l2ns.randomizeBatchAndIdle
            && l2ns.minBatchSize != l2ns.maxBatchSize)
        {
            const int batchLevel
                = static_cast<int>(numIdleSteps / l2ns.batchIdleThreshold);
            batchSize = std::min(
                l2ns.maxBatchSize, l2ns.minBatchSize + 2 * batchLevel);
        }

        const int activeNodes = graph.getNumNodes()
            - static_cast<int>(graph.getRemovedNodes().size());
        batchSize = std::min(batchSize, std::max(0, activeNodes - 1));
        if (batchSize <= 0)
        {
            ++numIdleSteps;
            continue;
        }

        int removedCount = 0;
        for (int i = 0; i < batchSize; ++i)
        {
            const Node nodeToRemove = rng.generateProbability() < l2ns.theta
                ? graph.findBestNodeToRemove()
                : graph.randomSelectNodeToRemove();
            if (nodeToRemove == INVALID_NODE)
            {
                break;
            }
            graph.removeNode(nodeToRemove);
            graph.setNodeAge(nodeToRemove, numSteps);
            ++removedCount;
        }

        for (int i = 0; i < removedCount; ++i)
        {
            const Node nodeToAdd = graph.findBestNodeToAdd();
            if (nodeToAdd != INVALID_NODE)
            {
                graph.addNode(nodeToAdd);
                graph.setNodeAge(nodeToAdd, numSteps);
            }
        }

        currentObjValue = graph.getObjectiveValue();
        if (currentObjValue < bestObjValue)
        {
            bestSolution = graph.getRemovedNodes();
            bestObjValue = currentObjValue;
            numIdleSteps = 0;
        }
        else
        {
            ++numIdleSteps;
        }
    }

    return {bestSolution, bestObjValue};
}
