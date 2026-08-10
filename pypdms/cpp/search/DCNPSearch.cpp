#include "DCNPSearch.h"

LocalSearchResult runDCNPCHNS(
    DCNP_Graph &graph, int seed, const CHNSConfig &chns, Deadline deadline)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    Solution bestSolution = graph.getRemovedNodes();
    int currentObjValue = graph.getObjectiveValue();
    int bestObjValue = currentObjValue;
    long numSteps = 0;
    long numIdleSteps = 0;

    int runMaxIdleSteps = chns.maxIdleSteps;
    int fixedBatchSize = chns.minBatchSize;

    if (chns.randomizeBatchAndIdle)
    {
        const int batchRange = chns.randomBatchMax - chns.randomBatchMin + 1;
        fixedBatchSize = chns.randomBatchMin + rng.generateIndex(batchRange);
        runMaxIdleSteps = std::clamp(
            chns.randomIdleProduct / fixedBatchSize,
            chns.randomMinIdleSteps,
            chns.randomMaxIdleSteps);
    }

    while (numIdleSteps < runMaxIdleSteps)
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            break;
        }

        ++numSteps;

        int batchSize = fixedBatchSize;
        if (!chns.randomizeBatchAndIdle
            && chns.minBatchSize != chns.maxBatchSize)
        {
            const int batchLevel
                = static_cast<int>(numIdleSteps / chns.batchIdleThreshold);
            batchSize = std::min(
                chns.maxBatchSize, chns.minBatchSize + 2 * batchLevel);
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
            const Node nodeToRemove = rng.generateProbability() < chns.theta
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
