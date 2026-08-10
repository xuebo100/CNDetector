#include "CHNSSearch.h"

LocalSearchResult runCHNS(
    CNP_Graph &graph, int seed, const CHNSConfig &chns, Deadline deadline)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    CNP_Graph &currentGraph = graph;
    Solution bestSolution = currentGraph.getRemovedNodes();
    int currentObjValue = currentGraph.getObjectiveValue();
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

        const int activeNodes = currentGraph.getNumNodes()
            - static_cast<int>(currentGraph.getRemovedNodes().size());
        batchSize = std::min(batchSize, std::max(0, activeNodes - 1));
        if (batchSize <= 0)
        {
            ++numIdleSteps;
            continue;
        }

        int removedCount = 0;
        for (int i = 0; i < batchSize; ++i)
        {
            const auto componentToRemove = currentGraph.selectRemovedComponent();
            const Node nodeToRemove
                = rng.generateProbability() < chns.theta
                    ? currentGraph.impactSelectNodeFromComponent(componentToRemove)
                    : currentGraph.ageSelectNodeFromComponent(componentToRemove);
            currentGraph.removeNode(nodeToRemove);
            currentGraph.setNodeAge(nodeToRemove, numSteps);
            ++removedCount;
        }

        for (int i = 0; i < removedCount; ++i)
        {
            const Node nodeToAdd = currentGraph.greedySelectNodeToAdd();
            if (nodeToAdd != INVALID_NODE)
            {
                currentGraph.addNode(nodeToAdd);
                currentGraph.setNodeAge(nodeToAdd, numSteps);
            }
        }

        currentObjValue = currentGraph.getObjectiveValue();
        if (currentObjValue < bestObjValue)
        {
            bestSolution = currentGraph.getRemovedNodes();
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
