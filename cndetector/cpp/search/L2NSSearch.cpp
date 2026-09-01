#include "L2NSSearch.h"

LocalSearchResult runL2NS(
    CNP_Graph &graph, int seed, const L2NSConfig &l2ns, Deadline deadline)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    CNP_Graph &currentGraph = graph;
    Solution bestSolution = currentGraph.getRemovedNodes();
    int currentObjValue = currentGraph.getObjectiveValue();
    int bestObjValue = currentObjValue;
    long numSteps = 0;
    long numIdleSteps = 0;

    int runMaxIdleSteps = l2ns.maxIdleSteps;
    int fixedBatchSize = l2ns.minBatchSize;

    if (l2ns.randomizeBatchAndIdle)
    {
        // Algorithm 3, lines 2-6: draw the destroy size lambda uniformly from
        // [1, 50], then cap the idle-iteration budget of this run at
        // xi' = min(500, xi / lambda), where xi is the allowable idle
        // iteration count of Table 1.
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
                = rng.generateProbability() < l2ns.theta
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
