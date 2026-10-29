#include "CHNSSearch.h"

int chnsDestroy(
    CNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    double impactSelectionRate, long step)
{
    int numRemoved = 0;
    for (int i = 0; i < destroySize; ++i)
    {
        const auto componentToRemove = graph.selectRemovedComponentCHNS();
        const Node nodeToRemove
            = rng.generateProbability() < impactSelectionRate
                ? graph.impactSelectNodeFromComponent(componentToRemove)
                : graph.ageSelectNodeFromComponent(componentToRemove);
        graph.removeNode(nodeToRemove);
        graph.setNodeAge(nodeToRemove, step);
        ++numRemoved;
    }
    return numRemoved;
}

LocalSearchResult runCHNS(
    CNP_Graph &graph, int seed, const CHNSConfig &chns, Deadline deadline)
{
    return runDestroyRepairSearch(
        graph, seed, chns, deadline,
        [&chns](CNP_Graph &g, int destroySize, RandomNumberGenerator &rng, long step)
        { return chnsDestroy(g, destroySize, rng, chns.impactSelectionRate, step); },
        [](CNP_Graph &g, int numNodes, long step) { greedyRepair(g, numNodes, step); });
}
