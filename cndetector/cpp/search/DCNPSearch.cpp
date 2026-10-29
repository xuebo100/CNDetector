#include "DCNPSearch.h"

int dcnpDestroy(
    DCNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    double impactSelectionRate, long step)
{
    int numRemoved = 0;
    for (int i = 0; i < destroySize; ++i)
    {
        const Node nodeToRemove
            = rng.generateProbability() < impactSelectionRate
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

LocalSearchResult runDCNPCLNS(
    DCNP_Graph &graph, int seed, const CLNSConfig &clns, Deadline deadline)
{
    return runDestroyRepairSearch(
        graph, seed, clns, deadline,
        [&clns](DCNP_Graph &g, int destroySize, RandomNumberGenerator &rng, long step)
        { return dcnpDestroy(g, destroySize, rng, clns.impactSelectionRate, step); },
        [](DCNP_Graph &g, int numNodes, long step) { dcnpGreedyRepair(g, numNodes, step); });
}
