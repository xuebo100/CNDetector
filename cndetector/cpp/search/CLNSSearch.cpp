#include "CLNSSearch.h"

int lccDrivenDestroy(
    CNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    double impactSelectionRate, long step)
{
    int numRemoved = 0;
    for (int i = 0; i < destroySize; ++i)
    {
        const auto componentToRemove = graph.selectRemovedComponent();
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

void greedyRepair(CNP_Graph &graph, int numNodes, long step)
{
    for (int i = 0; i < numNodes; ++i)
    {
        const Node nodeToAdd = graph.greedySelectNodeToAdd();
        if (nodeToAdd == INVALID_NODE)
        {
            break;
        }
        graph.addNode(nodeToAdd);
        graph.setNodeAge(nodeToAdd, step);
    }
}

LocalSearchResult runCLNS(
    CNP_Graph &graph, int seed, const CLNSConfig &clns, Deadline deadline)
{
    return runDestroyRepairSearch(
        graph, seed, clns, deadline,
        [&clns](CNP_Graph &g, int destroySize, RandomNumberGenerator &rng, long step)
        { return lccDrivenDestroy(g, destroySize, rng, clns.impactSelectionRate, step); },
        [](CNP_Graph &g, int numNodes, long step) { greedyRepair(g, numNodes, step); });
}
