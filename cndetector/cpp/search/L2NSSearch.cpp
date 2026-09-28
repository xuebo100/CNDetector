#include "L2NSSearch.h"

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

LocalSearchResult runL2NS(
    CNP_Graph &graph, int seed, const L2NSConfig &l2ns, Deadline deadline)
{
    return runDestroyRepairSearch(
        graph, seed, l2ns, deadline,
        [&l2ns](CNP_Graph &g, int destroySize, RandomNumberGenerator &rng, long step)
        { return lccDrivenDestroy(g, destroySize, rng, l2ns.impactSelectionRate, step); },
        [](CNP_Graph &g, int numNodes, long step) { greedyRepair(g, numNodes, step); });
}
