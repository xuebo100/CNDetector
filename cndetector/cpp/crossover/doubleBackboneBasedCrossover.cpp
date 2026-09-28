#include "doubleBackboneBasedCrossover.h"

#include "../RandomNumberGenerator.h"
#include "reduceSolveCombine.h"

std::unique_ptr<CNP_Graph> doubleBackboneBasedCrossover(
    const CNP_Graph &originalGraph,
    const std::pair<const Solution *, const Solution *> &parents,
    int targetSize,
    int seed)
{
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    const Solution &first = *parents.first;
    const Solution &second = *parents.second;

    Solution inherited;
    inherited.reserve(first.size() + second.size());
    for (Node node : first)
    {
        // The common backbone is always inherited; an exclusive node only with
        // probability DBX_EXCLUSIVE_INHERIT_RATE.
        if (second.contains(node)
            || rng.generateProbability() < DBX_EXCLUSIVE_INHERIT_RATE)
        {
            inherited.insert(node);
        }
    }
    for (Node node : second)
    {
        if (inherited.contains(node))
        {
            continue;
        }
        if (rng.generateProbability() < DBX_EXCLUSIVE_INHERIT_RATE)
        {
            inherited.insert(node);
        }
    }

    auto offspring = originalGraph.clone();
    offspring->reseed(deterministicSeed(seed, 0x7001u));
    offspring->updateGraphByRemovedNodes(inherited);

    const int inheritedCount = static_cast<int>(inherited.size());
    for (int i = inheritedCount; i < targetSize; ++i)
    {
        const ComponentIndex componentIndex = offspring->selectRemovedComponentCHNS();
        offspring->removeNode(offspring->randomSelectNodeFromComponent(componentIndex));
    }
    for (int i = targetSize; i < inheritedCount; ++i)
    {
        offspring->addNode(offspring->greedySelectNodeToAdd());
    }

    return offspring;
}
