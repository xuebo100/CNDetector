#ifndef CLNS_SEARCH_H
#define CLNS_SEARCH_H

#include "../Graph/CNP_Graph.h"
#include "DestroyRepairSearch.h"

// Budget of one CLNS run; see runDestroyRepairSearch for how the fields drive
// the loop.
struct CLNSConfig
{
    // gamma: allowable number of idle (non-improving) iterations of one CLNS
    // run, before it is divided by the destroy size lambda.
    int allowableIdleIterations = 1000;
    // Each run draws its destroy size lambda uniformly from
    // [minDestroySize, maxDestroySize].
    int minDestroySize = 1;
    int maxDestroySize = 50;
    // The run stops after min(idleIterationCap, gamma / lambda) consecutive
    // non-improving iterations; idleIterationFloor keeps that limit at 1 or
    // above when gamma is smaller than lambda.
    int idleIterationCap = 500;
    int idleIterationFloor = 1;
    // Probability that the destroy operator picks the least-impact node of the
    // chosen large connected component rather than its oldest node. Not one of
    // the tuned solver parameters; it only steers the choice of victim node.
    double impactSelectionRate = 0.3;
};

// Destroy operator: remove destroySize nodes from the large connected
// components of the residual graph (CNP_Graph::selectRemovedComponent).
// Returns the number of nodes removed.
int lccDrivenDestroy(
    CNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    double impactSelectionRate, long step);

// Repair operator: reinsert numNodes nodes one at a time, each time picking
// the node whose reinsertion degrades the objective the least.
void greedyRepair(CNP_Graph &graph, int numNodes, long step);

// Component-oriented large neighborhood search: the destroy-and-repair loop with
// the two operators above. The incumbent solution is carried by the graph's
// removed-node set.
LocalSearchResult runCLNS(
    CNP_Graph &graph, int seed, const CLNSConfig &clns,
    Deadline deadline = noDeadline());

#endif // CLNS_SEARCH_H
