#ifndef CHNS_SEARCH_H
#define CHNS_SEARCH_H

#include "CLNSSearch.h"

// Budget of one run of the component-based hybrid neighborhood search (CHNS),
// the local search that CLNS replaces; it is kept so the two can be compared.
// CHNS runs the same destroy-and-repair loop as CLNS with a larger idle budget,
// and picks each component to destroy with
// CNP_Graph::selectRemovedComponentCHNS instead of the large-connected-component
// rule of CLNS.
struct CHNSConfig
{
    int allowableIdleIterations = 2000;
    int minDestroySize = 1;
    int maxDestroySize = 50;
    int idleIterationCap = 1000;
    int idleIterationFloor = 40;
    double impactSelectionRate = 0.3;
};

// Destroy operator of CHNS: like lccDrivenDestroy, but each component is drawn
// by CNP_Graph::selectRemovedComponentCHNS.
int chnsDestroy(
    CNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    double impactSelectionRate, long step);

// Component-based hybrid neighborhood search: the destroy-and-repair loop with
// chnsDestroy and the greedy repair of CLNS. CNP only.
LocalSearchResult runCHNS(
    CNP_Graph &graph, int seed, const CHNSConfig &chns,
    Deadline deadline = noDeadline());

#endif // CHNS_SEARCH_H
