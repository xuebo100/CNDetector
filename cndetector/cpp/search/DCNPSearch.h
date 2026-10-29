#ifndef DCNP_SEARCH_H
#define DCNP_SEARCH_H

#include "../Graph/DCNP_Graph.h"
#include "CLNSSearch.h"

// DCNP counterparts of the two CLNS operators. The residual graph of the DCNP
// has no meaningful connected components, so instead of drawing from a large
// connected component the destroy operator removes, with probability
// impactSelectionRate, the node whose removal cuts the most b-hop pairs, and a
// random node otherwise; the repair operator is the same greedy reinsertion.
int dcnpDestroy(
    DCNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    double impactSelectionRate, long step);

void dcnpGreedyRepair(DCNP_Graph &graph, int numNodes, long step);

// The DCNP variant of the large neighborhood search: same destroy/repair loop
// and same stopping rule, with the operators above.
LocalSearchResult runDCNPCLNS(
    DCNP_Graph &graph, int seed, const CLNSConfig &clns,
    Deadline deadline = noDeadline());

#endif // DCNP_SEARCH_H
