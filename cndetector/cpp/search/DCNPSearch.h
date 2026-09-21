#ifndef DCNP_SEARCH_H
#define DCNP_SEARCH_H

#include "../Graph/DCNP_Graph.h"
#include "L2NSSearch.h"

// DCNP counterparts of the two L2NS operators. The residual graph of the DCNP
// has no meaningful connected components, so the destroy operator picks the
// node with the largest b-hop-tree impact instead of a node from a large
// connected component; the repair operator is the same greedy reinsertion.
int dcnpDestroy(
    DCNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    const L2NSConfig &l2ns, long step);

void dcnpGreedyRepair(DCNP_Graph &graph, int numNodes, long step);

// The DCNP variant of the large neighbourhood search: same destroy/repair loop
// and same stopping rule, with the operators above.
LocalSearchResult runDCNPL2NS(
    DCNP_Graph &graph, int seed, const L2NSConfig &l2ns,
    Deadline deadline = noDeadline());

#endif // DCNP_SEARCH_H
