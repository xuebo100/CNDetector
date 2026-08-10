#ifndef DCNP_SEARCH_H
#define DCNP_SEARCH_H

#include "../Graph/DCNP_Graph.h"
#include "CHNSSearch.h"

LocalSearchResult runDCNPCHNS(
    DCNP_Graph &graph, int seed, const CHNSConfig &chns,
    Deadline deadline = noDeadline());

#endif // DCNP_SEARCH_H
