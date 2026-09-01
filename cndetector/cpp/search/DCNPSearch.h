#ifndef DCNP_SEARCH_H
#define DCNP_SEARCH_H

#include "../Graph/DCNP_Graph.h"
#include "L2NSSearch.h"

LocalSearchResult runDCNPL2NS(
    DCNP_Graph &graph, int seed, const L2NSConfig &l2ns,
    Deadline deadline = noDeadline());

#endif // DCNP_SEARCH_H
