#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "L2NSSearch.h"
#include "DCNPSearch.h"

// Overload set dispatching the L2NS-style local search to the right
// implementation for each graph type, so population/crossover code can be
// written once as templates over the graph type.
inline LocalSearchResult runLocalSearch(
    CNP_Graph &graph, int seed, const L2NSConfig &l2ns,
    Deadline deadline = noDeadline())
{
    return runL2NS(graph, seed, l2ns, deadline);
}

inline LocalSearchResult runLocalSearch(
    DCNP_Graph &graph, int seed, const L2NSConfig &l2ns,
    Deadline deadline = noDeadline())
{
    return runDCNPL2NS(graph, seed, l2ns, deadline);
}

#endif // LOCAL_SEARCH_H
