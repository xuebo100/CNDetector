#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "CHNSSearch.h"
#include "DCNPSearch.h"

// Overload set dispatching the CHNS-style local search to the right
// implementation for each graph type, so population/crossover code can be
// written once as templates over the graph type.
inline LocalSearchResult runLocalSearch(
    CNP_Graph &graph, int seed, const CHNSConfig &chns,
    Deadline deadline = noDeadline())
{
    return runCHNS(graph, seed, chns, deadline);
}

inline LocalSearchResult runLocalSearch(
    DCNP_Graph &graph, int seed, const CHNSConfig &chns,
    Deadline deadline = noDeadline())
{
    return runDCNPCHNS(graph, seed, chns, deadline);
}

#endif // LOCAL_SEARCH_H
