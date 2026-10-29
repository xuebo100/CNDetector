#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "CHNSSearch.h"
#include "CLNSSearch.h"
#include "DCNPSearch.h"

#include <stdexcept>

enum class SearchStrategy { CLNS, CHNS };

// The local search that refines every new solution, with the budget of each
// strategy.
struct LocalSearchSettings
{
    SearchStrategy strategy = SearchStrategy::CLNS;
    CLNSConfig clns;
    CHNSConfig chns;
};

// Overload set dispatching the local search to the implementation for each
// graph type, so population/crossover code can be written once as templates
// over the graph type.
inline LocalSearchResult runLocalSearch(
    CNP_Graph &graph, int seed, const LocalSearchSettings &search,
    Deadline deadline = noDeadline())
{
    if (search.strategy == SearchStrategy::CHNS)
    {
        return runCHNS(graph, seed, search.chns, deadline);
    }
    return runCLNS(graph, seed, search.clns, deadline);
}

inline LocalSearchResult runLocalSearch(
    DCNP_Graph &graph, int seed, const LocalSearchSettings &search,
    Deadline deadline = noDeadline())
{
    if (search.strategy != SearchStrategy::CLNS)
    {
        throw std::invalid_argument("the CHNS search supports CNP only");
    }
    return runDCNPCLNS(graph, seed, search.clns, deadline);
}

#endif // LOCAL_SEARCH_H
