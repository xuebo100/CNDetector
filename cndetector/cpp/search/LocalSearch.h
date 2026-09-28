#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "CHNSSearch.h"
#include "DCNPSearch.h"
#include "L2NSSearch.h"

#include <stdexcept>

enum class SearchStrategy { L2NS, CHNS };

// The local search that refines every new solution, with the budget of each
// strategy.
struct LocalSearchSettings
{
    SearchStrategy strategy = SearchStrategy::L2NS;
    L2NSConfig l2ns;
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
    return runL2NS(graph, seed, search.l2ns, deadline);
}

inline LocalSearchResult runLocalSearch(
    DCNP_Graph &graph, int seed, const LocalSearchSettings &search,
    Deadline deadline = noDeadline())
{
    if (search.strategy != SearchStrategy::L2NS)
    {
        throw std::invalid_argument("the CHNS search supports CNP only");
    }
    return runDCNPL2NS(graph, seed, search.l2ns, deadline);
}

#endif // LOCAL_SEARCH_H
