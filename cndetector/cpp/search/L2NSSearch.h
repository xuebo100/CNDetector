#ifndef L2NS_SEARCH_H
#define L2NS_SEARCH_H

#include "../Graph/CNP_Graph.h"
#include "../RandomNumberGenerator.h"
#include <algorithm>
#include <chrono>

// Budget of one L2NS run. The first block holds the parameters of the search
// itself; the second block configures the optional alternative destroy-size
// schedules reachable through SolverConfig::search.
struct L2NSConfig
{
    // gamma: allowable number of idle (non-improving) iterations of one L2NS
    // run, before it is scaled down by the destroy size.
    int allowableIdleIterations = 1000;
    // Each run draws its destroy size lambda uniformly from
    // [minDestroySize, maxDestroySize].
    int minDestroySize = 1;
    int maxDestroySize = 50;
    // The run stops after min(idleIterationCap, gamma / lambda) consecutive
    // non-improving iterations: a larger destroy size makes each iteration
    // more expensive, so fewer of them are allowed. idleIterationFloor keeps
    // that limit at 1 or above when gamma is smaller than lambda.
    int idleIterationCap = 500;
    int idleIterationFloor = 1;
    // Probability that the destroy operator picks the least-impact node of the
    // chosen large connected component rather than its oldest node. Not one of
    // the tuned solver parameters; it only steers the choice of victim node.
    double impactSelectionRate = 0.3;

    // Optional alternative destroy-size schedule, selected by
    // SolverConfig::search ("L2NS-ADAPT" or "L2NS<N>"). With the default
    // search = "L2NS" these fields are unused and lambda is drawn at random
    // as described above.
    bool randomizeDestroySize = true;
    int adaptiveMaxIdleIterations = 500;
    int adaptiveMinDestroySize = 2;
    int adaptiveMaxDestroySize = 10;
    int adaptiveGrowthInterval = 100;
};

struct LocalSearchResult
{
    Solution solution;
    int objValue = std::numeric_limits<int>::max();
};

// A hard wall-clock deadline. The search returns the best solution found so
// far once the deadline is reached. The default (time_point::max()) means
// "no deadline" so existing callers keep running to natural completion.
using Deadline = std::chrono::steady_clock::time_point;
inline Deadline noDeadline() { return Deadline::max(); }

// Destroy operator: remove destroySize nodes from the large connected
// components of the residual graph. Returns the number of nodes actually
// removed, which is below destroySize only when the residual graph runs out of
// nodes.
int lccDrivenDestroy(
    CNP_Graph &graph, int destroySize, RandomNumberGenerator &rng,
    const L2NSConfig &l2ns, long step);

// Repair operator: reinsert numNodes nodes one at a time, each time picking
// the node whose reinsertion degrades the objective the least.
void greedyRepair(CNP_Graph &graph, int numNodes, long step);

// LCC-oriented large neighbourhood search: alternate the destroy and repair
// operators above until the idle-iteration limit is reached, and return the
// best solution seen. The incumbent solution is carried by the graph's
// removed-node set.
LocalSearchResult runL2NS(
    CNP_Graph &graph, int seed, const L2NSConfig &l2ns,
    Deadline deadline = noDeadline());

#endif // L2NS_SEARCH_H
