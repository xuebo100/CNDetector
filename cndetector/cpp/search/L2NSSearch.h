#ifndef L2NS_SEARCH_H
#define L2NS_SEARCH_H

#include "../Graph/CNP_Graph.h"
#include "../RandomNumberGenerator.h"
#include <algorithm>
#include <chrono>

struct L2NSConfig
{
    int maxIdleSteps = 500;
    double theta = 0.3;
    int minBatchSize = 2;
    int maxBatchSize = 10;
    int batchIdleThreshold = 100;
    bool randomizeBatchAndIdle = true;
    // Algorithm 3: lambda ~ U[randomBatchMin, randomBatchMax] = U[1, 50], and
    // the per-run idle budget is xi' = min(500, xi / lambda). randomIdleProduct
    // is xi (the allowable idle iteration count, Table 1) and
    // randomMaxIdleSteps is the constant 500 cap of Algorithm 3, line 4.
    int randomBatchMin = 1;
    int randomBatchMax = 50;
    int randomIdleProduct = 1000;
    int randomMinIdleSteps = 1;
    int randomMaxIdleSteps = 500;
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

LocalSearchResult runL2NS(
    CNP_Graph &graph, int seed, const L2NSConfig &l2ns,
    Deadline deadline = noDeadline());

#endif // L2NS_SEARCH_H
