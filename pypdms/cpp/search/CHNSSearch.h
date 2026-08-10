#ifndef CHNS_SEARCH_H
#define CHNS_SEARCH_H

#include "../Graph/CNP_Graph.h"
#include "../RandomNumberGenerator.h"
#include <algorithm>
#include <chrono>

struct CHNSConfig
{
    int maxIdleSteps = 500;
    double theta = 0.3;
    int minBatchSize = 2;
    int maxBatchSize = 10;
    int batchIdleThreshold = 100;
    bool randomizeBatchAndIdle = true;
    int randomBatchMin = 1;
    int randomBatchMax = 50;
    int randomIdleProduct = 2000;
    int randomMinIdleSteps = 40;
    int randomMaxIdleSteps = 1000;
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

LocalSearchResult runCHNS(
    CNP_Graph &graph, int seed, const CHNSConfig &chns,
    Deadline deadline = noDeadline());

#endif // CHNS_SEARCH_H
