#ifndef REDUCE_SOLVE_COMBINE_H
#define REDUCE_SOLVE_COMBINE_H

#include "../Graph/Types.h"
#include "../search/LocalSearch.h"
#include <memory>
#include <optional>
#include <string>

// The defaults are the tuned values of the solver: theta = 10, kappa = 2,
// beta = 20, alpha = 0.05, gamma = 1000 (CLNSConfig::allowableIdleIterations)
// and delta = 500.
struct SolverConfig
{
    // theta: size of each of the two populations.
    int populationSize = 10;
    // kappa: thread count. Each population produces this many offspring per
    // generation, and the search runs on up to this many worker threads (see
    // cndetector::setMaxThreads for a lower cap).
    int threadCount = 2;
    // beta: generations between two heterogeneous population cooperations,
    // i.e. how often the two populations exchange information.
    int interactionPeriod = 20;
    int seed = 0;
    // alpha: relaxation coefficient. The auxiliary population carries
    // floor(k * (1 - alpha)) nodes per solution instead of k.
    double relaxationCoefficient = 0.05;
    // delta: consecutive non-improving generations after which the auxiliary
    // population is rebuilt from scratch.
    int stagnationThreshold = 500;
    // Probability with which the crossover keeps a node of the backbone shared
    // by the two parents. Not one of the tuned parameters above.
    double backboneRate = 0.9;
    double displayInterval = 60.0;
    // Hard wall-clock budget in seconds. <= 0 means "no limit".
    double maxRuntime = 0.0;
    // Local search refining every new solution: "CLNS" (default) or "CHNS".
    std::string search = "CLNS";
    // Crossover producing the offspring: "RSC" (default) or "DBX".
    std::string crossover = "RSC";
    CLNSConfig clns;
    CHNSConfig chns;
};

enum class CrossoverStrategy { RSC, DBX };

uint64_t splitmix64(uint64_t value);

int deterministicSeed(
    int baseSeed, uint64_t streamId, uint64_t iteration = 0, uint64_t slot = 0);

// Parse config.search and config.crossover; both throw std::invalid_argument
// on an unknown name.
LocalSearchSettings resolveLocalSearch(const SolverConfig &config);
CrossoverStrategy resolveCrossover(const SolverConfig &config);

// Reduce-solve-combine crossover: fix the backbone shared by the two parents
// (reduce), run the local search on the reduced instance (solve), then merge
// the backbone with that result (combine). Generic over the graph type
// (CNP_Graph or DCNP_Graph); explicitly instantiated in reduceSolveCombine.cpp.
template <typename GraphT>
std::unique_ptr<GraphT> reduceSolveCombine(
    const GraphT &originalGraph,
    const std::pair<const Solution *, const Solution *> &parents,
    double backboneRate,
    std::optional<int> targetBudget,
    int seed,
    const LocalSearchSettings &search,
    Deadline deadline = noDeadline());

#endif // REDUCE_SOLVE_COMBINE_H
