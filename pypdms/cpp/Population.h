#ifndef POPULATION_H
#define POPULATION_H

#include "Graph/CNP_Graph.h"
#include "Graph/DCNP_Graph.h"
#include "RandomNumberGenerator.h"
#include "crossover/reduceSolveCombine.h"

#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

constexpr double ALPHA = 0.7;

struct PopulationItem
{
    std::shared_ptr<const Solution> solution;
    int objValue = std::numeric_limits<int>::max();
    double fitness = 0.0;
    size_t id = 0;
    using Similarity = std::vector<std::pair<double, size_t>>;
    Similarity similarity;
};

struct ExchangeReport
{
    bool exchangeTriggered = false;
    int firstPopulationCandidateObj = -1;
    bool firstPopulationImprovedBest = false;
};

struct IterationEvent
{
    int iteration = 0;
    double elapsed = 0.0;
    int bestObjective = std::numeric_limits<int>::max();
    int populationSize = 0;
};

struct ExchangeEvent
{
    int iteration = 0;
    ExchangeReport report;
};

// Dual-population memetic search, generic over the graph type (CNP_Graph or
// DCNP_Graph). Explicitly instantiated in Population.cpp.
template <typename GraphT>
class DualPopulationT
{
public:
    DualPopulationT(const GraphT &originalGraph,
                    int feasibleBudget,
                    int infeasibleBudget,
                    SolverConfig config,
                    std::chrono::steady_clock::time_point startTime);

    std::pair<Solution, int> initialize();
    void advanceOneGeneration();

    std::vector<IterationEvent> drainIterationEvents();
    std::vector<ExchangeEvent> drainExchangeEvents();

    std::pair<Solution, int> getBestFeasibleSolution() const;
    std::vector<std::pair<Solution, int>> getFeasiblePopulation() const;
    int getFeasiblePopulationSize() const;
    int getFeasibleIterationCount() const;
    int getInfeasibleIterationCount() const;

private:
    using PopulationItems = std::vector<PopulationItem>;
    using ParentHandles
        = std::pair<std::shared_ptr<const Solution>, std::shared_ptr<const Solution>>;
    enum class PopulationKind { Feasible, Infeasible };

    void runPopulationIteration(
        PopulationKind kind, int iteration, RandomNumberGenerator &rng);
    ExchangeReport performExchange(int iteration);
    IterationEvent buildFeasibleIterationEvent(int iteration) const;
    double elapsedSeconds() const;
    bool reachedDeadline() const;

    std::pair<Solution, int> generateRandomSolution(int budget, int seed) const;
    std::pair<Solution, int> createOffspring(
        const ParentHandles &parents,
        std::optional<int> targetBudget,
        int seed) const;
    std::pair<Solution, int> completeSolutionToTargetBudget(
        const Solution &baseSolution, int targetBudget, int seed) const;

    const PopulationItem &getBestItem(const PopulationItems &population) const;
    bool isDuplicate(const Solution &solution, const PopulationItems &population) const;
    void addSolution(PopulationItems &population, const Solution &solution, int objValue);
    void updateFitness(PopulationItems &population) const;
    void removeWorstSolution(PopulationItems &population) const;
    void updatePopulation(
        PopulationItems &population,
        const Solution &solution,
        int objValue,
        size_t maxPopulationSize);
    ParentHandles selectRandomParents(
        const PopulationItems &population, RandomNumberGenerator &rng);

    static double computeSimilarity(const Solution &lhs, const Solution &rhs);

    const GraphT &originalGraph_;
    int feasibleBudget_;
    int infeasibleBudget_;
    SolverConfig config_;
    PopulationItems feasiblePopulation_;
    PopulationItems infeasiblePopulation_;
    std::vector<IterationEvent> iterationEvents_;
    std::vector<ExchangeEvent> exchangeEvents_;
    RandomNumberGenerator feasibleSelectionRng_;
    RandomNumberGenerator infeasibleSelectionRng_;
    size_t nextItemId_ = 0;
    int feasibleIterationCount_ = 0;
    int infeasibleIterationCount_ = 0;
    std::chrono::steady_clock::time_point startTime_;
    // Hard wall-clock deadline (startTime_ + config.maxRuntime). When the
    // budget is unlimited this is time_point::max(). All CHNS calls are
    // bounded by it, and population initialization / generation steps bail
    // out as soon as it passes.
    std::chrono::steady_clock::time_point deadline_;
};

using DualPopulation = DualPopulationT<CNP_Graph>;
using DCNPDualPopulation = DualPopulationT<DCNP_Graph>;

#endif // POPULATION_H
