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

// Weight of the cost rank in the quality-diversity fitness that decides which
// individual to evict; the diversity rank carries the remaining
// 1 - QUALITY_WEIGHT.
constexpr double QUALITY_WEIGHT = 0.7;

struct PopulationItem
{
    std::shared_ptr<const Solution> solution;
    int objValue = std::numeric_limits<int>::max();
    double fitness = 0.0;
    size_t id = 0;
    using Similarity = std::vector<std::pair<double, size_t>>;
    Similarity similarity;
};

struct HPCReport
{
    bool hpcTriggered = false;
    int candidateObjValue = -1;
    bool improvedMainBest = false;
};

struct IterationEvent
{
    int iteration = 0;
    double elapsed = 0.0;
    int bestObjective = std::numeric_limits<int>::max();
    int populationSize = 0;
};

struct HPCEvent
{
    int iteration = 0;
    HPCReport report;
};

// The parallel co-evolutionary memetic search: a main population of solutions
// removing exactly k nodes, and an auxiliary population of solutions removing
// only floor(k * (1 - alpha)) nodes. Both evolve in parallel and exchange
// information every interactionPeriod generations. Generic over the graph type
// (CNP_Graph or DCNP_Graph); explicitly instantiated in Population.cpp.
template <typename GraphT>
class DualPopulationT
{
public:
    DualPopulationT(const GraphT &originalGraph,
                    int mainBudget,
                    int auxiliaryBudget,
                    SolverConfig config,
                    std::chrono::steady_clock::time_point startTime);

    // Parallel population initialization: fill both populations with theta
    // distinct local optima and return the best solution found.
    std::pair<Solution, int> initialize();
    // One generation: a parallel memetic search step on each population,
    // followed by a population cooperation step every interactionPeriod
    // generations.
    void advanceOneGeneration();

    std::vector<IterationEvent> drainIterationEvents();
    std::vector<HPCEvent> drainHPCEvents();

    std::pair<Solution, int> getBestSolution() const;
    std::vector<std::pair<Solution, int>> getMainPopulation() const;
    int getMainPopulationSize() const;
    int getMainGenerationCount() const;
    int getAuxiliaryGenerationCount() const;

private:
    using PopulationItems = std::vector<PopulationItem>;
    using ParentHandles
        = std::pair<std::shared_ptr<const Solution>, std::shared_ptr<const Solution>>;
    enum class PopulationKind { Main, Auxiliary };

    /**
     * One offspring to create during a generation. Parents are selected
     * sequentially up front (fixed RNG order), the heavy createOffspring
     * calls then run in parallel, and results are applied to the populations
     * in job order — so results are deterministic for any thread count.
     */
    struct OffspringJob
    {
        PopulationKind kind;
        ParentHandles parents;
        std::optional<int> targetBudget;
        int seed;
    };

    // Pick the threadCount parent pairs of one generation, one per offspring.
    void collectOffspringJobs(PopulationKind kind,
                              int iteration,
                              RandomNumberGenerator &rng,
                              std::vector<OffspringJob> &jobs);
    // Take the cheapest of the generation's offspring and offer it to the
    // population update.
    void applyBestOffspring(PopulationItems &population,
                            PopulationKind kind,
                            const std::vector<OffspringJob> &jobs,
                            const std::vector<std::pair<Solution, int>> &results);
    // Heterogeneous population cooperation: complete the best solution of the
    // auxiliary population to the full budget and offer it to the main one.
    HPCReport runHPC(int iteration);
    // Population reconstruction: rebuild the auxiliary population, keeping only
    // its incumbent.
    void runPR(int iteration);
    IterationEvent buildIterationEvent(int iteration) const;
    double elapsedSeconds() const;
    bool reachedDeadline() const;

    // One initial individual: a random solution of the given size, improved to
    // a local optimum by the large neighbourhood search.
    std::pair<Solution, int> generatePPIIndividual(int budget, int seed) const;
    // Recombine the two parents with the crossover, then refine the result with
    // the large neighbourhood search.
    std::pair<Solution, int> createOffspring(
        const ParentHandles &parents,
        std::optional<int> targetBudget,
        int seed) const;
    // Fix the argument as a partial solution, complete it at random up to
    // targetBudget nodes, then refine the result with the local search.
    std::pair<Solution, int> completePartialSolution(
        const Solution &baseSolution, int targetBudget, int seed) const;

    const PopulationItem &getBestItem(const PopulationItems &population) const;
    const PopulationItem &getWorstItem(const PopulationItems &population) const;
    bool isDuplicate(const Solution &solution, const PopulationItems &population) const;
    void addSolution(PopulationItems &population, const Solution &solution, int objValue);
    void updateFitness(PopulationItems &population) const;
    void removeWorstSolution(PopulationItems &population) const;
    // Quality-diversity population updating: reject the offspring when it
    // duplicates an individual of the population or is not better than the
    // population's worst cost; otherwise admit it in place of the individual
    // with the worst quality-diversity fitness.
    void applyQDPU(
        PopulationItems &population,
        const Solution &solution,
        int objValue,
        size_t maxPopulationSize);
    ParentHandles selectRandomParents(
        const PopulationItems &population, RandomNumberGenerator &rng);

    static double computeSimilarity(const Solution &lhs, const Solution &rhs);

    const GraphT &originalGraph_;
    int mainBudget_;
    int auxiliaryBudget_;
    SolverConfig config_;
    PopulationItems mainPopulation_;
    PopulationItems auxiliaryPopulation_;
    std::vector<IterationEvent> iterationEvents_;
    std::vector<HPCEvent> hpcEvents_;
    // Generations since the incumbent of the main population last improved.
    int idleGenerations_ = 0;
    int bestObjective_ = std::numeric_limits<int>::max();
    RandomNumberGenerator mainSelectionRng_;
    RandomNumberGenerator auxiliarySelectionRng_;
    size_t nextItemId_ = 0;
    int mainGenerationCount_ = 0;
    int auxiliaryGenerationCount_ = 0;
    std::chrono::steady_clock::time_point startTime_;
    // Hard wall-clock deadline (startTime_ + config.maxRuntime). When the
    // budget is unlimited this is time_point::max(). All L2NS calls are
    // bounded by it, and population initialization / generation steps bail
    // out as soon as it passes.
    std::chrono::steady_clock::time_point deadline_;
};

using DualPopulation = DualPopulationT<CNP_Graph>;
using DCNPDualPopulation = DualPopulationT<DCNP_Graph>;

#endif // POPULATION_H
