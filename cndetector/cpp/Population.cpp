#include "Population.h"

#include "ParallelFor.h"
#include "search/LocalSearch.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

template <typename GraphT>
double DualPopulationT<GraphT>::computeSimilarity(const Solution &lhs, const Solution &rhs)
{
    size_t intersection = 0;
    for (Node node : lhs)
    {
        if (rhs.contains(node)) ++intersection;
    }
    const auto unionSize = lhs.size() + rhs.size() - intersection;
    if (unionSize == 0) return 0.0;
    return static_cast<double>(intersection) / static_cast<double>(unionSize);
}

template <typename GraphT>
DualPopulationT<GraphT>::DualPopulationT(
    const GraphT &originalGraph,
    int feasibleBudget,
    int infeasibleBudget,
    SolverConfig config,
    std::chrono::steady_clock::time_point startTime)
    : originalGraph_(originalGraph),
      feasibleBudget_(feasibleBudget),
      infeasibleBudget_(infeasibleBudget),
      config_(std::move(config)),
      startTime_(startTime)
{
    config_.resolvedL2ns = resolveL2NSConfig(config_);
    deadline_ = config_.maxRuntime > 0.0
                    ? startTime_
                          + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(config_.maxRuntime))
                    : std::chrono::steady_clock::time_point::max();
}

template <typename GraphT>
bool DualPopulationT<GraphT>::reachedDeadline() const
{
    return std::chrono::steady_clock::now() >= deadline_;
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::initialize()
{
    feasiblePopulation_.clear();
    infeasiblePopulation_.clear();
    nextItemId_ = 0;
    feasibleIterationCount_ = 0;
    infeasibleIterationCount_ = 0;
    feasibleSelectionRng_.setSeed(deterministicSeed(config_.seed, 0xF001u));
    infeasibleSelectionRng_.setSeed(deterministicSeed(config_.seed, 0xF002u));

    iterationEvents_.clear();
    exchangeEvents_.clear();

    // Population members are independent local-search runs with fixed
    // per-member seeds, so they are generated in parallel and appended in a
    // fixed order afterwards. The first (feasible) member always runs so the
    // population is never empty; later members are skipped once the time
    // budget is exhausted, mirroring the sequential early-break.
    struct InitJob
    {
        int budget;
        int seed;
    };
    std::vector<InitJob> jobs;
    jobs.reserve(static_cast<size_t>(config_.populationSize) * 2);
    for (int i = 0; i < config_.populationSize; ++i)
    {
        jobs.push_back({feasibleBudget_,
                        deterministicSeed(config_.seed, 0x1001u, 0,
                                          static_cast<uint64_t>(i))});
        jobs.push_back({infeasibleBudget_,
                        deterministicSeed(config_.seed, 0x1002u, 0,
                                          static_cast<uint64_t>(i))});
    }

    std::vector<std::optional<std::pair<Solution, int>>> results(jobs.size());
    pdms::parallelFor(0, static_cast<int>(jobs.size()),
                      [&](int k, int /*workerIdx*/)
                      {
                          // Both member-0 solutions always run (matching the
                          // sequential per-member deadline check), so neither
                          // population is ever left empty.
                          if (k > 1 && reachedDeadline())
                          {
                              return;
                          }
                          results[k] = generateRandomSolution(jobs[k].budget,
                                                              jobs[k].seed);
                      });

    for (size_t k = 0; k < jobs.size(); ++k)
    {
        if (!results[k].has_value())
        {
            continue;
        }
        auto &population
            = k % 2 == 0 ? feasiblePopulation_ : infeasiblePopulation_;
        addSolution(population, results[k]->first, results[k]->second);
    }

    const auto &bestItem = getBestItem(feasiblePopulation_);
    return {*bestItem.solution, bestItem.objValue};
}

template <typename GraphT>
void DualPopulationT<GraphT>::advanceOneGeneration()
{
    const int feasibleIteration = feasibleIterationCount_ + 1;
    const int infeasibleIteration = infeasibleIterationCount_ + 1;

    // Collect the generation's offspring jobs for both populations (parent
    // selection stays sequential so the RNG streams match the serial order),
    // run the heavy createOffspring calls in parallel, then apply the results
    // in job order.
    std::vector<OffspringJob> jobs;
    jobs.reserve(static_cast<size_t>(config_.threadCount) * 2);
    collectOffspringJobs(
        PopulationKind::Feasible, feasibleIteration, feasibleSelectionRng_, jobs);
    collectOffspringJobs(
        PopulationKind::Infeasible, infeasibleIteration, infeasibleSelectionRng_, jobs);

    std::vector<std::pair<Solution, int>> results(jobs.size());
    pdms::parallelFor(0, static_cast<int>(jobs.size()),
                      [&](int k, int /*workerIdx*/)
                      {
                          results[k] = createOffspring(
                              jobs[k].parents, jobs[k].targetBudget, jobs[k].seed);
                      });

    // PMS (Algorithm 2, lines 8-9): the offspring of a generation form a
    // temporary population P', from which only the single best solution
    // S* = argmin f(S_i) is passed to the population update. Ties keep the
    // earliest job so the outcome stays independent of the thread schedule.
    applyBestOffspring(feasiblePopulation_, PopulationKind::Feasible, jobs, results);
    applyBestOffspring(infeasiblePopulation_, PopulationKind::Infeasible, jobs, results);

    feasibleIterationCount_ = feasibleIteration;
    infeasibleIterationCount_ = infeasibleIteration;

    // Algorithm 1, lines 10-15: track I'_g, the number of consecutive
    // generations in which the incumbent did not improve.
    const int generationBest = getBestItem(feasiblePopulation_).objValue;
    if (generationBest < bestObjective_)
    {
        bestObjective_ = generationBest;
        idleGenerations_ = 0;
    }
    else
    {
        ++idleGenerations_;
    }

    if (feasibleIteration % config_.interactionPeriod == 0 && !reachedDeadline())
    {
        exchangeEvents_.push_back(
            ExchangeEvent{feasibleIteration, performExchange(feasibleIteration)});
    }

    iterationEvents_.push_back(buildFeasibleIterationEvent(feasibleIteration));
}

template <typename GraphT>
void DualPopulationT<GraphT>::applyBestOffspring(
    PopulationItems &population,
    PopulationKind kind,
    const std::vector<OffspringJob> &jobs,
    const std::vector<std::pair<Solution, int>> &results)
{
    const std::pair<Solution, int> *best = nullptr;
    for (size_t k = 0; k < jobs.size(); ++k)
    {
        if (jobs[k].kind != kind)
        {
            continue;
        }
        if (best == nullptr || results[k].second < best->second)
        {
            best = &results[k];
        }
    }

    if (best != nullptr)
    {
        updatePopulation(population, best->first, best->second,
                         static_cast<size_t>(config_.populationSize));
    }
}

template <typename GraphT>
void DualPopulationT<GraphT>::collectOffspringJobs(
    PopulationKind kind,
    int iteration,
    RandomNumberGenerator &rng,
    std::vector<OffspringJob> &jobs)
{
    PopulationItems &population
        = kind == PopulationKind::Feasible ? feasiblePopulation_ : infeasiblePopulation_;
    const std::optional<int> targetBudget
        = kind == PopulationKind::Feasible ? std::nullopt
                                           : std::optional<int>(infeasibleBudget_);

    for (int i = 0; i < config_.threadCount; ++i)
    {
        if (reachedDeadline())
        {
            break;
        }
        const uint64_t streamId = kind == PopulationKind::Feasible ? 0x2001u : 0x2002u;
        const int offspringSeed = deterministicSeed(
            config_.seed, streamId,
            static_cast<uint64_t>(iteration), static_cast<uint64_t>(i));

        jobs.push_back({kind, selectRandomParents(population, rng),
                        targetBudget, offspringSeed});
    }
}

template <typename GraphT>
ExchangeReport DualPopulationT<GraphT>::performExchange(int iteration)
{
    ExchangeReport report;
    report.exchangeTriggered = true;
    const Solution bestInfeasible = *getBestItem(infeasiblePopulation_).solution;

    const int completionSeed = deterministicSeed(
        config_.seed, 0x3001u, static_cast<uint64_t>(iteration));
    const auto [completedSolution, completedObjValue]
        = completeSolutionToTargetBudget(bestInfeasible, feasibleBudget_, completionSeed);
    report.firstPopulationCandidateObj = completedObjValue;

    const int bestBefore = getBestItem(feasiblePopulation_).objValue;
    updatePopulation(feasiblePopulation_, completedSolution, completedObjValue,
                     static_cast<size_t>(config_.populationSize));
    report.firstPopulationImprovedBest
        = getBestItem(feasiblePopulation_).objValue < bestBefore;

    // Algorithm 4, lines 8-11: once the search has stagnated for more than
    // delta generations, rebuild the auxiliary population and reset I'_g.
    if (idleGenerations_ > config_.stagnationThreshold)
    {
        reconstructAuxiliaryPopulation(iteration);
        idleGenerations_ = 0;
    }
    return report;
}

template <typename GraphT>
void DualPopulationT<GraphT>::reconstructAuxiliaryPopulation(int iteration)
{
    // Section 4.4: keep the incumbent of P_a and replace the remaining
    // theta - 1 individuals with fresh solutions built by the PPI procedure.
    const PopulationItem &incumbent = getBestItem(infeasiblePopulation_);
    const Solution keptSolution = *incumbent.solution;
    const int keptObjValue = incumbent.objValue;

    const int replacements = std::max(0, config_.populationSize - 1);
    std::vector<std::optional<std::pair<Solution, int>>> results(
        static_cast<size_t>(replacements));
    pdms::parallelFor(0, replacements,
                      [&](int k, int /*workerIdx*/)
                      {
                          if (reachedDeadline())
                          {
                              return;
                          }
                          results[static_cast<size_t>(k)] = generateRandomSolution(
                              infeasibleBudget_,
                              deterministicSeed(config_.seed, 0x4001u,
                                                static_cast<uint64_t>(iteration),
                                                static_cast<uint64_t>(k)));
                      });

    infeasiblePopulation_.clear();
    addSolution(infeasiblePopulation_, keptSolution, keptObjValue);
    for (const auto &result : results)
    {
        if (result.has_value())
        {
            addSolution(infeasiblePopulation_, result->first, result->second);
        }
    }
}

template <typename GraphT>
IterationEvent DualPopulationT<GraphT>::buildFeasibleIterationEvent(int iteration) const
{
    return {
        iteration,
        elapsedSeconds(),
        getBestItem(feasiblePopulation_).objValue,
        static_cast<int>(feasiblePopulation_.size()),
    };
}

template <typename GraphT>
double DualPopulationT<GraphT>::elapsedSeconds() const
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - startTime_)
        .count();
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::generateRandomSolution(int budget, int seed) const
{
    std::unique_ptr<GraphT> graph;
    if (budget == feasibleBudget_)
    {
        graph = originalGraph_.getRandomFeasibleGraph(deterministicSeed(seed, 0x4001u));
    }
    else
    {
        graph = originalGraph_.getRandomPartialGraph(budget, deterministicSeed(seed, 0x4002u));
    }
    const LocalSearchResult result
        = runLocalSearch(*graph, deterministicSeed(seed, 0x4003u), config_.resolvedL2ns, deadline_);
    return {result.solution, result.objValue};
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::createOffspring(
    const ParentHandles &parents,
    std::optional<int> targetBudget,
    int seed) const
{
    auto parentPtrs = std::pair<const Solution *, const Solution *>(
        parents.first.get(), parents.second.get());
    auto offspringGraph = reduceSolveCombine(
        originalGraph_, parentPtrs, config_.beta, targetBudget, seed, config_, deadline_);
    const LocalSearchResult result
        = runLocalSearch(*offspringGraph, seed + 10000, config_.resolvedL2ns, deadline_);
    return {result.solution, result.objValue};
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::completeSolutionToTargetBudget(
    const Solution &baseSolution, int targetBudget, int seed) const
{
    if (static_cast<int>(baseSolution.size()) > targetBudget)
    {
        throw std::invalid_argument("base solution size exceeds target budget");
    }

    auto workingGraph = originalGraph_.clone();
    workingGraph->getReducedGraphByRemovedNodes(baseSolution);

    std::unique_ptr<GraphT> reducedGraph;
    if (static_cast<int>(baseSolution.size()) == targetBudget)
    {
        reducedGraph = std::move(workingGraph);
    }
    else if (targetBudget == feasibleBudget_)
    {
        reducedGraph = workingGraph->getRandomFeasibleGraph(deterministicSeed(seed, 0x6001u));
    }
    else
    {
        reducedGraph = workingGraph->getRandomPartialGraph(
            targetBudget - static_cast<int>(baseSolution.size()),
            deterministicSeed(seed, 0x6002u));
    }

    LocalSearchResult result
        = runLocalSearch(*reducedGraph, deterministicSeed(seed, 0x6003u), config_.resolvedL2ns, deadline_);
    Solution finalNodes = baseSolution;
    finalNodes.reserve(baseSolution.size() + result.solution.size());
    finalNodes.insert(result.solution.begin(), result.solution.end());

    auto improvedGraph = originalGraph_.clone();
    improvedGraph->updateGraphByRemovedNodes(finalNodes);
    result = runLocalSearch(*improvedGraph, deterministicSeed(seed, 0x6004u), config_.resolvedL2ns, deadline_);
    return {result.solution, result.objValue};
}

template <typename GraphT>
const PopulationItem &DualPopulationT<GraphT>::getBestItem(const PopulationItems &population) const
{
    if (population.empty())
    {
        throw std::runtime_error("Population is empty");
    }
    return *std::min_element(population.begin(), population.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.objValue < rhs.objValue; });
}

template <typename GraphT>
bool DualPopulationT<GraphT>::isDuplicate(
    const Solution &solution, const PopulationItems &population) const
{
    for (const auto &item : population)
    {
        if (*item.solution == solution) return true;
    }
    return false;
}

template <typename GraphT>
void DualPopulationT<GraphT>::addSolution(
    PopulationItems &population, const Solution &solution, int objValue)
{
    PopulationItem item;
    item.solution = std::make_shared<Solution>(solution);
    item.objValue = objValue;
    item.id = nextItemId_++;
    item.similarity.reserve(population.size());

    for (auto &otherItem : population)
    {
        const double similarity = computeSimilarity(*item.solution, *otherItem.solution);
        item.similarity.push_back({similarity, otherItem.id});
        otherItem.similarity.push_back({similarity, item.id});
    }

    population.push_back(std::move(item));
}

template <typename GraphT>
void DualPopulationT<GraphT>::updateFitness(PopulationItems &population) const
{
    if (population.size() <= 1)
    {
        if (!population.empty()) population.front().fitness = 0.0;
        return;
    }

    std::vector<double> costs(population.size());
    std::vector<double> diversity(population.size());
    for (size_t i = 0; i < population.size(); ++i)
    {
        costs[i] = population[i].objValue;
        if (population[i].similarity.empty())
        {
            diversity[i] = 0.0;
            continue;
        }
        double sumSimilarity = 0.0;
        for (const auto &[similarity, _] : population[i].similarity)
        {
            sumSimilarity += similarity;
        }
        diversity[i] = sumSimilarity / static_cast<double>(population[i].similarity.size());
    }

    auto calculateRanks = [](const std::vector<double> &values)
    {
        std::vector<size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(),
            [&values](size_t lhs, size_t rhs)
            {
                if (values[lhs] != values[rhs]) return values[lhs] < values[rhs];
                return lhs < rhs;
            });
        std::vector<int> ranks(values.size(), 0);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            ranks[indices[i]] = static_cast<int>(i + 1);
        }
        return ranks;
    };

    const auto costRanks = calculateRanks(costs);
    const auto diversityRanks = calculateRanks(diversity);
    for (size_t i = 0; i < population.size(); ++i)
    {
        population[i].fitness = ALPHA * costRanks[i] + (1.0 - ALPHA) * diversityRanks[i];
    }
}

template <typename GraphT>
void DualPopulationT<GraphT>::removeWorstSolution(PopulationItems &population) const
{
    if (population.empty()) return;

    updateFitness(population);
    auto worstIt = std::max_element(population.begin(), population.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.fitness < rhs.fitness; });

    const size_t worstId = worstIt->id;
    for (auto &item : population)
    {
        if (item.id == worstId) continue;
        auto &similarity = item.similarity;
        similarity.erase(
            std::remove_if(similarity.begin(), similarity.end(),
                [worstId](const auto &entry) { return entry.second == worstId; }),
            similarity.end());
    }

    population.erase(worstIt);
}

template <typename GraphT>
void DualPopulationT<GraphT>::updatePopulation(
    PopulationItems &population,
    const Solution &solution,
    int objValue,
    size_t maxPopulationSize)
{
    addSolution(population, solution, objValue);
    if (population.size() > maxPopulationSize)
    {
        removeWorstSolution(population);
    }
}

template <typename GraphT>
typename DualPopulationT<GraphT>::ParentHandles DualPopulationT<GraphT>::selectRandomParents(
    const PopulationItems &population, RandomNumberGenerator &rng)
{
    if (population.size() < 2)
    {
        throw std::runtime_error("Cannot select two parents from a population with size < 2");
    }
    const int first = rng.generateIndex(static_cast<int>(population.size()));
    int second = first;
    while (second == first)
    {
        second = rng.generateIndex(static_cast<int>(population.size()));
    }
    return {population[first].solution, population[second].solution};
}

template <typename GraphT>
std::vector<IterationEvent> DualPopulationT<GraphT>::drainIterationEvents()
{
    std::vector<IterationEvent> drained;
    drained.swap(iterationEvents_);
    return drained;
}

template <typename GraphT>
std::vector<ExchangeEvent> DualPopulationT<GraphT>::drainExchangeEvents()
{
    std::vector<ExchangeEvent> drained;
    drained.swap(exchangeEvents_);
    return drained;
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::getBestFeasibleSolution() const
{
    const auto &bestItem = getBestItem(feasiblePopulation_);
    return {*bestItem.solution, bestItem.objValue};
}

template <typename GraphT>
std::vector<std::pair<Solution, int>> DualPopulationT<GraphT>::getFeasiblePopulation() const
{
    std::vector<std::pair<Solution, int>> snapshot;
    snapshot.reserve(feasiblePopulation_.size());
    for (const auto &item : feasiblePopulation_)
    {
        snapshot.emplace_back(*item.solution, item.objValue);
    }
    return snapshot;
}

template <typename GraphT>
int DualPopulationT<GraphT>::getFeasiblePopulationSize() const
{
    return static_cast<int>(feasiblePopulation_.size());
}

template <typename GraphT>
int DualPopulationT<GraphT>::getFeasibleIterationCount() const
{
    return feasibleIterationCount_;
}

template <typename GraphT>
int DualPopulationT<GraphT>::getInfeasibleIterationCount() const
{
    return infeasibleIterationCount_;
}

template class DualPopulationT<CNP_Graph>;
template class DualPopulationT<DCNP_Graph>;
