#include "Population.h"

#include "ParallelFor.h"
#include "crossover/doubleBackboneBasedCrossover.h"
#include "search/LocalSearch.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>

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
    int mainBudget,
    int auxiliaryBudget,
    SolverConfig config,
    std::chrono::steady_clock::time_point startTime)
    : originalGraph_(originalGraph),
      mainBudget_(mainBudget),
      auxiliaryBudget_(auxiliaryBudget),
      config_(std::move(config)),
      search_(resolveLocalSearch(config_)),
      crossover_(resolveCrossover(config_)),
      startTime_(startTime)
{
    if constexpr (!std::is_same_v<GraphT, CNP_Graph>)
    {
        if (search_.strategy != SearchStrategy::L2NS
            || crossover_ != CrossoverStrategy::RSC)
        {
            throw std::invalid_argument(
                "the CHNS search and the DBX crossover support CNP only");
        }
    }
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
    cndetector::setSolverThreads(config_.threadCount);
    mainPopulation_.clear();
    auxiliaryPopulation_.clear();
    nextItemId_ = 0;
    mainGenerationCount_ = 0;
    auxiliaryGenerationCount_ = 0;
    mainSelectionRng_.setSeed(deterministicSeed(config_.seed, 0xF001u));
    auxiliarySelectionRng_.setSeed(deterministicSeed(config_.seed, 0xF002u));

    generationEvents_.clear();
    hpcEvents_.clear();

    fillPopulations({{&mainPopulation_, mainBudget_, 0x1001u},
                     {&auxiliaryPopulation_, auxiliaryBudget_, 0x1002u}},
                    0);

    // Best solution of the main population.
    const auto &bestItem = getBestItem(mainPopulation_);
    return {*bestItem.solution, bestItem.objValue};
}

template <typename GraphT>
void DualPopulationT<GraphT>::advanceOneGeneration()
{
    cndetector::setSolverThreads(config_.threadCount);
    const int mainGeneration = mainGenerationCount_ + 1;
    const int auxiliaryGeneration = auxiliaryGenerationCount_ + 1;

    // Collect the generation's offspring jobs for both populations (parent
    // selection stays sequential so the RNG streams match the serial order),
    // run the heavy createOffspring calls in parallel, then apply the results
    // in job order.
    std::vector<OffspringJob> jobs;
    jobs.reserve(static_cast<size_t>(config_.threadCount) * 2);
    collectOffspringJobs(
        PopulationKind::Main, mainGeneration, mainSelectionRng_, jobs);
    collectOffspringJobs(
        PopulationKind::Auxiliary, auxiliaryGeneration, auxiliarySelectionRng_, jobs);

    std::vector<std::pair<Solution, int>> results(jobs.size());
    cndetector::parallelFor(0, static_cast<int>(jobs.size()),
                      [&](int k, int /*workerIdx*/)
                      {
                          results[k] = createOffspring(
                              jobs[k].parents, jobs[k].targetBudget, jobs[k].seed);
                      });

    // The offspring of a generation form a temporary pool from which only the
    // single cheapest solution is passed to the population update. Ties keep
    // the earliest job so the outcome stays independent of the thread
    // schedule.
    applyBestOffspring(mainPopulation_, PopulationKind::Main, jobs, results);
    applyBestOffspring(auxiliaryPopulation_, PopulationKind::Auxiliary, jobs, results);

    mainGenerationCount_ = mainGeneration;
    auxiliaryGenerationCount_ = auxiliaryGeneration;

    // The two populations cooperate every interactionPeriod generations, using
    // the idle-generation count accumulated by the preceding generations.
    if (mainGeneration % config_.interactionPeriod == 0 && !reachedDeadline())
    {
        hpcEvents_.push_back(
            HPCEvent{mainGeneration, runHPC(mainGeneration)});
    }

    // Refresh the incumbent and the count of consecutive generations without
    // improvement. Runs after the cooperation step so a solution injected by
    // it counts for this generation.
    const int generationBest = getBestItem(mainPopulation_).objValue;
    if (generationBest < bestObjective_)
    {
        bestObjective_ = generationBest;
        idleGenerations_ = 0;
    }
    else
    {
        ++idleGenerations_;
    }

    generationEvents_.push_back(buildGenerationEvent(mainGeneration));
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
        applyQDPU(population, best->first, best->second,
                         static_cast<size_t>(config_.populationSize));
    }
}

template <typename GraphT>
void DualPopulationT<GraphT>::collectOffspringJobs(
    PopulationKind kind,
    int generation,
    RandomNumberGenerator &rng,
    std::vector<OffspringJob> &jobs)
{
    PopulationItems &population
        = kind == PopulationKind::Main ? mainPopulation_ : auxiliaryPopulation_;
    const std::optional<int> targetBudget
        = kind == PopulationKind::Main ? std::nullopt
                                           : std::optional<int>(auxiliaryBudget_);

    for (int i = 0; i < config_.threadCount; ++i)
    {
        if (reachedDeadline())
        {
            break;
        }
        const uint64_t streamId = kind == PopulationKind::Main ? 0x2001u : 0x2002u;
        const int offspringSeed = deterministicSeed(
            config_.seed, streamId,
            static_cast<uint64_t>(generation), static_cast<uint64_t>(i));

        jobs.push_back({kind, selectRandomParents(population, rng),
                        targetBudget, offspringSeed});
    }
}

template <typename GraphT>
HPCReport DualPopulationT<GraphT>::runHPC(int generation)
{
    HPCReport report;
    report.hpcTriggered = true;
    const Solution bestAuxiliary = *getBestItem(auxiliaryPopulation_).solution;

    const int completionSeed = deterministicSeed(
        config_.seed, 0x3001u, static_cast<uint64_t>(generation));
    const auto [completedSolution, completedObjValue]
        = completePartialSolution(bestAuxiliary, mainBudget_, completionSeed);
    report.candidateObjValue = completedObjValue;

    const int bestBefore = getBestItem(mainPopulation_).objValue;
    applyQDPU(mainPopulation_, completedSolution, completedObjValue,
                     static_cast<size_t>(config_.populationSize));
    report.improvedMainBest
        = getBestItem(mainPopulation_).objValue < bestBefore;

    // Once the search has stagnated for more than stagnationThreshold
    // generations, rebuild the auxiliary population and reset the counter.
    if (idleGenerations_ > config_.stagnationThreshold)
    {
        runPR(generation);
        idleGenerations_ = 0;
    }
    return report;
}

template <typename GraphT>
void DualPopulationT<GraphT>::runPR(int generation)
{
    // Keep the incumbent of the auxiliary population and replace the remaining
    // populationSize - 1 individuals with freshly initialized solutions.
    const PopulationItem &incumbent = getBestItem(auxiliaryPopulation_);
    const Solution keptSolution = *incumbent.solution;
    const int keptObjValue = incumbent.objValue;

    auxiliaryPopulation_.clear();
    addSolution(auxiliaryPopulation_, keptSolution, keptObjValue);
    fillPopulations({{&auxiliaryPopulation_, auxiliaryBudget_, 0x4001u}}, generation);
}

template <typename GraphT>
void DualPopulationT<GraphT>::fillPopulations(
    const std::vector<FillTarget> &targets, int generation)
{
    // Each round generates the missing members of every target in one
    // parallel region, then admits them in a fixed order when they are
    // distinct from the members already present, so the outcome does not
    // depend on the thread schedule. Rounds repeat until every target is full
    // or has used its attempts. A target that is still empty always runs its
    // first job, even past the deadline, so no population is ever left empty.
    struct FillJob
    {
        size_t target;
        int seed;
        bool mandatory;
    };

    const size_t targetSize = static_cast<size_t>(config_.populationSize);
    const int maxAttempts = FILL_ATTEMPTS_PER_MEMBER * config_.populationSize;
    std::vector<int> attempts(targets.size(), 0);

    for (size_t round = 0;; ++round)
    {
        std::vector<FillJob> jobs;
        for (size_t t = 0; t < targets.size(); ++t)
        {
            const PopulationItems &population = *targets[t].population;
            const bool empty = population.empty();
            for (size_t i = population.size();
                 i < targetSize && attempts[t] < maxAttempts; ++i, ++attempts[t])
            {
                jobs.push_back({t,
                                deterministicSeed(config_.seed, targets[t].streamId,
                                                  static_cast<uint64_t>(generation),
                                                  round * targetSize + i),
                                empty && i == 0});
            }
        }
        if (jobs.empty())
        {
            break;
        }

        std::vector<std::optional<std::pair<Solution, int>>> results(jobs.size());
        cndetector::parallelFor(0, static_cast<int>(jobs.size()),
                          [&](int k, int /*workerIdx*/)
                          {
                              if (!jobs[k].mandatory && reachedDeadline())
                              {
                                  return;
                              }
                              results[k] = generatePPIIndividual(
                                  targets[jobs[k].target].budget, jobs[k].seed);
                          });

        for (size_t k = 0; k < jobs.size(); ++k)
        {
            if (!results[k].has_value())
            {
                continue;
            }
            PopulationItems &population = *targets[jobs[k].target].population;
            if (population.size() >= targetSize
                || isDuplicate(results[k]->first, population))
            {
                continue;
            }
            addSolution(population, results[k]->first, results[k]->second);
        }

        if (reachedDeadline())
        {
            break;
        }
    }
}

template <typename GraphT>
GenerationEvent DualPopulationT<GraphT>::buildGenerationEvent(int generation) const
{
    return {
        generation,
        elapsedSeconds(),
        getBestItem(mainPopulation_).objValue,
        static_cast<int>(mainPopulation_.size()),
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
std::pair<Solution, int> DualPopulationT<GraphT>::generatePPIIndividual(int budget, int seed) const
{
    std::unique_ptr<GraphT> graph;
    if (budget == mainBudget_)
    {
        graph = originalGraph_.getRandomFullBudgetGraph(deterministicSeed(seed, 0x4001u));
    }
    else
    {
        graph = originalGraph_.getRandomPartialGraph(budget, deterministicSeed(seed, 0x4002u));
    }
    const LocalSearchResult result
        = runLocalSearch(*graph, deterministicSeed(seed, 0x4003u), search_, deadline_);
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
    std::unique_ptr<GraphT> offspringGraph;
    if constexpr (std::is_same_v<GraphT, CNP_Graph>)
    {
        if (crossover_ == CrossoverStrategy::DBX)
        {
            offspringGraph = doubleBackboneBasedCrossover(
                originalGraph_, parentPtrs, targetBudget.value_or(mainBudget_), seed);
        }
    }
    if (!offspringGraph)
    {
        offspringGraph = reduceSolveCombine(
            originalGraph_, parentPtrs, config_.backboneRate, targetBudget, seed,
            search_, deadline_);
    }
    const LocalSearchResult result
        = runLocalSearch(*offspringGraph, seed + 10000, search_, deadline_);
    return {result.solution, result.objValue};
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::completePartialSolution(
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
    else if (targetBudget == mainBudget_)
    {
        reducedGraph = workingGraph->getRandomFullBudgetGraph(deterministicSeed(seed, 0x6001u));
    }
    else
    {
        reducedGraph = workingGraph->getRandomPartialGraph(
            targetBudget - static_cast<int>(baseSolution.size()),
            deterministicSeed(seed, 0x6002u));
    }

    LocalSearchResult result
        = runLocalSearch(*reducedGraph, deterministicSeed(seed, 0x6003u), search_, deadline_);
    Solution finalNodes = baseSolution;
    finalNodes.reserve(baseSolution.size() + result.solution.size());
    finalNodes.insert(result.solution.begin(), result.solution.end());

    auto improvedGraph = originalGraph_.clone();
    improvedGraph->updateGraphByRemovedNodes(finalNodes);
    result = runLocalSearch(*improvedGraph, deterministicSeed(seed, 0x6004u), search_, deadline_);
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
const PopulationItem &DualPopulationT<GraphT>::getWorstItem(
    const PopulationItems &population) const
{
    if (population.empty())
    {
        throw std::runtime_error("Population is empty");
    }
    return *std::max_element(population.begin(), population.end(),
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
        population[i].fitness = QUALITY_WEIGHT * costRanks[i]
            + (1.0 - QUALITY_WEIGHT) * diversityRanks[i];
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
void DualPopulationT<GraphT>::applyQDPU(
    PopulationItems &population,
    const Solution &solution,
    int objValue,
    size_t maxPopulationSize)
{
    // An offspring identical to an individual of the population is always
    // discarded. Once the population is full, so is an offspring whose cost is
    // worse than the cost of the current worst individual; any other offspring
    // is admitted in place of the individual with the worst quality-diversity
    // fitness.
    if (isDuplicate(solution, population))
    {
        return;
    }
    if (population.size() >= maxPopulationSize
        && objValue > getWorstItem(population).objValue)
    {
        return;
    }

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
    if (population.empty())
    {
        throw std::runtime_error("Cannot select parents from an empty population");
    }
    if (population.size() == 1)
    {
        // Instances with a single distinct local optimum leave the population
        // with one individual; recombine it with itself.
        return {population[0].solution, population[0].solution};
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
std::vector<GenerationEvent> DualPopulationT<GraphT>::drainGenerationEvents()
{
    std::vector<GenerationEvent> drained;
    drained.swap(generationEvents_);
    return drained;
}

template <typename GraphT>
std::vector<HPCEvent> DualPopulationT<GraphT>::drainHPCEvents()
{
    std::vector<HPCEvent> drained;
    drained.swap(hpcEvents_);
    return drained;
}

template <typename GraphT>
std::pair<Solution, int> DualPopulationT<GraphT>::getBestSolution() const
{
    const auto &bestItem = getBestItem(mainPopulation_);
    return {*bestItem.solution, bestItem.objValue};
}

template <typename GraphT>
std::vector<std::pair<Solution, int>> DualPopulationT<GraphT>::getMainPopulation() const
{
    std::vector<std::pair<Solution, int>> snapshot;
    snapshot.reserve(mainPopulation_.size());
    for (const auto &item : mainPopulation_)
    {
        snapshot.emplace_back(*item.solution, item.objValue);
    }
    return snapshot;
}

template <typename GraphT>
int DualPopulationT<GraphT>::getMainPopulationSize() const
{
    return static_cast<int>(mainPopulation_.size());
}

template <typename GraphT>
int DualPopulationT<GraphT>::getMainGenerationCount() const
{
    return mainGenerationCount_;
}

template <typename GraphT>
int DualPopulationT<GraphT>::getAuxiliaryGenerationCount() const
{
    return auxiliaryGenerationCount_;
}

template class DualPopulationT<CNP_Graph>;
template class DualPopulationT<DCNP_Graph>;
