#include "CNP_Graph.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

CNP_Graph::CNP_Graph(NodeSet nodes,
                     std::vector<NodeSet> adjList,
                     int budget,
                     int seed)
{
    numNodes_ = static_cast<int>(nodes.size());
    originalNodesSet_ = std::move(nodes);
    originalAdjList_ = std::move(adjList);
    nodeAge_.resize(numNodes_, 0);
    numToRemove_ = budget;
    nodeToComponentIndex_.resize(numNodes_, -1);
    rng_.setSeed(seed);
    dfsVisitEpoch_.resize(numNodes_, 0);
    dfsStack_.reserve(numNodes_);
    nodeToIdxScratch_.assign(static_cast<size_t>(numNodes_), -1);
    compSeenEpoch_.assign(static_cast<size_t>(numNodes_), 0);
    rebuildCSR();
}

void CNP_Graph::rebuildCSR()
{
    // Flatten originalAdjList_ into CSR (offsets + concatenated neighbors).
    csrOffset_.assign(static_cast<size_t>(numNodes_) + 1, 0);
    for (int v = 0; v < numNodes_; ++v)
    {
        csrOffset_[v + 1]
            = csrOffset_[v] + static_cast<int>(originalAdjList_[v].size());
    }
    csrAdj_.resize(static_cast<size_t>(csrOffset_[numNodes_]));
    for (int v = 0; v < numNodes_; ++v)
    {
        size_t pos = static_cast<size_t>(csrOffset_[v]);
        for (Node neighbor : originalAdjList_[v])
        {
            csrAdj_[pos++] = neighbor;
        }
    }

    // Resync the O(1) removed flag from the authoritative removedNodes set.
    removedFlag_.assign(static_cast<size_t>(numNodes_), 0);
    for (Node node : removedNodes)
    {
        removedFlag_[node] = 1;
    }
}

int CNP_Graph::componentContribution(size_t size) const
{
    // CNP1 pairwise connectivity.
    return static_cast<int>((size * (size - 1)) / 2);
}

int CNP_Graph::nextDfsEpoch() const
{
    if (dfsCurrentEpoch_ == std::numeric_limits<int>::max())
    {
        std::fill(dfsVisitEpoch_.begin(), dfsVisitEpoch_.end(), 0);
        dfsCurrentEpoch_ = 0;
    }
    return ++dfsCurrentEpoch_;
}

void CNP_Graph::initializeComponentsAndMapping()
{
    std::fill(nodeToComponentIndex_.begin(), nodeToComponentIndex_.end(), -1);

    connectedComponents_.clear();
    connectedPairs_ = 0;

    const int epoch = nextDfsEpoch();

    for (Node node : originalNodesSet_)
    {
        if (dfsVisitEpoch_[node] == epoch || removedFlag_[node])
        {
            continue;
        }

        Component component;
        dfsCollect(node, epoch, component);
        if (component.nodes.empty())
        {
            continue;
        }

        const auto componentIndex
            = static_cast<ComponentIndex>(connectedComponents_.size());
        for (Node componentNode : component.nodes)
        {
            nodeToComponentIndex_[componentNode] = componentIndex;
        }
        connectedPairs_ += componentContribution(component.size());
        connectedComponents_.push_back(std::move(component));
    }
}

void CNP_Graph::dfsCollect(Node startNode, int epoch, Component &out) const
{
    dfsStack_.clear();
    dfsStack_.push_back(startNode);

    while (!dfsStack_.empty())
    {
        const Node node = dfsStack_.back();
        dfsStack_.pop_back();

        if (dfsVisitEpoch_[node] == epoch || removedFlag_[node])
        {
            continue;
        }

        dfsVisitEpoch_[node] = epoch;
        out.nodes.push_back(node);

        const int rowEnd = csrOffset_[node + 1];
        for (int idx = csrOffset_[node]; idx < rowEnd; ++idx)
        {
            const Node neighbor = csrAdj_[idx];
            if (dfsVisitEpoch_[neighbor] != epoch && !removedFlag_[neighbor])
            {
                dfsStack_.push_back(neighbor);
            }
        }
    }
}

Component CNP_Graph::dfsFindComponent(Node startNode) const
{
    Component newComponent;
    dfsCollect(startNode, nextDfsEpoch(), newComponent);
    return newComponent;
}

void CNP_Graph::updateGraphByRemovedNodes(const NodeSet &nodesToRemove)
{
    removedNodes = nodesToRemove;

    std::fill(removedFlag_.begin(), removedFlag_.end(), static_cast<uint8_t>(0));
    for (Node node : removedNodes)
    {
        removedFlag_[node] = 1;
    }

    initializeComponentsAndMapping();
}

void CNP_Graph::getReducedGraphByRemovedNodes(const NodeSet &removeSet)
{
    removedNodes.clear();
    numToRemove_ -= static_cast<int>(removeSet.size());

    for (Node node : removeSet)
    {
        originalNodesSet_.erase(node);

        for (Node neighbor : originalAdjList_[node])
        {
            originalAdjList_[neighbor].erase(node);
        }
        originalAdjList_[node].clear();
    }

    rebuildCSR();
    initializeComponentsAndMapping();
}

void CNP_Graph::addNode(Node nodeToAdd)
{
    if (!isNodeRemoved(nodeToAdd))
    {
        return;
    }

    removedNodes.erase(nodeToAdd);
    removedFlag_[nodeToAdd] = 0;

    // Collect the distinct components adjacent to the re-added node.
    if (compSeenCurrentEpoch_ == std::numeric_limits<int>::max())
    {
        std::fill(compSeenEpoch_.begin(), compSeenEpoch_.end(), 0);
        compSeenCurrentEpoch_ = 0;
    }
    const int seenEpoch = ++compSeenCurrentEpoch_;

    mergeSlotsScratch_.clear();
    bool mappingStale = false;
    const int rowEnd = csrOffset_[nodeToAdd + 1];
    for (int idx = csrOffset_[nodeToAdd]; idx < rowEnd; ++idx)
    {
        const Node neighbor = csrAdj_[idx];
        if (removedFlag_[neighbor])
        {
            continue;
        }
        const ComponentIndex ci = nodeToComponentIndex_[neighbor];
        if (ci < 0)
        {
            mappingStale = true;
            break;
        }
        if (compSeenEpoch_[ci] != seenEpoch)
        {
            compSeenEpoch_[ci] = seenEpoch;
            mergeSlotsScratch_.push_back(ci);
        }
    }

    if (mappingStale)
    {
        // Component data was never built for this state; fall back to the
        // wholesale rebuild rather than merging against a stale mapping.
        initializeComponentsAndMapping();
        return;
    }

    if (mergeSlotsScratch_.empty())
    {
        // Isolated node: becomes its own component.
        nodeToComponentIndex_[nodeToAdd]
            = static_cast<ComponentIndex>(connectedComponents_.size());
        Component component;
        component.nodes.push_back(nodeToAdd);
        connectedPairs_ += componentContribution(1);
        connectedComponents_.push_back(std::move(component));
        return;
    }

    // Merge all adjacent components (plus the node) into one, swapping their
    // objective contributions for the merged component's contribution.
    std::sort(mergeSlotsScratch_.begin(), mergeSlotsScratch_.end());
    const ComponentIndex keep = mergeSlotsScratch_.front();

    mergedNodesScratch_.clear();
    mergedNodesScratch_.push_back(nodeToAdd);
    for (ComponentIndex ci : mergeSlotsScratch_)
    {
        const auto &nodes = connectedComponents_[ci].nodes;
        connectedPairs_ -= componentContribution(nodes.size());
        mergedNodesScratch_.insert(
            mergedNodesScratch_.end(), nodes.begin(), nodes.end());
    }
    connectedPairs_ += componentContribution(mergedNodesScratch_.size());

    // Free the vacated slots in descending order (keep the smallest slot for
    // the merged component). Descending order guarantees the component moved
    // in from the back is never one of the still-pending vacated slots.
    for (size_t k = mergeSlotsScratch_.size(); k-- > 1;)
    {
        const ComponentIndex slot = mergeSlotsScratch_[k];
        const auto last
            = static_cast<ComponentIndex>(connectedComponents_.size() - 1);
        if (slot != last)
        {
            connectedComponents_[slot] = std::move(connectedComponents_[last]);
            for (Node moved : connectedComponents_[slot].nodes)
            {
                nodeToComponentIndex_[moved] = slot;
            }
        }
        connectedComponents_.pop_back();
    }

    connectedComponents_[keep].nodes.swap(mergedNodesScratch_);
    for (Node member : connectedComponents_[keep].nodes)
    {
        nodeToComponentIndex_[member] = keep;
    }
}

void CNP_Graph::removeNode(Node nodeToRemove)
{
    if (isNodeRemoved(nodeToRemove))
    {
        return;
    }

    const ComponentIndex oldComponent = nodeToComponentIndex_[nodeToRemove];

    removedNodes.insert(nodeToRemove);
    removedFlag_[nodeToRemove] = 1;

    if (oldComponent < 0)
    {
        // Component data was never built for this state; fall back to the
        // wholesale rebuild rather than splitting against a stale mapping.
        initializeComponentsAndMapping();
        return;
    }

    nodeToComponentIndex_[nodeToRemove] = -1;
    connectedPairs_
        -= componentContribution(connectedComponents_[oldComponent].size());

    // Re-partition the old component: every remaining node of it is reachable
    // from some neighbor of the removed node, and the DFS cannot escape the
    // old component (it was maximal). One shared epoch dedupes neighbors that
    // land in the same sub-component.
    const int epoch = nextDfsEpoch();
    bool reusedSlot = false;
    const int rowEnd = csrOffset_[nodeToRemove + 1];
    for (int idx = csrOffset_[nodeToRemove]; idx < rowEnd; ++idx)
    {
        const Node neighbor = csrAdj_[idx];
        if (removedFlag_[neighbor] || dfsVisitEpoch_[neighbor] == epoch)
        {
            continue;
        }

        Component subComponent;
        dfsCollect(neighbor, epoch, subComponent);

        ComponentIndex slot;
        if (!reusedSlot)
        {
            slot = oldComponent;
            connectedComponents_[slot] = std::move(subComponent);
            reusedSlot = true;
        }
        else
        {
            slot = static_cast<ComponentIndex>(connectedComponents_.size());
            connectedComponents_.push_back(std::move(subComponent));
        }

        for (Node member : connectedComponents_[slot].nodes)
        {
            nodeToComponentIndex_[member] = slot;
        }
        connectedPairs_
            += componentContribution(connectedComponents_[slot].size());
    }

    if (!reusedSlot)
    {
        // The removed node was isolated in its component: drop the slot.
        const auto last
            = static_cast<ComponentIndex>(connectedComponents_.size() - 1);
        if (oldComponent != last)
        {
            connectedComponents_[oldComponent]
                = std::move(connectedComponents_[last]);
            for (Node moved : connectedComponents_[oldComponent].nodes)
            {
                nodeToComponentIndex_[moved] = oldComponent;
            }
        }
        connectedComponents_.pop_back();
    }
}

void CNP_Graph::removeBestNodesFromSet(const NodeSet &candidateNodes)
{
    initializeComponentsAndMapping();

    if (static_cast<int>(removedNodes.size()) >= numToRemove_)
    {
        return;
    }

    std::vector<Node> eligibleNodes;
    eligibleNodes.reserve(candidateNodes.size());

    for (Node node : candidateNodes)
    {
        if (originalNodesSet_.find(node) == originalNodesSet_.end())
        {
            throw std::invalid_argument("candidate node is not present in the graph");
        }
        if (!isNodeRemoved(node))
        {
            eligibleNodes.push_back(node);
        }
    }

    const auto remainingBudget
        = static_cast<size_t>(numToRemove_ - static_cast<int>(removedNodes.size()));
    if (eligibleNodes.size() < remainingBudget)
    {
        throw std::invalid_argument(
            "candidate node set does not contain enough available nodes to reach budget");
    }

    while (static_cast<int>(removedNodes.size()) < numToRemove_)
    {
        bool found = false;
        Node bestNode = INVALID_NODE;
        int bestObjectiveValue = 0;

        for (Node node : eligibleNodes)
        {
            if (isNodeRemoved(node))
            {
                continue;
            }

            const int objectiveValue = getObjectiveValue()
                - calculateObjectiveDecreaseByRemovingNodeInternal(node);

            if (!found || objectiveValue < bestObjectiveValue)
            {
                found = true;
                bestNode = node;
                bestObjectiveValue = objectiveValue;
            }
        }

        if (!found)
        {
            throw std::runtime_error(
                "failed to select a candidate node while filling removal budget");
        }

        removeNode(bestNode);
        std::erase(eligibleNodes, bestNode);
    }
}

int CNP_Graph::calculateObjectiveDecreaseByRemovingNodeInternal(Node node)
{
    const int objectiveBefore = getObjectiveValue();
    removeNode(node);
    const int objectiveAfter = getObjectiveValue();
    addNode(node);
    return objectiveBefore - objectiveAfter;
}

ComponentIndex CNP_Graph::selectRemovedComponent() const
{
    size_t numComponents = connectedComponents_.size();
    std::vector<ComponentIndex> largeComponents;
    largeComponents.reserve(numComponents);

    if (numComponents > 50)
    {
        return selectRemovedLargerComponent();
    }

    int minSize = numNodes_;
    int maxSize = 0;

    for (size_t i = 0; i < numComponents; ++i)
    {
        const size_t size = connectedComponents_[i].size();
        minSize = std::min(minSize, static_cast<int>(size));
        maxSize = std::max(maxSize, static_cast<int>(size));
    }

    // Definition 1 (large connected component): C is large when
    // |C| >= (|C_max| + |C_min|) / 2, taken over all components of the
    // residual graph.
    const double sizeThreshold = (maxSize + minSize) / 2.0;

    for (size_t i = 0; i < numComponents; ++i)
    {
        if (connectedComponents_[i].size() >= sizeThreshold)
        {
            largeComponents.push_back(i);
        }
    }

    if (largeComponents.empty())
    {
        ComponentIndex fallbackIndex = 0;
        size_t fallbackSize = 0;
        for (size_t i = 0; i < numComponents; ++i)
        {
            const size_t size = connectedComponents_[i].size();
            if (size > fallbackSize)
            {
                fallbackSize = size;
                fallbackIndex = static_cast<ComponentIndex>(i);
            }
        }
        return fallbackIndex;
    }

    return largeComponents[rng_.generateIndex(largeComponents.size())];
}

ComponentIndex CNP_Graph::selectRemovedLargerComponent() const
{
    size_t totalSize = numNodes_ - removedNodes.size();
    size_t numComponents = connectedComponents_.size();

    // Compute the average component size.
    size_t avgComponentSize = std::max(
        static_cast<size_t>(2),
        static_cast<size_t>(std::round(static_cast<float>(totalSize)
                                    / static_cast<float>(numComponents))));

    std::vector<ComponentIndex> largeComponents;
    std::vector<size_t> componentSizes;
    largeComponents.reserve(numComponents);
    componentSizes.reserve(numComponents);

    size_t totalNodesInBigComponents = 0;
    size_t maxSize = 0;
    size_t maxIndex = 0;
    size_t secondMaxSize = 0;
    size_t secondMaxIndex = 0;


    for (size_t i = 0; i < numComponents; ++i)
    {
        const size_t currentSize = connectedComponents_[i].size();

        if (currentSize > avgComponentSize)
        {
            largeComponents.push_back(i);
            componentSizes.push_back(currentSize);
            totalNodesInBigComponents += currentSize;

            if (currentSize > maxSize)
            {
                secondMaxSize = maxSize;
                secondMaxIndex = maxIndex;
                maxSize = currentSize;
                maxIndex = i;
            }
            else if (currentSize > secondMaxSize)
            {
                secondMaxSize = currentSize;
                secondMaxIndex = i;
            }
        }
    }



    if (largeComponents.empty())
    {
        // Fallback to component with maximum size when heuristic set is empty.
        size_t fallbackIdx = 0;
        size_t fallbackSize = 0;
        for (size_t i = 0; i < connectedComponents_.size(); ++i)
        {
            const size_t currentSize = connectedComponents_[i].size();
            if (currentSize > fallbackSize)
            {
                fallbackSize = currentSize;
                fallbackIdx = i;
            }
        }
        if (fallbackSize == 0)
        {
            throw std::runtime_error("no components available for selection");
        }
        return static_cast<ComponentIndex>(fallbackIdx);
    }

    if (largeComponents.size() == 1)
    {
        return rng_.generateBool(0.5) ? secondMaxIndex : largeComponents[0];
    }

    const int index = rng_.generateIndex(totalNodesInBigComponents);
    int sum = 0;

    for (size_t i = 0; i < largeComponents.size(); ++i)
    {
        sum += componentSizes[i];
        if (index < sum)
        {
            return largeComponents[i];
        }
    }

    return largeComponents.back();
}

Node CNP_Graph::randomSelectNodeFromComponent(
    ComponentIndex componentIndex) const
{
    const auto &component = connectedComponents_[componentIndex];
    if (component.nodes.empty())
    {
        throw std::runtime_error("component is empty, can not select node");
    }
    return component.nodes[rng_.generateIndex(component.size())];
}

Node CNP_Graph::ageSelectNodeFromComponent(ComponentIndex componentIndex) const
{
    const auto &component = connectedComponents_[componentIndex];
    std::vector<Node> candidateNodes;
    candidateNodes.reserve(component.size());

    const Node firstNode = component.nodes[0];
    Age minAge = nodeAge_[firstNode];
    candidateNodes.push_back(firstNode);

    for (size_t i = 1; i < component.size(); ++i)
    {
        Node currentNode = component.nodes[i];
        if (nodeAge_[currentNode] < minAge)
        {
            minAge = nodeAge_[currentNode];
            candidateNodes.clear();
            candidateNodes.push_back(currentNode);
        }
        else if (nodeAge_[currentNode] == minAge)
        {
            candidateNodes.push_back(currentNode);
        }
    }

    return candidateNodes.size() == 1
                ? candidateNodes[0]
                : candidateNodes[rng_.generateIndex(candidateNodes.size())];
}

Node CNP_Graph::impactSelectNodeFromComponent(
    ComponentIndex componentIndex) const
{
    const auto &component = connectedComponents_[componentIndex];
    if (component.size() == 0)
    {
        throw std::runtime_error("component is empty, can not select node");
    }

    for (size_t i = 0; i < component.size(); ++i)
    {
        nodeToIdxScratch_[component.nodes[i]] = static_cast<int>(i);
    }

    const int size = static_cast<int>(component.size());
    dfn_.assign(size + 1, 0);
    lowVec_.assign(size + 1, 0);
    stSizeVec_.assign(size + 1, 1);
    cutSizeVec_.assign(size + 1, 1);
    impactVec_.assign(size + 1, 0);
    flagVec_.assign(size + 1, 0);
    isCutVec_.assign(size + 1, false);

    std::vector<Node> candidateNodes;
    candidateNodes.reserve(size);

    timeStamp_ = 0;
    nodeRoot_ = 1;

    tarjanInComponent(componentIndex);

    int minImpact = std::numeric_limits<int>::max();
    candidateNodes.clear();

    for (int i = 1; i <= size; ++i)
    {
        int currentImpact = impactVec_[i];

        if (isCutVec_[i])
        {
            currentImpact += ((timeStamp_ - cutSizeVec_[i])
                              * (timeStamp_ - cutSizeVec_[i] - 1))
                             / 2;
        }
        else
        {
            currentImpact += ((timeStamp_ - 1) * (timeStamp_ - 2)) / 2;
        }

        if (currentImpact < minImpact)
        {
            minImpact = currentImpact;
            candidateNodes.clear();
            candidateNodes.push_back(component.nodes[i - 1]);
        }
        else if (currentImpact == minImpact)
        {
            candidateNodes.push_back(component.nodes[i - 1]);
        }
    }

    // Reset only the entries this call touched.
    for (Node member : component.nodes)
    {
        nodeToIdxScratch_[member] = -1;
    }

    return candidateNodes.size() == 1
               ? candidateNodes[0]
               : candidateNodes[rng_.generateIndex(candidateNodes.size())];
}

void CNP_Graph::tarjanInComponent(ComponentIndex compIndex) const
{
    struct Frame
    {
        int nodeIdx;   ///< Local 1-based Tarjan index.
        int edgePos;   ///< Next CSR position to inspect.
        int edgeEnd;   ///< CSR row end.
    };

    const auto &componentNodes = connectedComponents_[compIndex].nodes;

    std::vector<Frame> stack;
    stack.reserve(componentNodes.size());

    const Node rootId = componentNodes[nodeRoot_ - 1];
    dfn_[nodeRoot_] = lowVec_[nodeRoot_] = ++timeStamp_;
    stack.push_back({nodeRoot_, csrOffset_[rootId], csrOffset_[rootId + 1]});

    while (!stack.empty())
    {
        Frame &frame = stack.back();

        if (frame.edgePos < frame.edgeEnd)
        {
            const Node neighbor = csrAdj_[frame.edgePos++];
            if (removedFlag_[neighbor]
                || nodeToComponentIndex_[neighbor] != compIndex)
            {
                continue;
            }

            const int neighborIdx = nodeToIdxScratch_[neighbor] + 1;
            if (dfn_[neighborIdx] == 0)
            {
                dfn_[neighborIdx] = lowVec_[neighborIdx] = ++timeStamp_;
                stack.push_back({neighborIdx,
                                 csrOffset_[neighbor],
                                 csrOffset_[neighbor + 1]});
            }
            else
            {
                lowVec_[frame.nodeIdx]
                    = std::min(lowVec_[frame.nodeIdx], dfn_[neighborIdx]);
            }
        }
        else
        {
            const int childIdx = frame.nodeIdx;
            stack.pop_back();
            if (stack.empty())
            {
                break;
            }

            const int parentIdx = stack.back().nodeIdx;

            lowVec_[parentIdx] = std::min(lowVec_[parentIdx], lowVec_[childIdx]);
            stSizeVec_[parentIdx] += stSizeVec_[childIdx];

            if (lowVec_[childIdx] >= dfn_[parentIdx])
            {
                flagVec_[parentIdx]++;

                if (parentIdx != nodeRoot_)
                {
                    isCutVec_[parentIdx] = true;
                    cutSizeVec_[parentIdx] += stSizeVec_[childIdx];
                    impactVec_[parentIdx] += (stSizeVec_[childIdx]
                                              * (stSizeVec_[childIdx] - 1))
                                             / 2;
                }
                else if (flagVec_[parentIdx] > 1)
                {
                    isCutVec_[parentIdx] = true;
                }
            }
        }
    }
}

Node CNP_Graph::greedySelectNodeToAdd() const
{
    if (removedNodes.empty())
    {
        return INVALID_NODE;
    }

    std::vector<Node> candidates(removedNodes.begin(), removedNodes.end());
    const size_t numCandidates = candidates.size();

    std::vector<int> gains(numCandidates);
    std::vector<size_t> componentSizes(connectedComponents_.size(), 0);
    for (size_t i = 0; i < numCandidates; ++i)
    {
        gains[i] = calculateConnectionGain(candidates[i], componentSizes);
    }

    int minDelta = gains[0];
    std::vector<Node> candidateNodes;
    candidateNodes.reserve(numCandidates);
    candidateNodes.push_back(candidates[0]);

    for (size_t i = 1; i < numCandidates; ++i)
    {
        if (gains[i] < minDelta)
        {
            minDelta = gains[i];
            candidateNodes.clear();
            candidateNodes.push_back(candidates[i]);
        }
        else if (gains[i] == minDelta)
        {
            candidateNodes.push_back(candidates[i]);
        }
    }

    return candidateNodes.size() == 1
               ? candidateNodes[0]
               : candidateNodes[rng_.generateIndex(candidateNodes.size())];
}

Node CNP_Graph::randomSelectNodeToRemove() const
{
    ComponentIndex compIndex = rng_.generateIndex(connectedComponents_.size());
    const Component &selectedComponent = connectedComponents_[compIndex];

    if (selectedComponent.nodes.empty())
    {
        throw std::runtime_error("selected component is empty, can not select node");
    }

    return selectedComponent.nodes[rng_.generateIndex(selectedComponent.size())];
}

int CNP_Graph::calculateConnectionGain(Node node, std::vector<size_t>& componentSizes) const
{
    int totalSize = 1;
    int oldConnectionsSum = 0;

    const int rowEnd = csrOffset_[node + 1];

    // Pass 1: Sum up sizes and mark components as visited
    for (int idx = csrOffset_[node]; idx < rowEnd; ++idx)
    {
        const Node neighbor = csrAdj_[idx];
        if (nodeToComponentIndex_[neighbor] != -1)
        {
            ComponentIndex compIndex = nodeToComponentIndex_[neighbor];

            // Only add if we haven't seen this component yet for this node
            if (componentSizes[compIndex] == 0)
            {
                size_t size = connectedComponents_[compIndex].size();
                componentSizes[compIndex] = size;
                totalSize += size;

                // Accumulate the objective contributed by this component
                // before the merge, in the same pass to avoid scanning all
                // components.
                oldConnectionsSum += componentContribution(size);
            }
        }
    }

    // Pass 2: Reset the componentSizes array for the indices we touched
    // This avoids the O(N) memset/fill in the caller
    for (int idx = csrOffset_[node]; idx < rowEnd; ++idx)
    {
        const Node neighbor = csrAdj_[idx];
        if (nodeToComponentIndex_[neighbor] != -1)
        {
            componentSizes[nodeToComponentIndex_[neighbor]] = 0;
        }
    }

    // Objective contributed by the single merged component the node joins.
    const int newConnections
        = componentContribution(static_cast<size_t>(totalSize));

    return newConnections - oldConnectionsSum;
}

std::unique_ptr<CNP_Graph> CNP_Graph::getRandomFeasibleGraph(int seed) const
{
    auto tempGraph = std::make_unique<CNP_Graph>(*this);
    RandomNumberGenerator rng;
    rng.setSeed(seed);
    Solution randomSolution;
    std::vector<Node> availableNodes(originalNodesSet_.begin(),
                                     originalNodesSet_.end());

    int nodesToRemove = std::min(numToRemove_, static_cast<int>(availableNodes.size()));

    for (int i = 0; i < nodesToRemove; ++i)
    {
        int randomIndex = i + rng.generateIndex(static_cast<int>(availableNodes.size()) - i);
        std::swap(availableNodes[i], availableNodes[randomIndex]);
        randomSolution.insert(availableNodes[i]);
    }

    tempGraph->updateGraphByRemovedNodes(randomSolution);
    return tempGraph;
}

std::unique_ptr<CNP_Graph> CNP_Graph::getRandomPartialGraph(
    int partialBudget, int seed) const
{
    auto tempGraph = std::make_unique<CNP_Graph>(*this);
    RandomNumberGenerator rng;
    rng.setSeed(seed);
    Solution randomSolution;
    std::vector<Node> availableNodes(originalNodesSet_.begin(),
                                    originalNodesSet_.end());

    int nodesToRemove = std::min(partialBudget, static_cast<int>(availableNodes.size()));

    for (int i = 0; i < nodesToRemove; ++i)
    {
        int randomIndex = i + rng.generateIndex(static_cast<int>(availableNodes.size()) - i);
        std::swap(availableNodes[i], availableNodes[randomIndex]);
        randomSolution.insert(availableNodes[i]);
    }

    tempGraph->updateGraphByRemovedNodes(randomSolution);
    return tempGraph;
}

std::unique_ptr<CNP_Graph> CNP_Graph::clone() const
{
    return std::make_unique<CNP_Graph>(*this);
}

bool CNP_Graph::isNodeRemoved(Node node) const
{
    // removedFlag_ mirrors removedNodes and is O(1) / cache-friendly; it is
    // resynced at every mutation point (rebuildCSR, add/removeNode,
    // updateGraphByRemovedNodes).
    return removedFlag_[node] != 0;
}

int CNP_Graph::getNumNodes() const
{
    return static_cast<int>(originalNodesSet_.size());
}

const NodeSet& CNP_Graph::getRemovedNodes() const
{
    return removedNodes;
}

void CNP_Graph::setNodeAge(Node node, Age age)
{
    nodeAge_[node] = age;
}

int CNP_Graph::getObjectiveValue() const
{
    return connectedPairs_;
}
