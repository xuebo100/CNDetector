
#include "DCNP_Graph.h"
#include <algorithm>
#include <limits>

DCNP_Graph::DCNP_Graph(NodeSet nodes,
                    int K,
                    std::vector<NodeSet> adjList,
                    int numToRemove,
                    int seed)
{
    numNodes_ = static_cast<int>(nodes.size());
    kHops_ = K;
    originalNodesSet_ = std::move(nodes);
    adjList_ = std::move(adjList);
    numToRemove_ = numToRemove;
    rng_.setSeed(seed);

    nodeAge_.assign(numNodes_, 0);
    treeMembers_.assign(static_cast<size_t>(numNodes_), std::vector<Node>());
    treeSize_.assign(static_cast<size_t>(numNodes_), 0);
    totalTreeSize_ = 0;

    visitEpoch_.assign(static_cast<size_t>(numNodes_), 0);
    bfsLevel_.assign(static_cast<size_t>(numNodes_), 0);
    bfsQueue_.resize(static_cast<size_t>(numNodes_));

    rebuildCSR();
    buildTree();
}

void DCNP_Graph::rebuildCSR()
{
    // Flatten adjList_ into CSR (offsets + concatenated neighbors).
    csrOffset_.assign(static_cast<size_t>(numNodes_) + 1, 0);
    for (int v = 0; v < numNodes_; ++v)
    {
        csrOffset_[v + 1]
            = csrOffset_[v] + static_cast<int>(adjList_[v].size());
    }
    csrAdj_.resize(static_cast<size_t>(csrOffset_[numNodes_]));
    for (int v = 0; v < numNodes_; ++v)
    {
        size_t pos = static_cast<size_t>(csrOffset_[v]);
        for (Node neighbor : adjList_[v])
        {
            csrAdj_[pos++] = neighbor;
        }
    }

    // Resync the O(1) removed flag from the authoritative removedNodes_ set.
    removedFlag_.assign(static_cast<size_t>(numNodes_), 0);
    for (Node node : removedNodes_)
    {
        removedFlag_[node] = 1;
    }
}

uint32_t DCNP_Graph::nextEpoch() const
{
    if (++currentEpoch_ == 0)
    {
        std::fill(visitEpoch_.begin(), visitEpoch_.end(), 0u);
        currentEpoch_ = 1;
    }
    return currentEpoch_;
}

void DCNP_Graph::bfsKTree(Node v)
{
    totalTreeSize_ -= treeSize_[v];
    auto &members = treeMembers_[v];
    members.clear();
    treeSize_[v] = 0;

    if (removedFlag_[v])
    {
        return;
    }

    const uint32_t epoch = nextEpoch();
    size_t head = 0;
    size_t tail = 0;
    bfsQueue_[tail++] = v;
    visitEpoch_[v] = epoch;
    bfsLevel_[v] = 0;

    while (head < tail)
    {
        const Node currentNode = bfsQueue_[head++];

        if (bfsLevel_[currentNode] < kHops_)
        {
            const int rowEnd = csrOffset_[currentNode + 1];
            for (int idx = csrOffset_[currentNode]; idx < rowEnd; ++idx)
            {
                const Node neighbor = csrAdj_[idx];
                if (removedFlag_[neighbor] || visitEpoch_[neighbor] == epoch)
                {
                    continue;
                }
                visitEpoch_[neighbor] = epoch;
                bfsLevel_[neighbor] = bfsLevel_[currentNode] + 1;
                bfsQueue_[tail++] = neighbor;
            }
        }
    }

    members.assign(bfsQueue_.begin() + 1, bfsQueue_.begin() + tail);
    treeSize_[v] = static_cast<int>(members.size());
    totalTreeSize_ += treeSize_[v];
}

int DCNP_Graph::bfsTreeSizeOnly(Node v) const
{
    if (removedFlag_[v])
    {
        return 0;
    }

    const uint32_t epoch = nextEpoch();
    size_t head = 0;
    size_t tail = 0;
    bfsQueue_[tail++] = v;
    visitEpoch_[v] = epoch;
    bfsLevel_[v] = 0;

    while (head < tail)
    {
        const Node currentNode = bfsQueue_[head++];

        if (bfsLevel_[currentNode] < kHops_)
        {
            const int rowEnd = csrOffset_[currentNode + 1];
            for (int idx = csrOffset_[currentNode]; idx < rowEnd; ++idx)
            {
                const Node neighbor = csrAdj_[idx];
                if (removedFlag_[neighbor] || visitEpoch_[neighbor] == epoch)
                {
                    continue;
                }
                visitEpoch_[neighbor] = epoch;
                bfsLevel_[neighbor] = bfsLevel_[currentNode] + 1;
                bfsQueue_[tail++] = neighbor;
            }
        }
    }

    return static_cast<int>(tail - 1);
}

void DCNP_Graph::buildTree()
{
    for (int i = 0; i < numNodes_; i++)
    {
        bfsKTree(i);
    }
}

void DCNP_Graph::updateGraphByRemovedNodes(const NodeSet &nodesToRemove)
{
    removedNodes_ = nodesToRemove;

    std::fill(removedFlag_.begin(), removedFlag_.end(), static_cast<uint8_t>(0));
    for (Node node : removedNodes_)
    {
        removedFlag_[node] = 1;
    }

    buildTree();
}

void DCNP_Graph::getReducedGraphByRemovedNodes(const NodeSet &removeSet)
{
    removedNodes_.clear();

    numToRemove_ -= static_cast<int>(removeSet.size());

    for (Node node : removeSet)
    {
        originalNodesSet_.erase(node);

        for (Node neighbor : adjList_[node])
        {
            adjList_[neighbor].erase(node);
        }

        adjList_[node].clear();
    }

    rebuildCSR();
    buildTree();
}

void DCNP_Graph::addNode(Node nodeToAdd)
{
    removedNodes_.erase(nodeToAdd);
    removedFlag_[nodeToAdd] = 0;

    bfsKTree(nodeToAdd);

    // Trees whose reachability changes are exactly those whose K-hop ball now
    // contains nodeToAdd — by symmetry, the members of nodeToAdd's new tree.
    for (Node v : treeMembers_[nodeToAdd])
    {
        bfsKTree(v);
    }
}

void DCNP_Graph::removeNode(Node nodeToRemove)
{
    removedNodes_.insert(nodeToRemove);
    removedFlag_[nodeToRemove] = 1;

    // By symmetry the trees containing nodeToRemove are the members of its own
    // tree. Steal the member list before bfsKTree(nodeToRemove) clears it.
    affectedScratch_.swap(treeMembers_[nodeToRemove]);

    bfsKTree(nodeToRemove);  // zeroes the removed node's own tree

    for (Node v : affectedScratch_)
    {
        bfsKTree(v);
    }
}

int DCNP_Graph::calculateKhopTreeSize() const
{
    // Removed nodes hold treeSize_ == 0, so the running total already covers
    // exactly the active nodes; each unordered pair is counted twice.
    return static_cast<int>(totalTreeSize_ / 2);
}

std::unique_ptr<DCNP_Graph> DCNP_Graph::getRandomFeasibleGraph(int seed) const
{
    auto tempGraph = std::make_unique<DCNP_Graph>(*this);
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    NodeSet nodesToRemove;

    std::vector<Node> availableNodes(originalNodesSet_.begin(),
                                    originalNodesSet_.end());

    int nodesToRemoveCount = std::min(numToRemove_, static_cast<int>(availableNodes.size()));

    for (int i = 0; i < nodesToRemoveCount; ++i)
    {
        int randomIndex = i + rng.generateIndex(static_cast<int>(availableNodes.size()) - i);
        std::swap(availableNodes[i], availableNodes[randomIndex]);
        nodesToRemove.insert(availableNodes[i]);
    }

    tempGraph->updateGraphByRemovedNodes(nodesToRemove);

    return tempGraph;
}

std::unique_ptr<DCNP_Graph> DCNP_Graph::getRandomPartialGraph(
    int partialBudget, int seed) const
{
    auto tempGraph = std::make_unique<DCNP_Graph>(*this);
    RandomNumberGenerator rng;
    rng.setSeed(seed);

    NodeSet nodesToRemove;

    std::vector<Node> availableNodes(originalNodesSet_.begin(),
                                    originalNodesSet_.end());

    int nodesToRemoveCount = std::min(partialBudget, static_cast<int>(availableNodes.size()));

    for (int i = 0; i < nodesToRemoveCount; ++i)
    {
        int randomIndex = i + rng.generateIndex(static_cast<int>(availableNodes.size()) - i);
        std::swap(availableNodes[i], availableNodes[randomIndex]);
        nodesToRemove.insert(availableNodes[i]);
    }

    tempGraph->updateGraphByRemovedNodes(nodesToRemove);

    return tempGraph;
}

Node DCNP_Graph::findBestNodeToRemove()
{
    Node bestNode = INVALID_NODE;
    std::vector<Node> bestList;
    // Deltas are in Σ-treeSize units (2× the objective improvement), which
    // preserves all comparisons without materializing each trial removal.
    long long maxDelta = 0;

    for (Node i = 0; i < numNodes_; i++)
    {
        if (removedFlag_[i])
        {
            continue;
        }

        removedFlag_[i] = 1;

        long long delta = treeSize_[i];
        for (Node v : treeMembers_[i])
        {
            delta += treeSize_[v] - bfsTreeSizeOnly(v);
        }

        removedFlag_[i] = 0;

        if (delta > maxDelta)
        {
            maxDelta = delta;
            bestNode = i;
            bestList.clear();
            bestList.push_back(bestNode);
        }
        else if (delta == maxDelta)
        {
            bestList.push_back(i);
        }
    }

    if (bestList.size() > 1)
    {
        bestNode = bestList[rng_.generateIndex(static_cast<int>(bestList.size()))];
    }

    return bestNode;
}

Node DCNP_Graph::findBestNodeToAdd()
{
    Node bestNode = INVALID_NODE;
    std::vector<Node> bestList;
    long long minDelta = std::numeric_limits<long long>::max();

    for (Node node : removedNodes_)
    {
        removedFlag_[node] = 0;

        const int newSize = bfsTreeSizeOnly(node);
        evalMembersScratch_.assign(bfsQueue_.begin() + 1,
                                   bfsQueue_.begin() + 1 + newSize);

        long long delta = newSize;
        for (Node v : evalMembersScratch_)
        {
            delta += bfsTreeSizeOnly(v) - treeSize_[v];
        }

        removedFlag_[node] = 1;

        if (delta < minDelta)
        {
            minDelta = delta;
            bestNode = node;
            bestList.clear();
            bestList.push_back(bestNode);
        }
        else if (delta == minDelta)
        {
            bestList.push_back(node);
        }
    }

    if (bestList.size() > 1)
    {
        bestNode = bestList[rng_.generateIndex(static_cast<int>(bestList.size()))];
    }

    return bestNode;
}

Node DCNP_Graph::randomSelectNodeToRemove() const
{
    int randomIndex = rng_.generateIndex(numNodes_);
    while (isNodeRemoved(randomIndex))
    {
        randomIndex = rng_.generateIndex(numNodes_);
    }
    return randomIndex;
}

std::unique_ptr<DCNP_Graph> DCNP_Graph::clone() const
{
    return std::make_unique<DCNP_Graph>(*this);
}

void DCNP_Graph::setNodeAge(Node node, Age age)
{
    nodeAge_[node] = age;
}

bool DCNP_Graph::isNodeRemoved(Node node) const
{
    // removedFlag_ mirrors removedNodes_ and is O(1) / cache-friendly; it is
    // resynced at every mutation point (rebuildCSR, add/removeNode,
    // updateGraphByRemovedNodes).
    return removedFlag_[node] != 0;
}

const NodeSet& DCNP_Graph::getRemovedNodes() const
{
    return removedNodes_;
}

int DCNP_Graph::getNumNodes() const
{
    return numNodes_;
}

int DCNP_Graph::getObjectiveValue() const
{
    return calculateKhopTreeSize();
}
