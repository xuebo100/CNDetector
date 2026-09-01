#ifndef DCNP_GRAPH_H
#define DCNP_GRAPH_H

#include "../RandomNumberGenerator.h"
#include "Types.h"
#include <cstdint>
#include <memory>
#include <vector>

/**
 * DCNP_Graph
 *
 * Graph implementation for the Distance-Based Critical Node Problem (DCNP).
 *
 * Represents a graph where the objective is to maximize pairwise distance after node removal.
 */
class DCNP_Graph
{
private:
    int numNodes_ = 0;
    int kHops_ = 0;      ///< K-hop limit.
    int numToRemove_ = 0;
    NodeSet originalNodesSet_;  ///< Set storing all original nodes.
    std::vector<Age> nodeAge_;  ///< Stores the "age" of each node.

    /**
     * Adjacency source of truth. Only mutated by wholesale reductions
     * (getReducedGraphByRemovedNodes); node add/remove during search merely
     * toggles removedFlag_, so no per-move adjacency maintenance is needed.
     */
    std::vector<NodeSet> adjList_;

    /**
     * Flat CSR mirror of adjList_ for the BFS hot path. Rebuilt only when the
     * adjacency wholesale changes (construction, updateGraphByRemovedNodes,
     * getReducedGraphByRemovedNodes). Iterating a flat array beats iterating
     * a vector<unordered_set> during BFS.
     */
    std::vector<int> csrOffset_;   ///< Row starts, size numNodes_+1.
    std::vector<Node> csrAdj_;     ///< Flattened neighbor lists.

    /**
     * O(1) removed-node test mirroring removedNodes_. Kept in sync at every
     * mutation point so the BFS inner loop and isNodeRemoved() avoid a hash
     * lookup per edge.
     */
    std::vector<uint8_t> removedFlag_;

    // Rebuild csrOffset_/csrAdj_ from adjList_ and resync removedFlag_ from
    // removedNodes_. Call after any wholesale adjacency/removed-set change.
    void rebuildCSR();

    /**
     * K-hop tree membership. treeMembers_[v] holds the active nodes
     * (excluding v itself) within kHops_ of v under the current removed set.
     * BFS distance is symmetric in an undirected graph, so
     * u ∈ treeMembers_[v] ⟺ v ∈ treeMembers_[u]; removeNode/addNode rely on
     * that symmetry to enumerate the affected trees without a reverse index.
     */
    std::vector<std::vector<Node>> treeMembers_;

    /**
     * K-hop tree sizes for each node (== treeMembers_[v].size(); 0 for
     * removed nodes).
     */
    std::vector<int> treeSize_;

    /// Running Σ treeSize_ over all nodes (2× the objective value). Updated
    /// by bfsKTree so the objective is O(1) to read.
    long long totalTreeSize_ = 0;

    NodeSet removedNodes_;  ///< Nodes currently removed.

    mutable RandomNumberGenerator rng_;

    /**
     * Per-worker BFS scratch. visitEpoch/currentEpoch implement O(1) logical
     * clears of the visited set (no O(n) fill per BFS). Each parallelFor
     * worker owns one slot, so concurrent evaluation BFS runs never share
     * mutable state.
     */
    struct BfsScratch
    {
        std::vector<uint32_t> visitEpoch;
        uint32_t currentEpoch = 0;
        std::vector<int> level;
        std::vector<Node> queue;
        std::vector<Node> members;  ///< Harvested tree members (eval paths).

        // Size the buffers for an n-node graph (no-op when already sized).
        void ensure(int n);
        // Bump and return the epoch, resetting visitEpoch on wraparound.
        uint32_t nextEpoch();
    };

    /**
     * Lazily grown pool of BfsScratch slots, one per parallelFor worker.
     * Scratch contents are throwaway, so copies of the graph deliberately
     * start with an empty pool instead of duplicating the buffers.
     */
    struct BfsScratchPool
    {
        std::vector<BfsScratch> workers;

        BfsScratchPool() = default;
        BfsScratchPool(const BfsScratchPool & /*other*/) {}
        BfsScratchPool &operator=(const BfsScratchPool & /*other*/)
        {
            return *this;
        }
        BfsScratchPool(BfsScratchPool &&) = default;
        BfsScratchPool &operator=(BfsScratchPool &&) = default;
    };

    mutable BfsScratchPool bfsScratch_;
    mutable std::vector<Node> affectedScratch_;

    // Make sure the pool holds at least `workerCount` slots (buffers inside
    // each slot are sized lazily by the worker that uses it).
    std::vector<BfsScratch> &ensureScratch(int workerCount) const;

    // Recompute v's K-hop tree (members + size) into treeMembers_/treeSize_
    // without touching totalTreeSize_.
    void computeTree(Node v, BfsScratch &scratch);

    // Recompute v's tree via computeTree and keep totalTreeSize_ in sync.
    void bfsKTree(Node v);

    /**
     * Evaluation-only BFS: returns the number of nodes (excluding v) within
     * kHops_ of v, without touching treeMembers_/treeSize_/totalTreeSize_.
     * `forcedRemoved` is treated as removed and `forcedPresent` as active
     * regardless of removedFlag_, so callers can score hypothetical moves
     * without mutating shared state (which keeps this thread-safe). The
     * visited nodes are left in scratch.queue[0 .. ret] with v at index 0.
     */
    int bfsCountFrom(Node v,
                     Node forcedRemoved,
                     Node forcedPresent,
                     BfsScratch &scratch) const;

public:
    DCNP_Graph(NodeSet nodes,
               int K,
               std::vector<NodeSet> adjList,
               int numToRemove,
               int seed);

    DCNP_Graph() = default;

    /**
     * Updates the graph to reflect node removals for DCNP.
     *
     * Parameters
     * ----------
     * nodesToRemove : NodeSet
     *     Set of nodes that have been removed.
     */
    void updateGraphByRemovedNodes(const NodeSet &nodesToRemove);

    /**
     * Creates a reduced graph after node removal for DCNP.
     *
     * Parameters
     * ----------
     * nodesToRemove : NodeSet
     *     Set of nodes to remove.
     *
     * Returns
     * -------
     * DCNP_Graph
     *     The reduced graph.
     */
    void getReducedGraphByRemovedNodes(const NodeSet &nodesToRemove);

    /**
     * Removes a node from the DCNP graph.
     *
     * Parameters
     * ----------
     * node : Node
     *     The node to remove.
     */
    void removeNode(Node node);

    /**
     * Adds a node back to the DCNP graph.
     *
     * Parameters
     * ----------
     * node : Node
     *     The node to add.
     */
    void addNode(Node node);

    /**
     * Sets the age of a node in the DCNP graph.
     *
     * Parameters
     * ----------
     * node : Node
     *     The node.
     * age : Age
     *     The age value.
     */
    void setNodeAge(Node node, Age age);

    /**
     * Checks if a node has been removed from the DCNP graph.
     *
     * Parameters
     * ----------
     * node : Node
     *     The node to check.
     *
     * Returns
     * -------
     * bool
     *     True if the node has been removed.
     */
    bool isNodeRemoved(Node node) const;

    /**
     * Returns the set of removed nodes for DCNP.
     *
     * Returns
     * -------
     * NodeSet
     *     Set of nodes that have been removed.
     */
    const NodeSet& getRemovedNodes() const;

    /**
     * Returns the total number of nodes in the DCNP graph.
     *
     * Returns
     * -------
     * int
     *     Number of nodes.
     */
    int getNumNodes() const;

    /**
     * Calculates the objective value (pairwise distance) for DCNP.
     *
     * Returns
     * -------
     * int
     *     The pairwise distance after node removal.
     */
    int getObjectiveValue() const;

    // Generate a random feasible solution.
    std::unique_ptr<DCNP_Graph> getRandomFeasibleGraph(int seed) const;

    // Generate a random partial solution with specified budget.
    std::unique_ptr<DCNP_Graph> getRandomPartialGraph(int partialBudget, int seed) const;

    // Build/rebuild K-hop tree info for all unremoved nodes.
    void buildTree();

    // Calculate sum of K-hop tree sizes.
    int calculateKhopTreeSize() const;

    // Find best node to remove (heuristic).
    Node findBestNodeToRemove();

    // Find best node to add back (heuristic).
    Node findBestNodeToAdd();

    // Randomly select a node to remove.
    Node randomSelectNodeToRemove() const;

    // Create a deep copy.
    std::unique_ptr<DCNP_Graph> clone() const;
};

#endif  // DCNP_GRAPH_H
