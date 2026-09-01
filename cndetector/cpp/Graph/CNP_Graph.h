#ifndef CNP_GRAPH_H
#define CNP_GRAPH_H

#include "../RandomNumberGenerator.h"
#include "Types.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

/**
 * CNP_Graph
 *
 * Graph implementation for the Critical Node Problem (CNP).
 *
 * Represents a graph where the objective is to minimize connectivity after node removal.
 */
class CNP_Graph
{
private:
    int numNodes_ = 0;          ///< Number of vertices
    NodeSet originalNodesSet_;  ///< Set of all existing nodes
    std::vector<Age> nodeAge_;  ///< Node "age"

    /**
     * Adjacency source of truth. Only mutated by wholesale reductions
     * (getReducedGraphByRemovedNodes); node add/remove during search merely
     * toggles removedFlag_, so no per-move adjacency maintenance is needed.
     */
    std::vector<NodeSet> originalAdjList_;

    /**
     * Flat CSR mirror of originalAdjList_ for the DFS/Tarjan hot paths.
     * Rebuilt only when the adjacency wholesale changes (construction,
     * getReducedGraphByRemovedNodes).
     */
    std::vector<int> csrOffset_;   ///< Row starts, size numNodes_+1.
    std::vector<Node> csrAdj_;     ///< Flattened neighbor lists.

    /**
     * O(1) removed-node test mirroring removedNodes. Kept in sync at every
     * mutation point so DFS inner loops avoid a hash lookup per edge.
     */
    std::vector<uint8_t> removedFlag_;

    int numToRemove_ = 0;
    std::vector<ComponentIndex>
        nodeToComponentIndex_;  ///< Stores the component index for each vertex
    std::vector<Component> connectedComponents_;
    int connectedPairs_ = 0;  ///< Active objective value (CNP1 pairwise connectivity)
    mutable RandomNumberGenerator rng_;

    // Rebuild csrOffset_/csrAdj_ from originalAdjList_ and resync
    // removedFlag_ from removedNodes. Call after wholesale adjacency changes.
    void rebuildCSR();

    // Objective contribution of one component of the given size (CNP1 pairwise
    // connectivity: |C|*(|C|-1)/2).
    int componentContribution(size_t size) const;

    // Selects the larger component to remove.
    ComponentIndex selectRemovedLargerComponent() const;

    // Tarjan algorithm auxiliary variables
    mutable std::vector<int> dfn_;         ///< Node discovery time
    mutable std::vector<int> lowVec_;      ///< Minimum discovery time reachable by node
    mutable std::vector<int> stSizeVec_;   ///< Subtree size
    mutable std::vector<int> cutSizeVec_;  ///< Cut size
    mutable std::vector<int> impactVec_;   ///< Node impact
    mutable std::vector<int> flagVec_;     ///< Flag vector
    mutable std::vector<bool> isCutVec_;   ///< Whether node is a cut vertex
    mutable int timeStamp_;                ///< Timestamp
    mutable int nodeRoot_;                 ///< Root node
    mutable std::vector<int> nodeToIdxScratch_;   ///< node id -> local Tarjan index (reset after use)
    mutable std::vector<int> dfsVisitEpoch_;      ///< Visit epochs for DFS
    mutable int dfsCurrentEpoch_ = 0;             ///< Current epoch id
    mutable std::vector<Node> dfsStack_;          ///< Reusable DFS stack

    // Scratch for incremental component maintenance.
    std::vector<ComponentIndex> mergeSlotsScratch_;  ///< Distinct neighbor components on addNode
    std::vector<Node> mergedNodesScratch_;           ///< Node list of the merged component
    std::vector<int> compSeenEpoch_;                 ///< Component-index dedup marks
    int compSeenCurrentEpoch_ = 0;

    // Bump and return the DFS epoch, resetting dfsVisitEpoch_ on wraparound.
    int nextDfsEpoch() const;

    // Collect the component containing startNode into `out`, marking visited
    // nodes with `epoch`. Nodes already carrying `epoch` are skipped, so
    // several calls sharing one epoch partition disjoint sub-components.
    void dfsCollect(Node startNode, int epoch, Component &out) const;

    // Iterative Tarjan articulation-point pass over one component (fills the
    // dfn_/lowVec_/... buffers indexed by local 1-based node index).
    void tarjanInComponent(ComponentIndex compIndex) const;
    int calculateObjectiveDecreaseByRemovingNodeInternal(Node node);

public:
    NodeSet removedNodes;

    /**
     * Creates a deep copy of the graph.
     *
     * Returns
     * -------
     * CNP_Graph
     *     A new graph instance with the same structure.
     */
    std::unique_ptr<CNP_Graph> clone() const;

    CNP_Graph(NodeSet nodes, std::vector<NodeSet> adjList, int budget, int seed);

    CNP_Graph() {};

    /**
     * Updates the graph to reflect node removals.
     *
     * Parameters
     * ----------
     * removedNodes : NodeSet
     *     Set of nodes that have been removed.
     */
    void updateGraphByRemovedNodes(const NodeSet &removedNodes);

    /**
     * Creates a reduced graph after node removal.
     *
     * Parameters
     * ----------
     * nodesToRemove : NodeSet
     *     Set of nodes to remove.
     *
     * Returns
     * -------
     * CNP_Graph
     *     The reduced graph.
     */
    void getReducedGraphByRemovedNodes(const NodeSet &nodesToRemove);

    /**
     * Adds a node back to the graph.
     *
     * Only the components adjacent to the node are touched: their objective
     * contributions are swapped for the merged component's contribution.
     *
     * Parameters
     * ----------
     * node : Node
     *     The node to add.
     */
    void addNode(Node node);

    /**
     * Removes a node from the graph.
     *
     * Only the node's own component is touched: it is re-partitioned into the
     * sub-components left behind and the objective updated incrementally.
     *
     * Parameters
     * ----------
     * node : Node
     *     The node to remove.
     */
    void removeNode(Node node);

    /**
     * Greedily removes candidate nodes until the graph reaches its removal budget.
     *
     * At each step, the node whose removal yields the smallest objective value
     * is selected and removed.
     *
     * Parameters
     * ----------
     * candidateNodes : NodeSet
     *     Candidate nodes that may be removed.
     */
    void removeBestNodesFromSet(const NodeSet &candidateNodes);

    /**
     * Checks if a node has been removed.
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
     * Returns the total number of nodes in the graph.
     *
     * Returns
     * -------
     * int
     *     Number of nodes.
     */
    int getNumNodes() const;

    /**
     * Returns the set of removed nodes.
     *
     * Returns
     * -------
     * NodeSet
     *     Set of nodes that have been removed.
     */
    const NodeSet& getRemovedNodes() const;

    /**
     * Sets the age of a node.
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
     * Calculates the objective value of the graph (CNP1 pairwise connectivity).
     *
     * Returns
     * -------
     * int
     *     CNP1 pairwise connectivity Σ|C|(|C|-1)/2.
     */
    int getObjectiveValue() const;

    // Converts to random feasible solution using the given seed.
    std::unique_ptr<CNP_Graph> getRandomFeasibleGraph(int seed) const;

    // Converts to random partial solution with specified budget and seed.
    std::unique_ptr<CNP_Graph> getRandomPartialGraph(int partialBudget, int seed) const;

    // Initializes components and node mapping.
    void initializeComponentsAndMapping();

    // Finds connected component using DFS.
    Component dfsFindComponent(Node startNode) const;

    // Selects component to remove.
    ComponentIndex selectRemovedComponent() const;

    // Randomly selects node to remove from specified component.
    Node randomSelectNodeFromComponent(ComponentIndex componentIndex) const;

    // Selects node to remove from component based on weight.
    Node ageSelectNodeFromComponent(ComponentIndex componentIndex) const;

    // Selects node with minimum impact to remove from component.
    Node impactSelectNodeFromComponent(ComponentIndex componentIndex) const;

    // Selects node to add back to graph.
    Node greedySelectNodeToAdd() const;

    // Randomly selects a node to remove.
    Node randomSelectNodeToRemove() const;

    // Calculates connection gain after adding a node.
    int calculateConnectionGain(Node node, std::vector<size_t>& componentSizes) const;
};

#endif
