#ifndef GRAPH_TYPES_H
#define GRAPH_TYPES_H

#include <unordered_set>
#include <vector>

using Node = int;
using Age = long;
using NodeSet = std::unordered_set<Node>;
using ComponentIndex = int;
using Solution = NodeSet;

/**
 * Represents a connected component in the graph.
 */
struct Component
{
    std::vector<Node> nodes;  ///< List of all nodes contained in this connected component.
    size_t size() const noexcept { return nodes.size(); }
};

using Components = std::vector<Component>;

constexpr Node INVALID_NODE = -1;

#endif  // GRAPH_TYPES_H
