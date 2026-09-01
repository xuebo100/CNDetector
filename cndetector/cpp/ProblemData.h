#ifndef PROBLEM_DATA_H
#define PROBLEM_DATA_H

#include "Graph/CNP_Graph.h"
#include "Graph/DCNP_Graph.h"
#include "Graph/Types.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

class ProblemData
{
public:
    explicit ProblemData(int numNodes);

    void addNode(int nodeId);
    void addEdge(int u, int v);

    std::unique_ptr<CNP_Graph> createOriginalGraph(
        int budget, int seed) const;
    std::unique_ptr<DCNP_Graph> createOriginalDCNPGraph(
        int budget, int distance, int seed) const;

    static ProblemData readFromAdjacencyListFile(const std::string &filename);
    static ProblemData readFromDimacsEdgeFile(const std::string &filename);

    const NodeSet &getNodesSet() const { return nodes_; }
    const std::vector<NodeSet> &getAdjList() const { return adjList_; }
    int numNodes() const { return static_cast<int>(nodes_.size()); }

private:
    NodeSet nodes_;
    std::vector<NodeSet> adjList_;
};

#endif // PROBLEM_DATA_H
