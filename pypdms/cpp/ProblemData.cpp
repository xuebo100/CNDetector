#include "ProblemData.h"
#include <cctype>
#include <stdexcept>

ProblemData::ProblemData(int numNodes)
{
    adjList_.resize(static_cast<size_t>(numNodes));
}

void ProblemData::addNode(int nodeId)
{
    if (nodeId < 0)
    {
        throw std::invalid_argument("node ID must be non-negative");
    }
    if (static_cast<size_t>(nodeId) >= adjList_.size())
    {
        adjList_.resize(static_cast<size_t>(nodeId) + 1);
    }
    nodes_.insert(nodeId);
}

void ProblemData::addEdge(int u, int v)
{
    if (u < 0 || v < 0)
    {
        throw std::invalid_argument("node IDs must be non-negative");
    }
    const size_t maxId = static_cast<size_t>(std::max(u, v));
    if (maxId >= adjList_.size())
    {
        adjList_.resize(maxId + 1);
    }
    adjList_[u].insert(v);
    adjList_[v].insert(u);
    nodes_.insert(u);
    nodes_.insert(v);
}

std::unique_ptr<CNP_Graph> ProblemData::createOriginalGraph(
    int budget, int seed) const
{
    auto graph = std::make_unique<CNP_Graph>(nodes_, adjList_, budget, seed);
    graph->initializeComponentsAndMapping();
    return graph;
}

std::unique_ptr<DCNP_Graph> ProblemData::createOriginalDCNPGraph(
    int budget, int distance, int seed) const
{
    if (distance < 1)
    {
        throw std::invalid_argument("DCNP distance must be positive");
    }
    return std::make_unique<DCNP_Graph>(
        nodes_, distance, adjList_, budget, seed);
}

ProblemData ProblemData::readFromAdjacencyListFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int numNodes = 0;
    file >> numNodes;
    if (file.fail())
    {
        throw std::runtime_error("Failed to read node count from file");
    }

    ProblemData data(numNodes);

    for (int expected = 0; expected < numNodes; ++expected)
    {
        int node = -1;
        file >> node;
        if (file.fail())
        {
            throw std::runtime_error("Failed to read node id from adjacency-list file");
        }

        data.addNode(node);

        char c = '\0';
        while (file.get(c))
        {
            if (c == ':') break;
            if (!std::isspace(static_cast<unsigned char>(c)))
            {
                throw std::runtime_error("Malformed adjacency-list separator");
            }
        }

        bool nextLine = false;
        while (!file.eof() && !nextLine)
        {
            while (std::isspace(file.peek()))
            {
                nextLine = (file.get() == '\n');
                if (nextLine) break;
            }

            if (!nextLine && !file.eof())
            {
                int neighbor = -1;
                file >> neighbor;
                if (file.fail()) break;
                data.addEdge(node, neighbor);
            }
        }
    }

    return data;
}

ProblemData ProblemData::readFromDimacsEdgeFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    ProblemData data(0);
    bool sawHeader = false;
    std::string tag;

    while (file >> tag)
    {
        if (tag == "c")
        {
            std::string ignored;
            std::getline(file, ignored);
            continue;
        }
        if (tag == "p")
        {
            std::string format;
            int numNodes = 0;
            int numEdges = 0;
            file >> format >> numNodes >> numEdges;
            if (file.fail() || numNodes < 0)
            {
                throw std::runtime_error("Malformed DIMACS problem line");
            }
            data = ProblemData(numNodes);
            for (int node = 0; node < numNodes; ++node)
            {
                data.addNode(node);
            }
            sawHeader = true;
            continue;
        }
        if (tag == "e")
        {
            int u = -1;
            int v = -1;
            file >> u >> v;
            if (file.fail())
            {
                throw std::runtime_error("Malformed DIMACS edge line");
            }
            if (!sawHeader)
            {
                const int maxNode = std::max(u, v);
                data = ProblemData(maxNode + 1);
                for (int node = 0; node <= maxNode; ++node)
                {
                    data.addNode(node);
                }
                sawHeader = true;
            }
            data.addEdge(u, v);
            continue;
        }

        std::string ignored;
        std::getline(file, ignored);
    }

    return data;
}
