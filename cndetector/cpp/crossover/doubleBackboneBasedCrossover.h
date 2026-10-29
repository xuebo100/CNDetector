#ifndef DOUBLE_BACKBONE_BASED_CROSSOVER_H
#define DOUBLE_BACKBONE_BASED_CROSSOVER_H

#include "../Graph/CNP_Graph.h"

#include <memory>
#include <utility>

// Probability with which the double backbone-based crossover inherits a node
// removed by only one of the two parents.
constexpr double DBX_EXCLUSIVE_INHERIT_RATE = 0.85;

// Double backbone-based crossover (DBX), the crossover that the
// reduce-solve-combine crossover replaces; it is kept so the two can be
// compared. The offspring inherits every node removed by both parents (the
// common backbone) and each node removed by only one parent (the exclusive
// backbones) with probability DBX_EXCLUSIVE_INHERIT_RATE. It is then brought to
// targetSize nodes: missing nodes are removed at random from components drawn
// by CNP_Graph::selectRemovedComponentCHNS, and surplus nodes are added back
// greedily. CNP only.
std::unique_ptr<CNP_Graph> doubleBackboneBasedCrossover(
    const CNP_Graph &originalGraph,
    const std::pair<const Solution *, const Solution *> &parents,
    int targetSize,
    int seed);

#endif // DOUBLE_BACKBONE_BASED_CROSSOVER_H
