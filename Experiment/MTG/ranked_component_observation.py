"""Independent exact component reachability using bounded witness ranks.

Present-edge closure prevents missing connected vertices; strictly decreasing
distances prevent extraneous self-supporting reached components.
"""

import networkx as nx
import z3

from Experiment.MTG.component_observation import ComponentQuery


class RankedComponentQuery(ComponentQuery):
    def _reach(self, union, root):
        nodes = sorted(nx.node_connected_component(union, root))
        reach = {n: z3.Bool(f"rank_reach_{root}_{n}", ctx=self.context) for n in nodes}
        distance = {
            n: z3.Int(f"rank_distance_{root}_{n}", ctx=self.context) for n in nodes
        }
        self.solver.add(reach[root], distance[root] == 0)
        self.hop_definitions += 2
        for pair, present in self.edges.items():
            if pair <= set(nodes):
                a, b = tuple(pair)
                self.solver.add(z3.Implies(present, reach[a] == reach[b]))
                self.hop_definitions += 1
        for node in nodes:
            if node == root:
                continue
            predecessors = [
                z3.And(
                    reach[other],
                    self.edges[frozenset((node, other))],
                    distance[other] < distance[node],
                )
                for other in union[node]
            ]
            self.solver.add(
                z3.Implies(
                    reach[node], z3.And(distance[node] > 0, distance[node] < len(nodes))
                ),
                z3.Implies(reach[node], z3.Or(*predecessors)),
            )
            self.hop_definitions += 2
        return reach
