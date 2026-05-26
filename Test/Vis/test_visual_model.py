import unittest

import networkx as nx

from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.MTG.mtg import MTG
from synkit.Vis.visual_model import (
    detect_visual_kind,
    iter_changed_edges,
    iter_changed_nodes,
    summarize_visual_graph,
    to_visual_graph,
)


class TestVisualModel(unittest.TestCase):
    @staticmethod
    def _atom(element, *, hcount=0, charge=0, lone_pairs=0, radical=0):
        return {
            "element": element,
            "aromatic": False,
            "hcount": hcount,
            "charge": charge,
            "lone_pairs": lone_pairs,
            "radical": radical,
            "valence_electrons": {"H": 1, "C": 4, "N": 5, "O": 6, "Cl": 7}[element],
        }

    @staticmethod
    def _bond(graph, u, v, sigma=1.0, pi=0.0):
        graph.add_edge(
            u,
            v,
            order=sigma + pi,
            kekule_order=sigma + pi,
            sigma_order=sigma,
            pi_order=pi,
        )

    def _graph(self, nodes, edges):
        graph = nx.Graph()
        for node, attrs in nodes.items():
            graph.add_node(node, **attrs)
        for edge in edges:
            self._bond(graph, *edge)
        return graph

    def test_detects_molecule(self):
        graph = self._graph(
            {
                1: self._atom("C", hcount=3),
                2: self._atom("Cl", lone_pairs=3),
            },
            [(1, 2, 1.0, 0.0)],
        )

        visual = to_visual_graph(graph)

        self.assertEqual(detect_visual_kind(graph), "molecule")
        self.assertEqual(visual.kind, "molecule")
        self.assertEqual(visual.edges[0].label, "—")

    def test_detects_legacy_its(self):
        its = nx.Graph()
        its.add_node(1, element="C", atom_map=1)
        its.add_node(2, element="O", atom_map=2)
        its.add_edge(1, 2, order=(1.0, 2.0), standard_order=-1.0)

        visual = to_visual_graph(its)

        self.assertEqual(detect_visual_kind(its), "legacy_its")
        self.assertEqual(visual.edges[0].state, "order_changed")
        self.assertEqual(visual.edges[0].label, "—>=")

    def test_detects_tuple_its_and_sigma_pi_labels(self):
        reactant = self._graph(
            {
                1: self._atom("N", hcount=2, lone_pairs=1),
                2: self._atom("C", hcount=3),
            },
            [(1, 2, 1.0, 0.0)],
        )
        product = self._graph(
            {
                1: self._atom("N", hcount=1, lone_pairs=1, radical=1),
                2: self._atom("C", hcount=3),
            },
            [(1, 2, 1.0, 1.0)],
        )
        its = ITSConstruction.construct(reactant, product)

        visual = to_visual_graph(its, mode="sigma_pi")

        self.assertEqual(detect_visual_kind(its), "tuple_its")
        self.assertIn("π0>1", visual.edges[0].label)
        self.assertEqual(visual.edges[0].state, "order_changed")
        self.assertEqual([node.node_id for node in iter_changed_nodes(visual)], [1])

    def test_detects_compact_mtg_and_transient_edges(self):
        g0 = self._graph(
            {
                1: self._atom("C", hcount=3),
                2: self._atom("Cl", lone_pairs=3),
            },
            [(1, 2, 1.0, 0.0)],
        )
        g1 = self._graph(
            {
                1: self._atom("C", hcount=3, radical=1),
                2: self._atom("Cl", radical=1, lone_pairs=3),
            },
            [],
        )
        g2 = self._graph(
            {
                1: self._atom("C", hcount=3),
                2: self._atom("Cl", lone_pairs=3),
            },
            [(1, 2, 1.0, 0.0)],
        )
        mtg = MTG(
            [ITSConstruction.construct(g0, g1), ITSConstruction.construct(g1, g2)],
            mappings=[{1: 1, 2: 2}],
        ).get_mtg()

        visual = to_visual_graph(mtg, mode="timeline")
        changed_edges = list(iter_changed_edges(visual))

        self.assertEqual(detect_visual_kind(mtg), "compact_mtg")
        self.assertEqual(changed_edges[0].state, "transient")
        self.assertIn("σ:1-0-1", changed_edges[0].label)
        summary = summarize_visual_graph(visual)
        self.assertEqual(summary["kind"], "compact_mtg")


if __name__ == "__main__":
    unittest.main()
