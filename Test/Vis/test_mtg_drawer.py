import unittest

import matplotlib
import networkx as nx

matplotlib.use("Agg")

from synkit.Graph.ITS.its_construction import ITSConstruction  # noqa: E402
from synkit.Graph.MTG.mtg import MTG  # noqa: E402
from synkit.Vis.mtg_drawer import draw_mtg_graph, draw_mtg_steps  # noqa: E402


class TestMTGDrawer(unittest.TestCase):
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

    def _mtg(self):
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
        return MTG(
            [ITSConstruction.construct(g0, g1), ITSConstruction.construct(g1, g2)],
            mappings=[{1: 1, 2: 2}],
        )

    def test_draw_mtg_graph_accepts_mtg_object(self):
        mtg = self._mtg()

        fig, ax = draw_mtg_graph(mtg, title="radical rebound")

        self.assertIs(fig, ax.figure)
        self.assertEqual(ax.get_title(), "radical rebound")

    def test_draw_mtg_graph_accepts_raw_graph_without_mutation(self):
        graph = self._mtg().get_mtg()
        before_nodes = dict(graph.nodes(data=True))
        before_edges = list(graph.edges(data=True))

        fig, ax = draw_mtg_graph(graph)

        self.assertIs(fig, ax.figure)
        self.assertEqual(dict(graph.nodes(data=True)), before_nodes)
        self.assertEqual(list(graph.edges(data=True)), before_edges)

    def test_draw_mtg_steps_draws_ordered_its_panels_and_composed_panel(self):
        mtg = self._mtg()

        fig, axes = draw_mtg_steps(mtg, include_composed=True, show_edge_labels=True)

        self.assertIs(fig, axes[0].figure)
        self.assertEqual(len(axes), 3)
        self.assertEqual([ax.get_title() for ax in axes], ["Step 1", "Step 2", "Composed"])

    def test_draw_mtg_steps_validates_indices(self):
        with self.assertRaises(IndexError):
            draw_mtg_steps(self._mtg(), steps=[2])


if __name__ == "__main__":
    unittest.main()
