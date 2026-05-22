import unittest

import matplotlib
import networkx as nx

matplotlib.use("Agg")

from synkit.Vis.visual_drawer import draw_graph  # noqa: E402
from synkit.Vis.visual_model import to_visual_graph  # noqa: E402


class TestVisualDrawer(unittest.TestCase):
    def test_draw_graph_returns_figure_and_axes_without_mutating_input(self):
        graph = nx.Graph()
        graph.add_node(1, element="C", atom_map=1, charge=0)
        graph.add_node(2, element="O", atom_map=2, charge=0)
        graph.add_edge(1, 2, order=(1.0, 2.0))
        before_nodes = dict(graph.nodes(data=True))
        before_edges = list(graph.edges(data=True))

        fig, ax = draw_graph(graph)

        self.assertIs(fig, ax.figure)
        self.assertEqual(dict(graph.nodes(data=True)), before_nodes)
        self.assertEqual(list(graph.edges(data=True)), before_edges)

    def test_draw_graph_accepts_visual_graph(self):
        graph = nx.Graph()
        graph.add_node(1, element="N", atom_map=1, hcount=(2, 1))
        graph.add_node(2, element="C", atom_map=2, hcount=(3, 3))
        graph.add_edge(
            1,
            2,
            order=(1.0, 2.0),
            kekule_order=(1.0, 2.0),
            sigma_order=(1.0, 1.0),
            pi_order=(0.0, 1.0),
        )
        visual = to_visual_graph(graph, mode="sigma_pi")

        fig, ax = draw_graph(visual, mode="sigma_pi")

        self.assertIs(fig, ax.figure)
        self.assertEqual(ax.get_title(), "tuple_its")


if __name__ == "__main__":
    unittest.main()
