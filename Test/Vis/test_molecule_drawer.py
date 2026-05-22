import unittest

import matplotlib
import networkx as nx

matplotlib.use("Agg")

from synkit.IO.chem_converter import smiles_to_graph  # noqa: E402
from synkit.Vis.molecule_drawer import draw_molecule_graph  # noqa: E402


class TestMoleculeDrawer(unittest.TestCase):
    def test_draw_molecule_graph_returns_axes_without_mutation(self):
        graph = nx.Graph()
        graph.add_node(1, element="C", charge=0, atom_map=1)
        graph.add_node(2, element="O", charge=0, atom_map=2)
        graph.add_edge(1, 2, order=2)
        before_nodes = dict(graph.nodes(data=True))
        before_edges = list(graph.edges(data=True))

        ax = draw_molecule_graph(graph, label_mode="all", show_atom_map=True)

        self.assertEqual(ax.get_aspect(), 1.0)
        self.assertEqual(dict(graph.nodes(data=True)), before_nodes)
        self.assertEqual(list(graph.edges(data=True)), before_edges)

    def test_draw_aromatic_molecule(self):
        graph = nx.cycle_graph(6)
        mapping = {node: node + 1 for node in graph.nodes}
        graph = nx.relabel_nodes(graph, mapping)
        nx.set_node_attributes(graph, "C", "element")
        nx.set_node_attributes(graph, 0, "charge")
        nx.set_node_attributes(graph, True, "aromatic")
        nx.set_edge_attributes(graph, 1.5, "order")
        nx.set_edge_attributes(graph, True, "aromatic")

        ax = draw_molecule_graph(graph, aromatic_style="circle")

        self.assertEqual(ax.get_aspect(), 1.0)

    def test_draw_with_rdkit_panel(self):
        graph = nx.Graph()
        graph.add_node(1, element="N", charge=1, atom_map=1)
        graph.add_node(2, element="C", charge=0, atom_map=2)
        graph.add_edge(1, 2, order=1)

        fig, axes = draw_molecule_graph(graph, include_rdkit_panel=True)

        self.assertEqual(len(axes), 2)
        self.assertIs(fig, axes[0].figure)

    def test_draw_real_smiles_graph_aspirin_like_case(self):
        graph = smiles_to_graph(
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            sanitize=True,
            use_index_as_atom_map=True,
        )

        ax = draw_molecule_graph(
            graph,
            label_mode="hetero",
            show_atom_map=True,
            aromatic_style="circle",
            title="Aspirin-like SMILES graph",
        )

        self.assertEqual(graph.number_of_nodes(), 13)
        self.assertEqual(graph.number_of_edges(), 13)
        self.assertEqual(ax.get_title(), "Aspirin-like SMILES graph")


if __name__ == "__main__":
    unittest.main()
