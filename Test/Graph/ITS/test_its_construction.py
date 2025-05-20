import unittest
import networkx as nx
from synkit.Graph.ITS.its_construction import ITSConstruction


class TestITSConstruction(unittest.TestCase):

    def setUp(self):
        # Create test graphs G and H with predefined attributes and edges

        # Ethylen C=C
        self.G = nx.Graph()
        self.G.add_node(1, element="C", aromatic=False, hcount=2, charge=0)
        self.G.add_node(2, element="C", aromatic=False, hcount=2, charge=0)
        self.G.add_edge(1, 2, order=2)

        # Ethan C-C
        self.H = nx.Graph()
        self.H.add_node(1, element="C", aromatic=False, hcount=3, charge=0)
        self.H.add_node(2, element="C", aromatic=False, hcount=3, charge=0)
        self.H.add_edge(1, 2, order=1)  # Different order

    def test_ITSGraph(self):
        ITS = ITSConstruction().ITSGraph(self.G, self.H)
        self.assertTrue(isinstance(ITS, nx.Graph))
        self.assertEqual(len(ITS.nodes()), 2)
        self.assertEqual(len(ITS.edges()), 1)
        self.assertEqual(ITS[1][2]["order"], (2, 1))

    def test_get_node_attributes_with_defaults(self):
        attributes = ITSConstruction().get_node_attributes_with_defaults(self.G, 1)
        self.assertEqual(attributes, ("C", False, 2, 0, ["", ""]))

    def test_add_edges_to_ITS(self):
        ITS = nx.Graph()
        ITS.add_node(1, element="C", aromatic=False, hcount=3, charge=0)
        ITS.add_node(2, element="C", aromatic=False, hcount=3, charge=0)
        new_ITS = ITSConstruction().add_edges_to_ITS(ITS, self.G, self.H)
        self.assertTrue(isinstance(new_ITS, nx.Graph))
        self.assertEqual(len(new_ITS.edges()), 1)
        self.assertEqual(new_ITS[1][2]["order"], (2, 1))

    def test_add_standard_order_attribute(self):
        graph = nx.Graph()
        graph.add_edge(1, 2, order=(1, 2))
        updated_graph = ITSConstruction().add_standard_order_attribute(graph)
        self.assertEqual(updated_graph[1][2]["standard_order"], -1)


if __name__ == "__main__":
    unittest.main()
