import unittest
import networkx as nx
from synkit.Graph.canon_graph import GraphCanonicaliser, CanonicalGraph


class TestGraphCanonicaliser(unittest.TestCase):
    def setUp(self):
        # Simple graph: two nodes with an edge
        self.G = nx.Graph()
        self.G.add_node(1, element="C", charge=0)
        self.G.add_node(2, element="O", charge=0)
        self.G.add_edge(1, 2, order=1)

        # Another graph with same structure but swapped node labels
        self.G_swapped = nx.Graph()
        self.G_swapped.add_node(10, element="C", charge=0)
        self.G_swapped.add_node(9, element="O", charge=0)
        self.G_swapped.add_edge(10, 9, order=1)

        # A different graph
        self.H = nx.Graph()
        self.H.add_node(1, element="C", charge=0)
        self.H.add_node(2, element="N", charge=0)
        self.H.add_edge(1, 2, order=1)

        self.canon = GraphCanonicaliser()

    def test_canonical_signature_consistency(self):
        sig1 = self.canon.canonical_signature(self.G)
        sig2 = self.canon.canonical_signature(self.G)
        self.assertEqual(sig1, sig2, "Signature should be consistent on repeated calls")

    def test_canonical_signature_equivalence(self):
        CG1 = self.canon.canonicalise_graph(self.G)
        CG2 = self.canon.canonicalise_graph(self.G_swapped)
        sig1 = CG1.canonical_hash
        sig2 = CG2.canonical_hash
        self.assertEqual(
            sig1, sig2, "Graphs with same structure should have equal signatures"
        )

    def test_canonical_signature_difference(self):
        sigG = self.canon.canonical_signature(self.G)
        sigH = self.canon.canonical_signature(self.H)
        self.assertNotEqual(
            sigG, sigH, "Different graphs should have different signatures"
        )

    def test_make_canonical_graph_structure(self):
        G_can = self.canon.make_canonical_graph(self.G_swapped)
        # Canonical graph should have nodes labeled 1 and 2
        self.assertSetEqual(set(G_can.nodes()), {1, 2})
        # Check that data preserved
        self.assertEqual(G_can.nodes[1]["element"], "C")
        self.assertEqual(G_can.nodes[2]["element"], "O")

    def test_make_canonical_graph_edge(self):
        G_can = self.canon.make_canonical_graph(self.G_swapped)
        self.assertTrue(G_can.has_edge(1, 2))
        self.assertEqual(G_can.edges[1, 2]["order"], 1)

    def test_canonicalise_graph_wrapper(self):
        cgraph = self.canon.canonicalise_graph(self.G)
        # Check properties
        self.assertIsInstance(cgraph, CanonicalGraph)
        self.assertEqual(cgraph.canonical_hash, self.canon.canonical_signature(self.G))
        # original_graph should point to the same object
        self.assertIs(cgraph.original_graph, self.G)
        # canonical_graph should be new
        self.assertIsNot(cgraph.canonical_graph, self.G)

    def test_canonicalgraph_equality_and_hash(self):
        c1 = self.canon.canonicalise_graph(self.G)
        c2 = self.canon.canonicalise_graph(self.G_swapped)
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))
        # Different graph wrapper
        cH = self.canon.canonicalise_graph(self.H)
        self.assertNotEqual(c1, cH)

    def test_repr_and_str(self):
        c1 = self.canon.canonicalise_graph(self.G)
        s = str(c1)
        self.assertIn("CanonicalGraph", s)
        # repr same as str
        self.assertEqual(repr(c1), s)


if __name__ == "__main__":
    unittest.main()
