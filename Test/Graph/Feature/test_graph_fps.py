import unittest
import networkx as nx
from synkit.Graph.Feature.graph_fps import GraphFP


class TestGraphFP(unittest.TestCase):

    def setUp(self):
        """Set up a test graph for use in all test cases."""
        self.graph = nx.gnp_random_graph(10, 0.5, seed=42)
        self.nBits = 512
        self.hash_alg = "sha256"
        self.fp_class = GraphFP(
            graph=self.graph, nBits=self.nBits, hash_alg=self.hash_alg
        )

    def test_spectrum_fp(self):
        """Test the spectrum-based fingerprint generation."""
        fingerprint = self.fp_class.fingerprint("spectrum")
        self.assertEqual(len(fingerprint), self.nBits)
        self.assertTrue(isinstance(fingerprint, str))

    def test_adjacency_fp(self):
        """Test the adjacency matrix-based fingerprint generation."""
        fingerprint = self.fp_class.fingerprint("adjacency")
        self.assertEqual(len(fingerprint), self.nBits)
        self.assertTrue(isinstance(fingerprint, str))

    def test_degree_sequence_fp(self):
        """Test the degree sequence-based fingerprint generation."""
        fingerprint = self.fp_class.fingerprint("degree")
        self.assertEqual(len(fingerprint), self.nBits)
        self.assertTrue(isinstance(fingerprint, str))

    def test_motif_count_fp(self):
        """Test the motif count-based fingerprint generation."""
        fingerprint = self.fp_class.fingerprint("motif")
        self.assertEqual(len(fingerprint), self.nBits)
        self.assertTrue(isinstance(fingerprint, str))

    def test_iterative_deepening(self):
        """Test the iterative deepening method."""
        short_fingerprint = "1010101010101010"
        remaining_bits = self.nBits - len(short_fingerprint)
        extended_fingerprint = self.fp_class.iterative_deepening(remaining_bits)
        self.assertEqual(len(extended_fingerprint), remaining_bits)
        self.assertTrue(isinstance(extended_fingerprint, str))

    def test_fingerprint_length(self):
        """Test that each method produces a fingerprint of exactly nBits."""
        methods = ["spectrum", "adjacency", "degree", "motif"]
        for method in methods:
            with self.subTest(method=method):
                fingerprint = self.fp_class.fingerprint(method)
                self.assertEqual(len(fingerprint), self.nBits)


if __name__ == "__main__":
    unittest.main()
