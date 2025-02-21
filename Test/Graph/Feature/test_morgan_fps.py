import unittest
import networkx as nx
from synkit.Graph.Feature.morgan_fps import MorganFPs
from synkit.IO.data_io import load_from_pickle


class TestMorganFPs(unittest.TestCase):
    def setUp(self):
        self.graph = nx.cycle_graph(5)  # Creates a cycle graph for testing
        self.morgan_fps = MorganFPs(self.graph, radius=2, nBits=128, hash_alg="sha256")

    def test_fingerprint_length(self):
        """Test that the fingerprint is exactly the specified bit length."""
        fingerprint = self.morgan_fps.generate_fingerprint()
        self.assertEqual(len(fingerprint), 128)

    def test_fingerprint_consistency(self):
        """Test that the same graph with the same parameters produces the same fingerprint."""
        fingerprint1 = self.morgan_fps.generate_fingerprint()
        fingerprint2 = self.morgan_fps.generate_fingerprint()
        self.assertEqual(fingerprint1, fingerprint2)

    def test_fingerprint_variation_with_radius(self):
        """Test that changing the radius changes the fingerprint."""
        new_morgan_fps = MorganFPs(self.graph, radius=1, nBits=128, hash_alg="sha256")
        fingerprint1 = self.morgan_fps.generate_fingerprint()
        fingerprint2 = new_morgan_fps.generate_fingerprint()
        self.assertNotEqual(fingerprint1, fingerprint2)

    def test_fps_rc(self):
        data = load_from_pickle("Data/Testcase/test.pkl.gz")
        graph = data[0]["GraphRules"][2]
        hasher = MorganFPs(graph, radius=3, nBits=1024, hash_alg="sha256")
        result = hasher.generate_fingerprint()
        self.assertEqual(len(result), 1024)


if __name__ == "__main__":
    unittest.main()
