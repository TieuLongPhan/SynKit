import unittest
import networkx as nx
from synkit.Graph.Feature.path_fps import PathFPs
from synkit.IO.data_io import load_from_pickle


class TestPathFPs(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(5)  # Creates a simple path graph
        self.path_fps = PathFPs(self.graph, max_length=3, nBits=64, hash_alg="sha256")

    def test_fingerprint_length(self):
        """Test that the fingerprint has the exact length specified by nBits."""
        fingerprint = self.path_fps.generate_fingerprint()
        self.assertEqual(len(fingerprint), 64)

    def test_fingerprint_consistency(self):
        """Test that the same graph with the same parameters produces the same
        fingerprint."""
        fingerprint1 = self.path_fps.generate_fingerprint()
        fingerprint2 = self.path_fps.generate_fingerprint()
        self.assertEqual(fingerprint1, fingerprint2)

    def test_fingerprint_variation(self):
        """Test that changing the parameters changes the fingerprint."""
        new_path_fps = PathFPs(self.graph, max_length=4, nBits=128, hash_alg="sha256")
        fingerprint1 = self.path_fps.generate_fingerprint()
        fingerprint2 = new_path_fps.generate_fingerprint()
        self.assertNotEqual(fingerprint1, fingerprint2)

    def test_fps_rc(self):
        data = load_from_pickle("Data/Testcase/test.pkl.gz")
        graph = data[0]["GraphRules"][2]
        hasher = PathFPs(graph, max_length=5, nBits=1024, hash_alg="sha256")
        result = hasher.generate_fingerprint()
        self.assertEqual(len(result), 1024)


if __name__ == "__main__":
    unittest.main()
