import unittest
import networkx as nx
from synkit.Graph.Feature.hash_fps import HashFPs
from synkit.IO.data_io import load_from_pickle


class TestHashFPs(unittest.TestCase):
    def setUp(self):
        """Set up a simple graph for testing."""
        self.graph = nx.cycle_graph(4)  # Simple cycle graph with 4 nodes
        self.hasher = HashFPs(self.graph, numBits=128, hash_alg="sha256")

    def test_hash_fps_default(self):
        """Test the default hash generation without specifying start or end nodes."""
        result = self.hasher.hash_fps()
        self.assertEqual(len(result), 128)
        self.assertIsInstance(result, str)
        self.assertTrue(all(c in "01" for c in result), "Hash must be binary")

    def test_hash_fps_path_specified(self):
        """Test hash generation with specified start and end nodes."""
        result = self.hasher.hash_fps(start_node=0, end_node=1)
        self.assertEqual(len(result), 128)
        self.assertTrue(all(c in "01" for c in result), "Hash must be binary")

    def test_hash_fps_invalid_hash_algorithm(self):
        """Test initialization with an invalid hash algorithm."""
        with self.assertRaises(ValueError):
            HashFPs(self.graph, numBits=128, hash_alg="invalid256")

    def test_hash_fps_negative_numBits(self):
        """Test initialization with negative numBits."""
        with self.assertRaises(ValueError):
            HashFPs(self.graph, numBits=-1, hash_alg="sha256")

    def test_hash_fps_large_numBits(self):
        """Test hash generation with a large numBits."""
        large_hasher = HashFPs(self.graph, numBits=1024, hash_alg="sha512")
        result = large_hasher.hash_fps()
        self.assertEqual(len(result), 1024)
        self.assertTrue(all(c in "01" for c in result), "Hash must be binary")

    def test_fps_rc(self):
        data = load_from_pickle("Data/Testcase/test.pkl.gz")
        graph = data[0]["GraphRules"][2]
        hasher = HashFPs(graph, numBits=1024, hash_alg="sha256")
        result = hasher.hash_fps()
        self.assertEqual(len(result), 1024)


if __name__ == "__main__":
    unittest.main()
