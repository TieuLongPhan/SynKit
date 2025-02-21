import unittest
import numpy as np

from synkit.Vis.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./cachedir"
        self.verbose = 0
        self.custom_tsne_params = {"perplexity": 40}
        self.data = np.random.rand(10, 5)  # Random data for embedding

    def test_initialization_with_custom_params(self):
        """
        Test initialization of the Embedding class with custom t-SNE parameters.
        """
        embedding = Embedding(self.cache_dir, self.verbose, self.custom_tsne_params)
        self.assertEqual(embedding.tsne_params["perplexity"], 40)

    def test_initialization_without_custom_params(self):
        """
        Test default initialization behavior of the Embedding class.
        """
        embedding = Embedding(self.cache_dir, self.verbose)
        self.assertEqual(embedding.tsne_params["perplexity"], 30)  # Default value

    def test_set_tsne_params(self):
        """
        Test that the set_tsne_params method correctly updates the parameters.
        """
        embedding = Embedding(self.cache_dir)
        embedding.set_tsne_params(perplexity=50, n_iter=300)
        self.assertEqual(embedding.tsne_params["perplexity"], 50)
        self.assertEqual(embedding.tsne_params["n_iter"], 300)

    def test_reset_tsne_params(self):
        """
        Test resetting t-SNE parameters to their default values after they have been changed.
        """
        embedding = Embedding(self.cache_dir)
        embedding.set_tsne_params(perplexity=50)  # Change default
        embedding.reset_tsne_params()
        self.assertEqual(
            embedding.tsne_params["perplexity"], 30
        )  # Check reset to default


if __name__ == "__main__":
    unittest.main()
