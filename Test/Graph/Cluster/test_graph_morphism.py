import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.Cluster.graph_morphism import graph_isomorphism, subgraph_isomorphism


class TestRule(unittest.TestCase):

    def setUp(self):
        self.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")

    def test_graph_isomorphism_true(self):
        result = graph_isomorphism(
            self.graphs[0]["RC"], self.graphs[3]["RC"], use_defaults=True
        )
        self.assertTrue(result)

    def test_graph_isomorphism_false(self):
        result = graph_isomorphism(
            self.graphs[0]["RC"], self.graphs[1]["RC"], use_defaults=True
        )
        self.assertFalse(result)

    def test_graph_subgraph_morphism_true(self):
        result = subgraph_isomorphism(self.graphs[0]["RC"], self.graphs[0]["ITS"])
        self.assertTrue(result)

    def test_graph_subgraph_morphism_false(self):
        result = subgraph_isomorphism(self.graphs[0]["RC"], self.graphs[1]["ITS"])
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
