import unittest
import networkx as nx
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Context.radius_expand import RadiusExpand
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism, subgraph_isomorphism


class TestRadiusExpand(unittest.TestCase):

    def setUp(self):
        self.data = load_from_pickle("Data/Testcase/graph.pkl.gz")

    def test_find_unequal_order_edges(self):
        # Create a graph with edges that include various 'order' and 'standard_order' attributes.
        G = nx.Graph()
        # Edge qualifies: unequal order and standard_order != 0.
        G.add_edge(1, 2, order=(1, 2), standard_order=1)
        # Edge does not qualify: equal order.
        G.add_edge(2, 3, order=(1, 1), standard_order=1)
        # Edge qualifies.
        G.add_edge(3, 4, order=(2, 3), standard_order=1)
        # Edge does not qualify because standard_order == 0.
        G.add_edge(1, 3, order=(1, 2), standard_order=0)
        result = set(RadiusExpand.find_unequal_order_edges(G))
        expected = {1, 2, 3, 4}
        self.assertEqual(result, expected)

    def test_extract_subgraph(self):
        # Create a test graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])

        # Node indices for subgraph extraction
        node_indices = [2, 3]

        # Extract subgraph and test
        subgraph = RadiusExpand.extract_subgraph(G, node_indices)
        self.assertTrue(nx.is_isomorphic(subgraph, nx.Graph([(2, 3)])))

    def test_longest_radius_extension(self):
        # Create a graph where a chain of edges with standard_order == 0 exists.
        G = nx.Graph()
        G.add_edge(0, 1, standard_order=0)
        G.add_edge(1, 2, standard_order=0)
        # This edge is blocked because standard_order != 0.
        G.add_edge(2, 3, standard_order=1)
        # Compute longest extension from node 0.
        path = RadiusExpand.longest_radius_extension(G, [0])
        self.assertEqual(path, [0, 1, 2])

        # Add an alternative branch: from node 1 to node 4.
        G.add_edge(1, 4, standard_order=0)
        # Now, DFS might find either [0, 1, 2] or [0, 1, 4]. Both have length 3.
        path2 = RadiusExpand.longest_radius_extension(G, [0])
        self.assertEqual(len(path2), 3)
        self.assertTrue(set(path2).issubset(set(G.nodes())))

    def test_remove_normal_edges(self):
        # Build a graph with a custom attribute 'weight' and remove those edges where weight == 0.
        G = nx.Graph()
        G.add_edge(1, 2, weight=0)
        G.add_edge(2, 3, weight=1)
        G.add_edge(3, 4, weight=0)
        G.add_edge(4, 5, weight=2)
        modified = RadiusExpand.remove_normal_edges(G, "weight")
        # Expected to keep only edges where weight is not 0: (2,3) and (4,5).
        expected_edges = {(2, 3), (4, 5)}
        self.assertEqual(set(modified.edges()), expected_edges)

    def test_extract_k(self):
        its = self.data[0]["ITS"]
        rc = get_rc(its)
        context = RadiusExpand.extract_k(its, 0)
        self.assertTrue(graph_isomorphism(rc, context, use_defaults=True))

        # Test with a positive n_knn value.
        context = RadiusExpand.extract_k(its, 1)
        self.assertTrue(subgraph_isomorphism(rc, context))
        self.assertTrue(subgraph_isomorphism(context, its))

        # Test with n_knn == -1 to automatically determine neighbor expansion.
        context = RadiusExpand.extract_k(its, -1)
        self.assertTrue(graph_isomorphism(its, context, use_defaults=True))

    def test_context_extraction(self):
        data = self.data[0]
        result = RadiusExpand.context_extraction(
            data, its_key="ITS", context_key="K", n_knn=1
        )
        self.assertIn("K", result)
        self.assertTrue(subgraph_isomorphism(result["RC"], result["K"]))
        self.assertTrue(subgraph_isomorphism(result["K"], result["ITS"]))

    def test_paralle_context_extraction(self):
        data_list = self.data[0:3]
        results = RadiusExpand.paralle_context_extraction(
            data_list, its_key="ITS", context_key="K", n_jobs=1, verbose=0, n_knn=1
        )
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIn("K", res)
            self.assertTrue(subgraph_isomorphism(res["RC"], res["K"]))
            self.assertTrue(subgraph_isomorphism(res["K"], res["ITS"]))


if __name__ == "__main__":
    unittest.main()
