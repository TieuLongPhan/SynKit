import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine
import networkx as nx


class TestGraphMatcherEngine(unittest.TestCase):

    def setUp(self):
        # Load test graphs and reaction center
        self.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")
        rsmi = (
            "[F:1][C:2]([F:3])([F:4])[c:5]1[cH:6][n:8][c:9]([Cl:10])"
            + "[c:11]([Br:12])[cH:7]1.[O:13]([CH2:14][Na:16])[H:15]"
            + ">>[Cl:10][Na:16].[F:1][C:2]([F:3])([F:4])[c:5]1[cH:6]"
            + "[n:8][c:9]([O:13][CH2:14][H:15])[c:11]([Br:12])[cH:7]1"
        )
        its = rsmi_to_its(rsmi)
        self.rc = get_rc(its)
        # GraphMatcherEngine for isomorphism and mapping tests
        self.gm = GraphMatcherEngine(
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            wl1_filter=False,
            max_mappings=None,
        )

    def test_full_graph_isomorphism_true(self):

        self.assertTrue(self.gm.isomorphic(self.graphs[0]["RC"], self.graphs[3]["RC"]))

    def test_full_graph_isomorphism_false(self):
        self.assertFalse(self.gm.isomorphic(self.graphs[0]["RC"], self.graphs[1]["RC"]))

    def test_subgraph_isomorphism_mappings(self):
        # Path of length 2 should be subgraph of a path of length 3
        host = self.graphs[0]["RC"]
        pattern = self.graphs[3]["RC"]
        mappings = self.gm.get_mappings(host, pattern)
        # should find at least one mapping of size 3
        self.assertTrue(isinstance(mappings, list))
        self.assertTrue(len(mappings) >= 1)
        m = mappings[0]

        self.assertEqual(set(m.keys()), set(pattern.nodes()))
        self.assertEqual(len(set(m.values())), 4)

    def test_edge_attribute_mismatch(self):
        # edge attribute mismatch should prevent isomorphism
        g1 = nx.Graph()
        g1.add_edge(1, 2, order=1)
        g1.nodes[1]["element"] = "C"
        g1.nodes[2]["element"] = "C"
        g1.nodes[1]["charge"] = 0
        g1.nodes[2]["charge"] = 0
        g2 = nx.Graph()
        g2.add_edge("a", "b", order=2)
        g2.nodes["a"]["element"] = "C"
        g2.nodes["b"]["element"] = "C"
        g2.nodes["a"]["charge"] = 0
        g2.nodes["b"]["charge"] = 0
        self.assertFalse(self.gm.isomorphic(g1, g2))

    def test_lone_pairs_use_host_greater_or_equal_semantics(self):
        host = nx.Graph()
        host.add_node(1, element="O", lone_pairs=3, radical=0, hcount=0)

        pattern = nx.Graph()
        pattern.add_node(10, element="O", lone_pairs=2, radical=0, hcount=0)

        gm = GraphMatcherEngine(
            node_attrs=["element", "lone_pairs", "radical"],
            edge_attrs=[],
            max_mappings=None,
        )
        self.assertEqual(gm.get_mappings(host, pattern), [{10: 1}])

    def test_radical_requires_exact_match(self):
        host = nx.Graph()
        host.add_node(1, element="O", lone_pairs=3, radical=1, hcount=0)

        pattern = nx.Graph()
        pattern.add_node(10, element="O", lone_pairs=2, radical=0, hcount=0)

        gm = GraphMatcherEngine(
            node_attrs=["element", "lone_pairs", "radical"],
            edge_attrs=[],
            max_mappings=None,
        )
        self.assertEqual(gm.get_mappings(host, pattern), [])

    def test_available_backends(self):
        self.assertEqual(GraphMatcherEngine.available_backends(), ["nx"])


if __name__ == "__main__":
    unittest.main()
