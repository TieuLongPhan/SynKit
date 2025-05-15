import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import rsmi_to_its, its_to_gml
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine
import networkx as nx

# Determine if the rule backend is available
try:
    from mod import ruleGMLString  # noqa: F401

    RULE_AVAILABLE = True
except ImportError:
    RULE_AVAILABLE = False


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

        self.small = """rule [
            ruleID "Small"
            left [
                node [ id 1 label "H" ]
                node [ id 2 label "O" ]
                edge [ source 1 target 2 label "-" ]
            ]
            right [
                node [ id 1 label "H+" ]
                node [ id 2 label "O-" ]
            ]
        ]"""
        self.large = """rule [
            ruleID "Large"
            left [
                node [ id 1 label "H" ]
                node [ id 2 label "O" ]
                edge [ source 1 target 2 label "-" ]
            ]
            context [
                node [ id 3 label "C" ]
                edge [ source 2 target 3 label "-" ]
            ]
            right [
                node [ id 1 label "H+" ]
                node [ id 2 label "O-" ]
            ]
        ]"""

    def test_full_graph_isomorphism_true(self):

        self.assertTrue(self.gm.isomorphic(self.graphs[0]["RC"], self.graphs[3]["RC"]))

    def test_full_graph_isomorphism_false(self):
        self.assertFalse(self.gm.isomorphic(self.graphs[0]["RC"], self.graphs[1]["RC"]))

    @unittest.skipUnless(RULE_AVAILABLE, "requires ruleGMLString")
    def test_rule_isomorphism(self):
        gm_rule = GraphMatcherEngine(
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            wl1_filter=False,
            max_mappings=None,
            backend="mod",
        )
        self.assertTrue(gm_rule._isomorphic_rule(self.small, self.small))
        self.assertTrue(gm_rule._isomorphic_rule(self.large, self.large))
        self.assertFalse(gm_rule._isomorphic_rule(self.small, self.large))

    @unittest.skipUnless(RULE_AVAILABLE, "requires ruleGMLString")
    def test_full_graph_isomorphism_true_use_rule(self):
        # Rule backend should agree on isomorphic graphs
        gm_rule = GraphMatcherEngine(
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            wl1_filter=False,
            max_mappings=None,
            backend="mod",
        )
        gml1 = its_to_gml(self.rc)
        # permute same as before
        mapping = dict(zip(list(self.rc.nodes), reversed(list(self.rc.nodes))))
        perm = nx.relabel_nodes(self.rc, mapping)
        gml2 = its_to_gml(perm)
        self.assertTrue(gm_rule.isomorphic(gml1, gml2))

    @unittest.skipUnless(RULE_AVAILABLE, "requires ruleGMLString")
    def test_full_graph_isomorphism_false_use_rule(self):
        # Rule backend should detect non-isomorphic
        gm_rule = GraphMatcherEngine(
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            wl1_filter=False,
            max_mappings=None,
            backend="mod",
        )
        # compare original vs with one node removed
        g0 = self.rc.copy()
        g_mod = g0.copy()
        g_mod.remove_node(next(iter(g_mod.nodes)))
        gml1 = its_to_gml(g0)
        gml3 = its_to_gml(g_mod)
        self.assertFalse(gm_rule.isomorphic(gml1, gml3))

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

    def test_available_backends(self):
        # available_backends should list at least 'nx'
        backends = GraphMatcherEngine.available_backends()
        self.assertIn("nx", backends)


if __name__ == "__main__":
    unittest.main()
