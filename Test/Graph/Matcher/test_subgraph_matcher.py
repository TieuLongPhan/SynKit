import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Matcher.subgraph_matcher import SubgraphMatch, SubgraphSearchEngine

# Determine if the rule backend is available
try:
    from mod import ruleGMLString  # noqa: F401

    RULE_AVAILABLE = True
except ImportError:
    RULE_AVAILABLE = False


class TestSubgraphMatch(unittest.TestCase):

    def setUp(self):
        # Load test graphs and reaction center
        self.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")
        rsmi = (
            "[F:1][C:2]([F:3])([F:4])[c:5]1[cH:6][n:8][c:9]([Cl:10])"
            + "[c:11]([Br:12])[cH:7]1.[O:13]([CH2:14][Na:16])[H:15]"
            + ">>[Cl:10][Na:16].[F:1][C:2]([F:3])([F:4])[c:5]1[cH:6]"
            + "[n:8][c:9]([O:13][CH2:14][H:15])[c:11]([Br:12])[cH:7]1"
        )
        self.its = rsmi_to_its(rsmi)
        self.rc = get_rc(self.its)

        self.gm = SubgraphMatch()
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

    def test_graph_subgraph_morphism_true(self):
        is_sub = self.gm.is_subgraph(
            self.graphs[0]["RC"],
            self.graphs[0]["RC"],
            node_label_names=["element", "charge"],
            edge_attribute="order",
            backend="nx",
            check_type="induced",
        )
        self.assertTrue(is_sub, 0)

    def test_graph_subgraph_morphism_false(self):
        is_sub = self.gm.is_subgraph(
            self.graphs[0]["RC"],
            self.graphs[1]["RC"],
            node_label_names=["element", "charge"],
            edge_attribute="order",
            backend="nx",
            check_type="induced",
        )
        self.assertFalse(is_sub, 0)

    def test_nx_subgraph_morphism(self):
        result = self.gm.subgraph_isomorphism(
            self.rc,
            self.its,
            node_label_names=["element", "charge"],
            edge_attribute="order",
            check_type="mono",
        )
        self.assertTrue(result)
        result = self.gm.subgraph_isomorphism(
            self.rc,
            self.its,
            node_label_names=["element", "charge"],
            edge_attribute="order",
            check_type="induced",
        )
        self.assertFalse(result)

    @unittest.skipUnless(RULE_AVAILABLE, "requires ruleGMLString")
    def test_rule_isomorphism_monomorphism(self):
        # small is a subgraph of large
        self.assertTrue(self.gm.rule_subgraph_morphism(self.small, self.large))
        # large is not a subgraph of small
        self.assertFalse(self.gm.rule_subgraph_morphism(self.large, self.small))


class TestSubGraphSearchEngine(unittest.TestCase):

    def setUp(self):
        # Load test graphs and reaction center
        self.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")

        self.gm = SubgraphSearchEngine()

    def test_graph_subgraph_morphism_true(self):
        mapping = self.gm.find_subgraph_mappings(
            self.graphs[0]["RC"],
            self.graphs[0]["RC"],
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
        )
        self.assertGreater(len(mapping), 0)

    def test_graph_subgraph_morphism_false(self):
        mapping = self.gm.find_subgraph_mappings(
            self.graphs[0]["RC"],
            self.graphs[1]["RC"],
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
        )
        self.assertEqual(len(mapping), 0)


if __name__ == "__main__":
    unittest.main()
