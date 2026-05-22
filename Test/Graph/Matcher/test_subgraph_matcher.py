import unittest
import networkx as nx
from synkit.IO.data_io import load_from_pickle
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.Matcher.subgraph_matcher import (
    SubgraphMatch,
    SubgraphSearchEngine,
    diagnose_candidate_node_match,
    electron_aware_edge_match,
    resolve_template_match_attrs,
)

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

    def test_electron_aware_node_matching(self):
        host = nx.Graph()
        host.add_node(1, element="O", lone_pairs=3, radical=0, hcount=1)

        pattern = nx.Graph()
        pattern.add_node(10, element="O", lone_pairs=2, radical=0, hcount=0)

        matches = self.gm.find_subgraph_mappings(
            host,
            pattern,
            node_attrs=["element", "lone_pairs", "radical"],
            edge_attrs=[],
        )
        self.assertEqual(matches, [{10: 1}])

    def test_electron_aware_node_matching_rejects_low_lone_pairs(self):
        host = nx.Graph()
        host.add_node(1, element="O", lone_pairs=1, radical=0, hcount=0)

        pattern = nx.Graph()
        pattern.add_node(10, element="O", lone_pairs=2, radical=0, hcount=0)

        matches = self.gm.find_subgraph_mappings(
            host,
            pattern,
            node_attrs=["element", "lone_pairs", "radical"],
            edge_attrs=[],
        )
        self.assertEqual(matches, [])

    def test_resolve_template_match_attrs_keeps_legacy_template_legacy(self):
        pattern = nx.Graph()
        pattern.add_node(1, element="O", charge=0)
        pattern.add_edge(1, 2, order=1.0)

        node_attrs, edge_attrs = resolve_template_match_attrs(pattern)

        self.assertEqual(node_attrs, ["element", "charge"])
        self.assertEqual(edge_attrs, ["order"])

    def test_resolve_template_match_attrs_uses_new_template_fields(self):
        pattern = nx.Graph()
        pattern.add_node(
            1,
            element="O",
            charge=0,
            aromatic=False,
            hcount=0,
            lone_pairs=2,
            radical=0,
        )
        pattern.add_node(
            2,
            element="C",
            charge=0,
            aromatic=False,
            hcount=3,
            lone_pairs=0,
            radical=0,
        )
        pattern.add_edge(1, 2, order=2.0, sigma_order=1.0, pi_order=1.0)

        node_attrs, edge_attrs = resolve_template_match_attrs(pattern)

        self.assertEqual(
            node_attrs,
            [
                "element",
                "charge",
                "aromatic",
                "hcount",
                "lone_pairs",
                "radical",
            ],
        )
        self.assertEqual(edge_attrs, ["order", "sigma_order", "pi_order"])

    def test_resolve_template_match_attrs_uses_aromatic_n_pi_role(self):
        pattern = nx.Graph()
        pattern.add_node(
            1,
            element="N",
            charge=0,
            aromatic=True,
            hcount=0,
            lone_pairs=1,
            radical=0,
            aromatic_n_pi_count=1,
        )

        node_attrs, _ = resolve_template_match_attrs(pattern)

        self.assertIn("aromatic_n_pi_count", node_attrs)

    def test_diagnose_candidate_node_match_reports_electron_reason(self):
        diagnostic = diagnose_candidate_node_match(
            {"element": "O", "lone_pairs": 1, "radical": 0},
            {"element": "O", "lone_pairs": 2, "radical": 1},
            ["element", "lone_pairs", "radical"],
        )

        self.assertFalse(diagnostic["matched"])
        self.assertEqual(
            diagnostic["reasons"],
            [
                "lone_pairs: host 1 < pattern 2",
                "radical: host 0 != pattern 1",
            ],
        )

    def test_electron_aware_edge_matching_ignores_aromatic_kekule_phase(self):
        self.assertTrue(
            electron_aware_edge_match(
                {"order": 1.5, "sigma_order": 1.0, "pi_order": 1.0},
                {"order": 1.5, "sigma_order": 1.0, "pi_order": 0.0},
                ["order", "sigma_order", "pi_order"],
            )
        )

    def test_electron_aware_edge_matching_keeps_non_aromatic_sigma_pi_exact(self):
        self.assertFalse(
            electron_aware_edge_match(
                {"order": 2.0, "sigma_order": 1.0, "pi_order": 1.0},
                {"order": 2.0, "sigma_order": 1.0, "pi_order": 0.0},
                ["order", "sigma_order", "pi_order"],
            )
        )


if __name__ == "__main__":
    unittest.main()
