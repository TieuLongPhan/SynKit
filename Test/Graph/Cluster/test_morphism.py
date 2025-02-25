import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.Graph.Cluster.morphism import graph_isomorphism, rule_isomorphism


class TestRule(unittest.TestCase):

    def setUp(self):
        self.graphs = load_from_pickle("Data/Testcase/graph.pkl.gz")

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

        self.gml = (
            "rule [\n"
            '   ruleID "rule"\n'
            "   left [\n"
            '      edge [ source 1 target 4 label "-" ]\n'
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 2 target 3 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "C" ]\n'
            '      node [ id 4 label "H" ]\n'
            '      node [ id 2 label "C" ]\n'
            '      node [ id 3 label "O" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 2 label "=" ]\n'
            '      edge [ source 4 target 3 label "-" ]\n'
            "   ]\n"
            "]"
        )

        self.gml_h = (
            "rule [\n"
            '   ruleID "rule"\n'
            "   left [\n"
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 1 target 3 label "-" ]\n'
            '      edge [ source 3 target 4 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "C" ]\n'
            '      node [ id 2 label "H" ]\n'
            '      node [ id 3 label "C" ]\n'
            '      node [ id 4 label "O" ]\n'
            '      node [ id 5 label "H" ]\n'
            '      node [ id 6 label "H" ]\n'
            '      node [ id 7 label "H" ]\n'
            '      node [ id 8 label "H" ]\n'
            '      node [ id 9 label "H" ]\n'
            '      edge [ source 1 target 5 label "-" ]\n'
            '      edge [ source 1 target 6 label "-" ]\n'
            '      edge [ source 3 target 7 label "-" ]\n'
            '      edge [ source 3 target 8 label "-" ]\n'
            '      edge [ source 4 target 9 label "-" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 3 label "=" ]\n'
            '      edge [ source 2 target 4 label "-" ]\n'
            "   ]\n"
            "]"
        )

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

    def test_rule_isomorphism_isomorphism(self):
        self.assertTrue(rule_isomorphism(self.small, self.small))
        self.assertTrue(rule_isomorphism(self.large, self.large))
        self.assertFalse(rule_isomorphism(self.small, self.large))

    def test_rule_isomorphism_monomorphism(self):
        # small is a subgraph of large
        self.assertTrue(rule_isomorphism(self.small, self.large, "monomorphism"))
        # large is not a subgraph of small
        self.assertFalse(rule_isomorphism(self.large, self.small, "monomorphism"))


if __name__ == "__main__":
    unittest.main()
