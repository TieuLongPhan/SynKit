import unittest
from synkit.Graph.Cluster.rule_morphism import rule_isomorphism, rule_subgraph_morphism


class TestRulMorphism(unittest.TestCase):

    def setUp(self):

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

    def test_rule_isomorphism_isomorphism(self):
        self.assertTrue(rule_isomorphism(self.small, self.small))
        self.assertTrue(rule_isomorphism(self.large, self.large))
        self.assertFalse(rule_isomorphism(self.small, self.large))

    def test_rule_isomorphism_monomorphism(self):
        # small is a subgraph of large
        self.assertTrue(rule_subgraph_morphism(self.small, self.large))
        # large is not a subgraph of small
        self.assertFalse(rule_subgraph_morphism(self.large, self.small))


if __name__ == "__main__":
    unittest.main()
