import unittest
import networkx as nx
from synkit.IO.data_io import load_from_pickle
from synkit.IO.nx_to_gml import NXToGML


class TestNXToGML(unittest.TestCase):

    def setUp(self) -> None:
        self.data = load_from_pickle("Data/Testcase/test.pkl.gz")[0]

    def test_charge_to_string(self):
        self.assertEqual(NXToGML._charge_to_string(3), "3+")
        self.assertEqual(NXToGML._charge_to_string(-2), "2-")
        self.assertEqual(NXToGML._charge_to_string(0), "")

    def test_find_changed_nodes(self):
        G1 = nx.Graph()
        G1.add_node(1, element="C", charge=0)
        G2 = nx.Graph()
        G2.add_node(1, element="C", charge=1)
        changed_nodes = NXToGML._find_changed_nodes(G1, G2, ["charge"])
        self.assertEqual(changed_nodes, [1])

    def test_convert_graph_to_gml_context(self):
        G = nx.Graph()
        G.add_node(1, element="C")
        G.add_node(2, element="H")
        changed_node_ids = [2]
        gml_str = NXToGML._convert_graph_to_gml(G, "context", changed_node_ids)
        expected_str = '   context [\n      node [ id 1 label "C" ]\n   ]\n'
        self.assertEqual(gml_str, expected_str)

    def test_convert_graph_to_gml_left_right(self):
        G = nx.Graph()
        G.add_node(1, element="C", charge=1)
        G.add_node(2, element="H", charge=0)
        G.add_edge(1, 2, order=2)
        changed_node_ids = [1]
        gml_str = NXToGML._convert_graph_to_gml(G, "left", changed_node_ids)
        expected_str = (
            '   left [\n      edge [ source 1 target 2 label "=" ]'
            + '\n      node [ id 1 label "C+" ]\n   ]\n'
        )
        self.assertEqual(gml_str, expected_str)

    def test_rules_grammar(self):
        L, R, K = self.data["GraphRules"]
        changed_node_ids = NXToGML._find_changed_nodes(L, R, ["charge"])
        rule_name = "test_rule"
        gml_str = NXToGML._rule_grammar(
            L, R, K, rule_name, changed_node_ids, explicit_hydrogen=False
        )
        expected_str = (
            "rule [\n"
            '   ruleID "test_rule"\n'
            "   left [\n"
            '      edge [ source 11 target 35 label "-" ]\n'
            '      edge [ source 28 target 29 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 11 label "N" ]\n'
            '      node [ id 35 label "H" ]\n'
            '      node [ id 28 label "C" ]\n'
            '      node [ id 29 label "Br" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 11 target 28 label "-" ]\n'
            '      edge [ source 35 target 29 label "-" ]\n'
            "   ]\n"
            "]"
        )
        self.assertEqual(gml_str, expected_str)

    def test_transform(self):
        graph_rules = self.data["GraphRules"]
        gml_str = NXToGML.transform(graph_rules, rule_name="test_rule", reindex=True)
        expected_str = (
            "rule [\n"
            '   ruleID "test_rule"\n'
            "   left [\n"
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 3 target 4 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "N" ]\n'
            '      node [ id 2 label "H" ]\n'
            '      node [ id 3 label "C" ]\n'
            '      node [ id 4 label "Br" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 3 label "-" ]\n'
            '      edge [ source 2 target 4 label "-" ]\n'
            "   ]\n"
            "]"
        )
        self.assertEqual(gml_str, expected_str)


if __name__ == "__main__":
    unittest.main()
