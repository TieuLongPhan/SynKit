import unittest
from synkit.IO.data_io import load_from_pickle
from synkit.IO.gml_to_nx import GMLToNX
from synkit.IO.nx_to_gml import NXToGML
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism


class TestGMLToNX(unittest.TestCase):

    def setUp(self) -> None:
        data = load_from_pickle("Data/Testcase/test.pkl.gz")[0]
        self.ground_truth_its = data["ITSGraph"]
        self.ground_truth_rc = data["GraphRules"]
        self.rule_its = NXToGML.transform(
            self.ground_truth_its, explicit_hydrogen=False
        )
        self.rule_rc = NXToGML.transform(self.ground_truth_rc, explicit_hydrogen=False)
        self.parser = GMLToNX(gml_text="")
        gml_formatted_str = (
            "rule [\n"
            '   ruleID "Test"\n'
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
        self.parser_gml = GMLToNX(gml_formatted_str)

    def test_parse_element(self):
        """
        Test the parsing of nodes and edges from the provided GML string.
        """
        # Manually parse elements for testing
        self.parser_gml._parse_element('node [ id 11 label "N" ]', "context")
        self.parser_gml._parse_element('edge [ source 11 target 35 label "-" ]', "left")

        expected_node = (
            "11",
            {"element": "N", "charge": 0, "atom_map": 11, "hcount": 0},
        )
        expected_edge = (11, 35, {"order": 1.0})

        actual_node = self.parser_gml.graphs["context"].nodes(data=True)[11]
        actual_edge = self.parser_gml.graphs["left"][11][35]

        self.assertEqual(
            expected_node[1],
            actual_node,
            "Node attributes do not match expected values.",
        )
        self.assertEqual(
            expected_edge[2],
            actual_edge,
            "Edge attributes do not match expected values.",
        )

    def test_synchronize_nodes_and_edges(self):
        """
        Test the synchronization of nodes across different graph sections after parsing.
        """
        # Simulate parsing nodes into the context graph
        self.parser_gml.graphs["context"].add_node(
            11, element="N", charge=0, atom_map=11
        )
        self.parser_gml.graphs["context"].add_node(
            35, element="H", charge=0, atom_map=35
        )
        # Running synchronization
        self.parser_gml._synchronize_nodes_and_edges()
        # Checking if nodes are present in left and right graphs
        self.assertIn(11, self.parser_gml.graphs["left"])
        self.assertIn(35, self.parser_gml.graphs["right"])

    def test_extract_simple_element(self):
        """
        Test the extraction of an element without a charge.
        """
        element, charge = self.parser._extract_element_and_charge("C")
        self.assertEqual(element, "C")
        self.assertEqual(charge, 0)

    def test_extract_element_with_positive_charge(self):
        """
        Test the extraction of an element with a positive charge.
        """
        element, charge = self.parser._extract_element_and_charge("Na+")
        self.assertEqual(element, "Na")
        self.assertEqual(charge, 1)

    def test_extract_element_with_negative_charge(self):
        """
        Test the extraction of an element with a negative charge.
        """
        element, charge = self.parser._extract_element_and_charge("Cl-")
        self.assertEqual(element, "Cl")
        self.assertEqual(charge, -1)

    def test_extract_element_with_multi_digit_charge(self):
        """
        Test the extraction of an element with a multiple digit charge.
        """
        element, charge = self.parser._extract_element_and_charge("Mg2+")
        self.assertEqual(element, "Mg")
        self.assertEqual(charge, 2)

    def test_extract_element_with_no_charge_number(self):
        """
        Test the extraction where the charge number is implied as 1.
        """
        element, charge = self.parser._extract_element_and_charge("K+")
        self.assertEqual(element, "K")
        self.assertEqual(charge, 1)

    def test_transform(self):
        self.graphs_its = GMLToNX(self.rule_its).transform()
        self.graphs_rc = GMLToNX(self.rule_rc).transform()
        for key, _ in enumerate(self.graphs_its):
            self.assertTrue(
                graph_isomorphism(self.graphs_its[key], self.ground_truth_its[key])
            )
        for key, _ in enumerate(self.graphs_rc):
            self.assertTrue(
                graph_isomorphism(self.graphs_rc[key], self.ground_truth_rc[key])
            )
