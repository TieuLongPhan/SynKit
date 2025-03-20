import unittest
import networkx as nx

from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.aam_validator import AAMValidator
from synkit.IO.chem_converter import (
    smiles_to_graph,
    rsmi_to_graph,
    graph_to_rsmi,
    smart_to_gml,
    gml_to_smart,
    rsmi_to_its,
    its_to_gml,
    gml_to_its,
)
from synkit.Graph.Cluster.graph_morphism import graph_isomorphism
from synkit.Graph.Cluster.rule_morphism import rule_isomorphism


class TestChemicalConversions(unittest.TestCase):

    def setUp(self) -> None:
        self.rsmi = "[CH2:1]([H:4])[CH2:2][OH:3]>>[CH2:1]=[CH2:2].[H:4][OH:3]"
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
        self.std = Standardize()

    def test_smiles_to_graph_valid(self):
        # Test converting a valid SMILES to a graph
        result = smiles_to_graph(
            "[CH3:1][CH2:2][OH:3]", False, True, True, use_index_as_atom_map=True
        )
        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(result.number_of_nodes(), 3)

    def test_smiles_to_graph_invalid(self):
        # Test converting an invalid SMILES string to a graph
        result = smiles_to_graph(
            "invalid_smiles", True, False, False, use_index_as_atom_map=True
        )
        self.assertIsNone(result)

    def test_rsmi_to_graph_valid(self):
        # now sanitize = True still keeps hydrogen
        # Test converting valid reaction SMILES to graphs for reactants and products
        reactants_graph, products_graph = rsmi_to_graph(self.rsmi, sanitize=True)
        self.assertIsInstance(reactants_graph, nx.Graph)
        self.assertEqual(reactants_graph.number_of_nodes(), 4)
        self.assertIsInstance(products_graph, nx.Graph)
        self.assertEqual(products_graph.number_of_nodes(), 4)

    def test_rsmi_to_graph_invalid(self):
        # Test handling of invalid RSMI format
        result = rsmi_to_graph("invalid_format")
        self.assertEqual((None, None), result)

    def test_graph_to_rsmi(self):
        r, p = rsmi_to_graph(self.rsmi, sanitize=False)
        its = ITSConstruction().ITSGraph(r, p)
        rsmi = graph_to_rsmi(
            r,
            p,
            its,
            explicit_hydrogen=False,
        )
        self.assertIsInstance(rsmi, str)
        self.assertTrue(AAMValidator.smiles_check(rsmi, self.rsmi, "ITS"))

    def test_smart_to_gml(self):
        result = smart_to_gml(self.rsmi, core=False, sanitize=False, reindex=False)
        self.assertIsInstance(result, str)
        self.assertEqual(result, self.gml)

        result = smart_to_gml(self.rsmi, core=False, sanitize=False, reindex=True)
        self.assertTrue(rule_isomorphism(result, self.gml))

    def test_gml_to_smart(self):
        smarts, _ = gml_to_smart(self.gml_h)
        self.assertIsInstance(smarts, str)
        self.assertTrue(AAMValidator.smiles_check(smarts, self.rsmi, "ITS"))

    def test_smart_to_gml_explicit_hydrogen(self):
        rsmi = "[CH2:1]([H:4])[CH2:2][OH:3]>>[CH2:1]=[CH2:2].[H:4][OH:3]"
        gml = smart_to_gml(rsmi, explicit_hydrogen=True, core=False, sanitize=True)
        self.assertFalse(rule_isomorphism(gml, self.gml))
        self.assertTrue(rule_isomorphism(gml, self.gml_h))

    def test_gml_to_smart_explicit_hydrogen(self):
        smart, _ = gml_to_smart(self.gml_h, explicit_hydrogen=True)
        expect = (
            "[C:1]([H:2])([C:3]([O:4][H:9])([H:7])[H:8])([H:5])[H:6]"
            + ">>[C:1](=[C:3]([H:7])[H:8])([H:5])[H:6].[H:2][O:4][H:9]"
        )
        self.assertFalse(AAMValidator.smiles_check(smart, self.rsmi, "ITS"))
        self.assertTrue(AAMValidator.smiles_check(smart, expect, "ITS"))

    def test_rsmi_to_its(self):
        its_1 = rsmi_to_its(self.rsmi)
        r, p = rsmi_to_graph(self.rsmi)
        its_2 = ITSConstruction().ITSGraph(r, p)
        self.assertTrue(graph_isomorphism(its_1, its_2))

    def test_its_to_gml(self):
        its = rsmi_to_its(self.rsmi)
        gml_1 = its_to_gml(its)
        gml_2 = smart_to_gml(self.rsmi)
        self.assertTrue(rule_isomorphism(gml_1, gml_2))

    def test_gml_to_its(self):
        gml = smart_to_gml(self.rsmi)
        its_1 = rsmi_to_its(self.rsmi)
        its_2 = gml_to_its(gml)
        self.assertTrue(graph_isomorphism(its_1, its_2))


if __name__ == "__main__":
    unittest.main()
