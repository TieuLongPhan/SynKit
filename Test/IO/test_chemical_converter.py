import unittest
import networkx as nx
from rdkit.Chem import rdChemReactions
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.IO import (
    smiles_to_graph,
    rsmi_to_graph,
    graph_to_smi,
    graph_to_rsmi,
    smart_to_gml,
    gml_to_smart,
    its_to_gml,
    gml_to_its,
    rsmi_to_its,
    its_to_rsmi,
    rsmarts_to_rsmi,
    rsmi_to_rsmarts,
)
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine
from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
import importlib.util

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


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
            "[CH3:1][CH2:2][OH:3]",
            False,
            True,
            True,
        )
        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(result.number_of_nodes(), 3)

    def test_smiles_to_graph_invalid(self):
        # Test converting an invalid SMILES string to a graph
        result = smiles_to_graph(
            "invalid_smiles",
            True,
            False,
            False,
        )
        self.assertIsNone(result)

    def test_rsmi_to_graph_valid(self):
        reactants_graph, products_graph = rsmi_to_graph(self.rsmi, sanitize=True)
        self.assertIsInstance(reactants_graph, nx.Graph)
        self.assertEqual(reactants_graph.number_of_nodes(), 4)
        self.assertIsInstance(products_graph, nx.Graph)
        self.assertEqual(products_graph.number_of_nodes(), 4)

    def test_rsmi_to_graph_invalid(self):
        # Test handling of invalid RSMI format
        result = rsmi_to_graph("invalid_format")
        self.assertEqual((None, None), result)

    def test_graph_to_smi_valid(self):
        g = smiles_to_graph(
            "[CH2:1]=[CH2:2]",
            sanitize=True,
            drop_non_aam=True,
            use_index_as_atom_map=True,
        )
        smi = graph_to_smi(g)
        self.assertEqual(smi, "[CH2:1]=[CH2:2]")

    def test_graph_to_smi_invalid(self):
        g = smiles_to_graph(
            "[CH3:1]=[CH2:2]",
            sanitize=False,
            drop_non_aam=True,
            use_index_as_atom_map=True,
        )
        smi = graph_to_smi(g)
        self.assertIsNone(smi)
        smi = graph_to_smi(g, sanitize=False)  # unsanitize can return invalid smiles
        self.assertEqual(smi, "[CH3:1]=[CH2:2]")

    def test_graph_to_rsmi(self):
        r, p = rsmi_to_graph(self.rsmi, sanitize=True)
        its = ITSConstruction().ITSGraph(r, p)
        rsmi = graph_to_rsmi(
            r,
            p,
            its,
            explicit_hydrogen=False,
        )
        self.assertIsInstance(rsmi, str)
        self.assertTrue(AAMValidator.smiles_check(rsmi, self.rsmi, "ITS"))

    def test_graph_to_invalid_rsmi(self):
        r, p = rsmi_to_graph(
            "[CH3:1]=[CH2:2].[H:3][H:4]>>[CH3:1][CH3:2]", sanitize=False
        )
        its = ITSConstruction().ITSGraph(r, p)
        rsmi = graph_to_rsmi(
            r,
            p,
            its,
            explicit_hydrogen=False,
        )
        self.assertIsNone(rsmi, str)

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_smart_to_gml(self):
        result = smart_to_gml(self.rsmi, core=False, sanitize=True, reindex=False)
        self.assertIsInstance(result, str)
        self.assertEqual(result, self.gml)

        result = smart_to_gml(self.rsmi, core=False, sanitize=True, reindex=True)
        self.assertTrue(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(result, self.gml)
        )

    def test_gml_to_rsmi(self):
        smarts = gml_to_smart(self.gml_h)
        self.assertIsInstance(smarts, str)
        self.assertTrue(AAMValidator.smiles_check(smarts, self.rsmi, "ITS"))

    def test_gml_to_smart(self):
        rsmi = gml_to_smart(self.gml_h, useSmiles=True)
        smarts = gml_to_smart(self.gml_h, useSmiles=False)
        self.assertIsInstance(smarts, str)
        self.assertNotEqual(rsmi, smarts)
        self.assertTrue(AAMValidator.smiles_check(smarts, self.rsmi, "ITS"))
        self.assertTrue(AAMValidator.smiles_check(smarts, rsmi, "ITS"))

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_smart_to_gml_explicit_hydrogen(self):
        rsmi = "[CH2:1]([H:4])[CH2:2][OH:3]>>[CH2:1]=[CH2:2].[H:4][OH:3]"
        gml = smart_to_gml(rsmi, explicit_hydrogen=True, core=False, sanitize=True)
        self.assertFalse(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(gml, self.gml)
        )
        self.assertTrue(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(gml, self.gml_h)
        )

    def test_gml_to_smart_explicit_hydrogen(self):
        smart = gml_to_smart(self.gml_h, explicit_hydrogen=True)
        expect = (
            "[C:1]([H:2])([C:3]([O:4][H:9])([H:7])[H:8])([H:5])[H:6]"
            + ">>[C:1](=[C:3]([H:7])[H:8])([H:5])[H:6].[H:2][O:4][H:9]"
        )
        self.assertFalse(AAMValidator.smiles_check(smart, self.rsmi, "ITS"))
        self.assertTrue(AAMValidator.smiles_check(smart, expect, "ITS"))

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_its_to_gml(self):
        its = rsmi_to_its(self.rsmi)
        gml_1 = its_to_gml(its)
        gml_2 = smart_to_gml(self.rsmi)
        self.assertTrue(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(gml_1, gml_2)
        )

    def test_gml_to_its(self):
        gml = smart_to_gml(self.rsmi)
        its_1 = rsmi_to_its(self.rsmi)
        its_2 = gml_to_its(gml)
        self.assertTrue(graph_isomorphism(its_1, its_2))

    def test_rsmi_to_its(self):
        its_1 = rsmi_to_its(self.rsmi)
        r, p = rsmi_to_graph(self.rsmi)
        its_2 = ITSConstruction().ITSGraph(r, p)
        self.assertTrue(graph_isomorphism(its_1, its_2))

    def test_rsmi_to_rc(self):
        smart = "[CH3:5][CH:1]=[CH2:2].[H:3][H:4]>>[CH3:5][CH2:1][CH3:2]"
        its = rsmi_to_its(smart)
        rc = rsmi_to_its(smart, core=True)
        self.assertFalse(graph_isomorphism(its, rc))

    def test_its_to_rsmi(self):
        smart = "[CH3:5][CH:1]=[CH2:2].[H:3][H:4]>>[CH3:5][CH:1]([H:3])[CH2:2][H:4]"
        its = rsmi_to_its(smart)
        new_smart = its_to_rsmi(its)
        self.assertEqual(
            CanonRSMI().canonicalise(smart).canonical_rsmi,
            CanonRSMI().canonicalise(new_smart).canonical_rsmi,
        )

    def test_rsmi_to_rsmarts_and_back(self):
        rsmi = "[H:3][O:4].[N:1][C:2]>>[C:2][O:4].[N:1][H:3]"

        # Convert to SMARTS
        rsmarts = rsmi_to_rsmarts(rsmi)
        self.assertIsInstance(rsmarts, str)
        self.assertIn(">>", rsmarts)

        # Convert back to SMILES
        back_rsmi = rsmarts_to_rsmi(rsmarts)
        self.assertIsInstance(back_rsmi, str)
        self.assertIn(">>", back_rsmi)

        # Validate structure equivalence via reaction object
        rxn_orig = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)
        rxn_back = rdChemReactions.ReactionFromSmarts(back_rsmi, useSmiles=True)

        self.assertEqual(
            rxn_orig.GetNumReactantTemplates(), rxn_back.GetNumReactantTemplates()
        )
        self.assertEqual(
            rxn_orig.GetNumProductTemplates(), rxn_back.GetNumProductTemplates()
        )


if __name__ == "__main__":
    unittest.main()
