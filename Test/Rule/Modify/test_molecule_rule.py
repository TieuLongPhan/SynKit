import unittest
from synkit.Rule.Modify.molecule_rule import MoleculeRule
from synkit.Graph.Matcher import GraphMatcherEngine
import importlib

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


class TestMoleculeRule(unittest.TestCase):
    """
    Unit test class for testing the MoleculeRule class methods.
    """

    def setUp(self):
        """Set up the MoleculeRule instance before each test."""
        self.molecule_rule = MoleculeRule()

    def test_generate_atom_map_valid(self):
        """
        Test the generate_atom_map method with a valid SMILES string.
        """
        smiles = "CCO.CC"  # Ethanol and Ethane connected
        expected_output = "[CH3:1][CH2:2][OH:3].[CH3:4][CH3:5]"
        result = self.molecule_rule.generate_atom_map(smiles)
        self.assertEqual(result, expected_output)

    def test_generate_atom_map_invalid(self):
        """
        Test the generate_atom_map method with an invalid SMILES string.
        """
        smiles = "INVALID_SMILES"
        result = self.molecule_rule.generate_atom_map(smiles)
        self.assertIsNone(result)

    def test_generate_molecule_smart_valid(self):
        """
        Test the generate_molecule_smart method with a valid SMILES string.
        """
        smiles = "CCO.CC"  # Ethanol and Ethane connected
        expected_output = (
            "[CH3:1][CH2:2][OH:3].[CH3:4][CH3:5]>>[CH3:1][CH2:2][OH:3].[CH3:4][CH3:5]"
        )
        result = self.molecule_rule.generate_molecule_smart(smiles)
        self.assertEqual(result, expected_output)

    def test_generate_molecule_smart_invalid(self):
        """
        Test the generate_molecule_smart method with an invalid SMILES string.
        """
        smiles = "INVALID_SMILES"
        result = self.molecule_rule.generate_molecule_smart(smiles)
        self.assertIsNone(result)

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_generate_molecule_rule_valid(self):
        """
        Test the generate_molecule_rule method with a valid SMILES string.
        """
        smiles = "CC"  # Ethanol and Ethane connected
        expected_output = (
            "rule [\n"
            '   ruleID "molecule"\n'
            "   left [\n"
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "C" ]\n'
            '      node [ id 2 label "C" ]\n'
            '      node [ id 3 label "H" ]\n'
            '      node [ id 4 label "H" ]\n'
            '      node [ id 5 label "H" ]\n'
            '      node [ id 6 label "H" ]\n'
            '      node [ id 7 label "H" ]\n'
            '      node [ id 8 label "H" ]\n'
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 1 target 3 label "-" ]\n'
            '      edge [ source 1 target 4 label "-" ]\n'
            '      edge [ source 1 target 5 label "-" ]\n'
            '      edge [ source 2 target 6 label "-" ]\n'
            '      edge [ source 2 target 7 label "-" ]\n'
            '      edge [ source 2 target 8 label "-" ]\n'
            "   ]\n"
            "   right [\n"
            "   ]\n"
            "]"
        )

        result = self.molecule_rule.generate_molecule_rule(smiles)
        self.assertTrue(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(result, expected_output)
        )

    def test_generate_molecule_rule_invalid(self):
        """
        Test the generate_molecule_rule method with an invalid SMILES string.
        """
        smiles = "INVALID_SMILES"
        result = self.molecule_rule.generate_molecule_rule(smiles)
        self.assertIsNone(result)

    def test_generate_molecule_rule_with_custom_parameters(self):
        """
        Test the generate_molecule_rule method with custom parameters (name, sanitize, explicit_hydrogen).
        """
        smiles = "CCO.CC"  # Ethanol and Ethane connected
        result = self.molecule_rule.generate_molecule_rule(
            smiles, name="custom_rule", sanitize=False, explicit_hydrogen=False
        )
        self.assertIsNotNone(result)

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_remove_edges_from_left_right(self):
        rule = (
            "rule [\n"
            '   ruleID "molecule"\n'
            "   left [\n"
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 2 target 3 label "-" ]\n'
            '      edge [ source 4 target 5 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 2 target 3 label "-" ]\n'
            '      edge [ source 4 target 5 label "-" ]\n'
            "   ]\n"
            "]"
        )

        expected = (
            "rule [\n"
            '   ruleID "molecule"\n'
            "   left [\n"
            "   ]\n"
            "   context [\n"
            "   ]\n"
            "   right [\n"
            "   ]\n"
            "]"
        )
        gml = self.molecule_rule.remove_edges_from_left_right(rule)
        self.assertTrue(
            GraphMatcherEngine(backend="mod")._isomorphic_rule(gml, expected)
        )


if __name__ == "__main__":
    unittest.main()
