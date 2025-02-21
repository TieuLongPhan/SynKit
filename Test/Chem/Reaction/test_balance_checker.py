import unittest
from synkit.Chem.Reaction.balance_check import BalanceReactionCheck


class TestBalanceReactionCheck(unittest.TestCase):

    def test_parse_input_string(self):
        input_data = "C>>O"
        expected_output = [{"reactions": "C>>O"}]
        result = BalanceReactionCheck.parse_input(input_data)
        self.assertEqual(result, expected_output)

    def test_parse_input_list_of_strings(self):
        input_data = ["C>>O", "[HH]+O=O>>O"]
        expected_output = [{"reactions": "C>>O"}, {"reactions": "[HH]+O=O>>O"}]
        result = BalanceReactionCheck.parse_input(input_data)
        self.assertEqual(result, expected_output)

    def test_parse_input_list_of_dicts(self):
        input_data = [{"reactions": "C>>O"}, {"reactions": "[HH]+O=O>>O"}]
        expected_output = [{"reactions": "C>>O"}, {"reactions": "[HH]+O=O>>O"}]
        result = BalanceReactionCheck.parse_input(input_data)
        self.assertEqual(result, expected_output)

    def test_parse_input_mixed_list(self):
        input_data = ["C>>O", {"reactions": "[HH]+O=O>>O"}]
        expected_output = [{"reactions": "C>>O"}, {"reactions": "[HH]+O=O>>O"}]
        result = BalanceReactionCheck.parse_input(input_data)
        self.assertEqual(result, expected_output)

    def test_parse_input_invalid_type(self):
        input_data = 123  # Not a string or list
        with self.assertRaises(ValueError):
            BalanceReactionCheck.parse_input(input_data)

    def test_parse_reaction(self):
        reaction_smiles = "C+C>>O+O"
        expected_output = ("C+C", "O+O")
        result = BalanceReactionCheck.parse_reaction(reaction_smiles)
        self.assertEqual(result, expected_output)

    def test_parse_reaction_single(self):
        reaction_smiles = "CC>>OO"
        expected_output = ("CC", "OO")
        result = BalanceReactionCheck.parse_reaction(reaction_smiles)
        self.assertEqual(result, expected_output)

    def test_single_smiles_balanced(self):
        """Test a single balanced reaction in SMILES format."""
        smiles = (
            "Clc1cnc2nc1Nc1ccc(OCCC3CCNCC3)c(c1)CCc1cncc(c1)N2.O=C=NCc1ccco1"
            + ">>O=C(NCc1ccco1)N1CCC(CCOc2ccc3cc2CCc2cncc(c2)Nc2ncc(Cl)c(n2)N3)CC1"
        )
        checker = BalanceReactionCheck()
        balanced = checker.rsmi_balance_check(smiles)
        self.assertTrue(balanced)

    def test_single_smiles_unbalanced(self):
        """Test a single unbalanced reaction in SMILES format."""
        smiles = "CC(=O)O.CCO>>CC(=O)OCC"
        checker = BalanceReactionCheck()
        balanced = checker.rsmi_balance_check(smiles)
        self.assertFalse(balanced)

    def test_list_of_dicts_mixed(self):
        """Test a list of dictionaries with mixed balanced and unbalanced reactions."""
        reactions = [
            {
                "R-id": "test_1",
                "reactions": (
                    "Clc1cnc2nc1Nc1ccc(OCCC3CCNCC3)c(c1)CCc1cncc(c1)N2.O=C=NCc1ccco1"
                    + ">>O=C(NCc1ccco1)N1CCC(CCOc2ccc3cc2CCc2cncc(c2)Nc2ncc(Cl)c(n2)N3)CC1"
                ),
            },
            {"R-id": "test_2", "reactions": "CC(=O)O.CCO>>CC(=O)OCC"},
        ]
        checker = BalanceReactionCheck()
        balanced, unbalanced = checker.dicts_balance_check(
            reactions, rsmi_column="reactions"
        )
        self.assertEqual(len(balanced), 1)
        self.assertEqual(len(unbalanced), 1)
        self.assertEqual(balanced[0]["R-id"], "test_1")
        self.assertEqual(unbalanced[0]["R-id"], "test_2")

    def test_valid_smiles_single_molecule(self):
        """
        Test that the function returns the correct molecular formula for a valid SMILES string.
        """
        smiles = "CCO"  # Ethanol
        expected_formula = "C2H6O"
        result = BalanceReactionCheck().get_combined_molecular_formula(smiles)
        self.assertEqual(
            result, expected_formula, "Molecular formula for ethanol should be C2H6O."
        )

    def test_valid_smiles_complex_molecule(self):
        """
        Test that the function returns the correct molecular formula for a more complex molecule.
        """
        smiles = "CC(C)C(=O)O"  # Isobutyric acid
        expected_formula = "C4H8O2"
        result = BalanceReactionCheck().get_combined_molecular_formula(smiles)
        self.assertEqual(
            result,
            expected_formula,
            "Molecular formula for isobutyric acid should be C4H8O2.",
        )

    def test_invalid_smiles(self):
        """
        Test that the function returns an empty string when an invalid SMILES string is provided.
        """
        invalid_smiles = "INVALID"
        result = BalanceReactionCheck().get_combined_molecular_formula(invalid_smiles)
        self.assertEqual(result, "", "Invalid SMILES should return an empty string.")


if __name__ == "__main__":
    unittest.main()
