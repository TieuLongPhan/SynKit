import unittest
from rdkit import Chem
from synkit.Chem.utils import (
    enumerate_tautomers,
    mapping_success_rate,
    remove_common_reagents,
    reverse_reaction,
    remove_duplicates,
    merge_reaction,
    find_longest_fragment,
)


class TestChemUtils(unittest.TestCase):
    def test_enumerate_tautomers_simple(self):
        # A simple keto-enol tautomerism: acetylacetone (CC(=O)CC=O) -> same product
        reaction = "CC(=O)CC=O>>O"
        tautomers = enumerate_tautomers(reaction)
        # Should return a list with at least the original reaction
        self.assertIsInstance(tautomers, list)
        self.assertIn(reaction, tautomers)
        # Each entry should be a valid reaction SMILES
        for rsmi in tautomers:
            self.assertIsInstance(rsmi, str)
            parts = rsmi.split(">>")
            self.assertEqual(len(parts), 2)
            # Reactant and product part parseable by RDKit
            self.assertIsNotNone(Chem.MolFromSmiles(parts[0]))
            self.assertIsNotNone(Chem.MolFromSmiles(parts[1]))

    def test_enumerate_tautomers_invalid(self):
        # Invalid SMILES input should raise ValueError
        bad = "INVALID>>SMILES"
        with self.assertRaises(ValueError) as cm:
            enumerate_tautomers(bad)
        self.assertIn("Invalid reactant or product SMILES", str(cm.exception))

    def test_mapping_success_rate_normal(self):
        data = ["C:1CC", "CCC", "O:3=O", ":5", "N"]
        rate = mapping_success_rate(data)
        # Entries with mapping: 'C:1CC', 'O:3=O', ':5' => 3/5 = 60.0%
        self.assertEqual(rate, 60.0)

    def test_mapping_success_rate_empty(self):
        with self.assertRaises(ValueError):
            mapping_success_rate([])

    def test_mapping_success_rate_all(self):
        data = [":1C", ":2", "N:3"]
        rate = mapping_success_rate(data)
        self.assertEqual(rate, 100.0)

    def test_mapping_success_rate_none(self):
        data = ["C", "O", "N"]
        rate = mapping_success_rate(data)
        self.assertEqual(rate, 0.0)

    def test_remove_common_reagents_no_common(self):
        reaction = "A.B.C>>D.E.F"
        expected_result = "A.B.C>>D.E.F"
        result = remove_common_reagents(reaction)
        self.assertEqual(result, expected_result)

    def test_remove_common_reagents_with_common(self):
        reaction = "A.B.C>>A.D.E"
        expected_result = "B.C>>D.E"
        result = remove_common_reagents(reaction)
        self.assertEqual(result, expected_result)

    def test_remove_common_reagents_all_common(self):
        reaction = "A.B.C>>A.B.C"
        expected_result = ">>"
        result = remove_common_reagents(reaction)
        self.assertEqual(result, expected_result)

    def test_remove_duplicates(self):
        input_list = ["CC", "C", "CC", "CCO", "C"]
        expected_result = ["CC", "C", "CCO"]
        result = remove_duplicates(input_list)
        self.assertEqual(result, expected_result)

    def test_reverse_reaction(self):
        reaction_smiles = "C=C.O>>CCO"
        expected_result = "CCO>>C=C.O"
        result = reverse_reaction(reaction_smiles)
        self.assertEqual(result, expected_result)

    def test_merge_reaction(self):
        rsmi_1 = "CCC(=O)OC.O>>CO.CCOC(=O)O"
        rsmi_2 = "CCC(=O)O.CCO>>O.CCOC(=O)CC"
        expected_result = "CCC(=O)OC.O.CCC(=O)O.CCO>>CO.CCOC(=O)O.O.CCOC(=O)CC"
        result = merge_reaction(rsmi_1, rsmi_2)
        self.assertEqual(result, expected_result)

    def test_find_longest_fragment(self):
        input_list = ["CCOC(=O)O", "O"]
        expected_result = "CCOC(=O)O"
        result = find_longest_fragment(input_list)
        self.assertEqual(result, expected_result)

    # Additional robustness for empty inputs or specific edge cases.
    def test_remove_duplicates_empty(self):
        input_list = []
        expected_result = []
        result = remove_duplicates(input_list)
        self.assertEqual(result, expected_result)

    def test_reverse_reaction_empty(self):
        reaction_smiles = ">>"
        expected_result = ">>"
        result = reverse_reaction(reaction_smiles)
        self.assertEqual(result, expected_result)

    def test_merge_reaction_empty(self):
        rsmi_1 = ">>"
        rsmi_2 = ">>"
        result = merge_reaction(rsmi_1, rsmi_2)
        self.assertIsNone(result)

    def test_find_longest_fragment_empty(self):
        input_list = []
        result = find_longest_fragment(input_list)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
