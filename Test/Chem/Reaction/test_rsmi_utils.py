import unittest
from synkit.Chem.Reaction.rsmi_utils import (
    remove_common_reagents,
    reverse_reaction,
    remove_duplicates,
    merge_reaction,
    find_longest_fragment,
)


class TestChemicalReactions(unittest.TestCase):

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
