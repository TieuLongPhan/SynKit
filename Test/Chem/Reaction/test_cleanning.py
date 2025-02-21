import unittest
from synkit.Chem.Reaction.cleanning import Cleanning


class TestCleaning(unittest.TestCase):

    def setUp(self):
        self.cleaner = Cleanning()

    def test_remove_duplicates(self):
        input_smiles = ["CC>>CC", "CC>>CC"]
        expected_output = ["CC>>CC"]
        result = self.cleaner.remove_duplicates(input_smiles)
        self.assertEqual(
            result, expected_output, "Failed to remove duplicates correctly"
        )

    def test_clean_smiles(self):
        input_smiles = ["CC>>CC", "CC>>CC", "CC>>CCC"]
        expected_output = ["CC>>CC"]  # Assuming 'CC>>CCC' is not balanced
        result = self.cleaner.clean_smiles(input_smiles)
        self.assertEqual(result, expected_output, "Failed to clean SMILES correctly")


if __name__ == "__main__":
    unittest.main()
