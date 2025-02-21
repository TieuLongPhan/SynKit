import unittest
from synkit.Reactor.reagent import (
    remove_reagent_from_smiles,
    add_catalysis,
)


class TestReagent(unittest.TestCase):
    def setUp(self) -> None:
        self.rsmi = "CC=O.CC=O.CCC=O>>CC=O.CC=C(C)C=O.O"

    def test_remove_reagent_from_smiles(self):
        rsmi = remove_reagent_from_smiles(self.rsmi)
        self.assertEqual(rsmi, "CC=O.CCC=O>>CC=C(C)C=O.O")

    def test_remove_reagent_no_common(self):
        # Test case with no common molecules
        rsmi = "CC=O>>C=O.CCC"
        result = remove_reagent_from_smiles(rsmi)
        self.assertEqual(result, rsmi)

    def test_remove_reagent_multiple_common(self):
        # Test case with multiple instances of the same common molecule
        rsmi = "C.O.C>>C.O.C.O"
        result = remove_reagent_from_smiles(rsmi)
        self.assertEqual(result, ">>O")

    def test_invalid_smiles(self):
        # Assuming your functions have error handling for invalid SMILES
        with self.assertRaises(ValueError):
            remove_reagent_from_smiles("invalid_smiles_string")

    def test_add_catalysis(self):
        rsmi = "CC=O.CCC=O>>CC=C(C)C=O.O"
        rsmi = add_catalysis(rsmi, "[H+]")
        self.assertEqual(rsmi, "CC=O.CCC=O.[H+]>>CC=C(C)C=O.O.[H+]")

    def test_add_catalysis_empty_catalyst(self):
        # Test adding no catalyst
        rsmi = "CC=O>>C=O"
        result = add_catalysis(rsmi, "")
        self.assertEqual(result, rsmi)

    def test_add_catalysis_none_catalyst(self):
        # Test adding None as catalyst
        rsmi = "CC=O>>C=O"
        result = add_catalysis(rsmi, None)
        self.assertEqual(result, rsmi)

    def test_combined_operations(self):
        rsmi = "C.O>>C.C"
        cleaned_smiles = remove_reagent_from_smiles(rsmi)
        modified_smiles = add_catalysis(cleaned_smiles, "[Pd]")
        self.assertEqual(modified_smiles, "O.[Pd]>>C.[Pd]")


if __name__ == "__main__":
    unittest.main()
