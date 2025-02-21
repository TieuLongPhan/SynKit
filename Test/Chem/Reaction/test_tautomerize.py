import unittest
from synkit.Chem.Reaction.tautomerize import (
    Tautomerize,
)


class TestTautomerize(unittest.TestCase):

    def test_enol_transformation(self):
        smiles = "C=C(O)CC"  # But-1-en-2-ol
        atom_indices = [0, 1, 2]  # Indices of C=O and adjacent C
        standardizer = Tautomerize()
        result = standardizer.standardize_enol(smiles, atom_indices)
        expected = "CCC(C)=O"  # Expected butanone
        self.assertEqual(result, expected, "Enol transformation failed or incorrect")

    def test_hemiketal_transformation(self):
        # Assuming the structure can form a hemiketal
        smiles = "C(O)(O)"
        atom_indices = [0, 1, 2]  # Indices for hemiketal formation
        standardizer = Tautomerize()
        result = standardizer.standardize_hemiketal(smiles, atom_indices)
        expected = "C=O.O"  # Expected hemiketal cyclic structure
        self.assertEqual(
            result, expected, "Hemiketal transformation failed or incorrect"
        )

    def test_fix_smiles(self):
        smiles = "C(O)(O)C=CO"
        standardizer = Tautomerize()
        result = standardizer.fix_smiles(smiles)
        expected = "O.O=CCC=O"
        self.assertEqual(result, expected, "MoleculeStandardizer failed or incorrect")

    def test_fix_dict(self):
        dict1 = {"smiles": "C(O)(O)C=CO", "id": 1}
        dict2 = {"smiles": "C(O)(O)C=CO>>CCC", "id": 2}
        standardizer = Tautomerize()
        result1 = standardizer.fix_dict(dict1, "smiles")
        expected1 = {"smiles": "O.O=CCC=O", "id": 1}
        self.assertEqual(result1, expected1, "MoleculeStandardizer failed or incorrect")

        result2 = standardizer.fix_dict(dict2, "smiles")
        expected2 = {"smiles": "O.O=CCC=O>>CCC", "id": 2}
        self.assertEqual(result2, expected2, "MoleculeStandardizer failed or incorrect")

    def test_fix_dicts(self):
        dicts = [
            {"smiles": "C(O)(O)C=CO", "id": 1},
            {"smiles": "C(O)(O)C=CO>>CCC", "id": 2},
        ]
        standardizer = Tautomerize()
        result = standardizer.fix_dicts(dicts, "smiles")
        expected = [
            {"smiles": "O.O=CCC=O", "id": 1},
            {"smiles": "O.O=CCC=O>>CCC", "id": 2},
        ]
        self.assertEqual(result, expected, "MoleculeStandardizer failed or incorrect")


# If the script is executed directly, run the tests
if __name__ == "__main__":
    unittest.main()
