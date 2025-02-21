import unittest
from rdkit import Chem
from synkit.Chem.Reaction.standardize import Standardize


class TestStandardize(unittest.TestCase):
    def setUp(self):
        self.standardizer = Standardize()

    def test_remove_atom_mapping_valid(self):
        reaction_smiles = "[CH3:1][C:2](=[O:3])[O:4]C>>[CH3:1][C:2](=[O:3])[OH:4]"
        expected = "COC(C)=O>>CC(=O)O"
        result = self.standardizer.remove_atom_mapping(reaction_smiles)
        self.assertEqual(result, expected)

    def test_remove_atom_mapping_invalid(self):
        reaction_smiles = "CC[CH2:1]"
        with self.assertRaises(ValueError):
            self.standardizer.remove_atom_mapping(reaction_smiles)

    def test_filter_valid_molecules(self):
        smiles_list = ["CCC", "O=C=O", "N#C"]
        result = self.standardizer.filter_valid_molecules(smiles_list)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(mol, Chem.Mol) for mol in result))

    def test_standardize_rsmi_valid(self):
        rsmi = "CC(=O)OC>>CC(=O)O"
        expected = "COC(C)=O>>CC(=O)O"
        result = self.standardizer.standardize_rsmi(rsmi)
        self.assertEqual(result, expected)

    def test_standardize_rsmi_invalid(self):
        rsmi = "CC.O"
        with self.assertRaises(ValueError):
            self.standardizer.standardize_rsmi(rsmi)

    def test_standardize_rsmi_none_found(self):
        rsmi = "invalid.smiles>>another.invalid"
        result = self.standardizer.standardize_rsmi(rsmi)
        self.assertIsNone(result)

    def test_fit(self):
        rsmi = "[CH3:1][C:2](=[O:3])[O:4]C>>[CH3:1][C:2](=[O:3])[OH:4]"
        expected = "COC(C)=O>>CC(=O)O"
        result = self.standardizer.fit(rsmi)
        self.assertEqual(result, expected)

    def test_fit_remove_aam_false(self):
        rsmi = "[CH3:1][C:2](=[O:3])[O:4]C>>[CH3:1][C:2](=[O:3])[OH:4]"
        expected = "C[O:4][C:2]([CH3:1])=[O:3]>>[CH3:1][C:2](=[O:3])[OH:4]"
        result = self.standardizer.fit(rsmi, remove_aam=False)
        print(result)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
