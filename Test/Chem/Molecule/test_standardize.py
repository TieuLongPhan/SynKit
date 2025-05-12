import unittest
from rdkit import Chem
from synkit.Chem.Molecule.standardize import (
    normalize_molecule,
    canonicalize_tautomer,
    salts_remover,
    uncharge_molecule,
    fragments_remover,
    remove_explicit_hydrogens,
    remove_radicals_and_add_hydrogens,
    remove_isotopes,
    clear_stereochemistry,
)


class TestMoleculeFunctions(unittest.TestCase):

    def test_normalize_molecule(self):
        smi = "[Na]OC(=O)c1ccc(C[S+2]([O-])([O-]))cc1"
        expect = "O=C([O][Na])c1ccc(C[S](=O)=O)cc1"
        mol = Chem.MolFromSmiles(smi)
        normalized_mol = normalize_molecule(mol)
        self.assertIsInstance(normalized_mol, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(normalized_mol))

    def test_canonicalize_tautomer(self):
        smi = "N=c1[nH]cc[nH]1"
        expect = "Nc1ncc[nH]1"
        mol = Chem.MolFromSmiles(smi)
        tautomer = canonicalize_tautomer(mol)
        self.assertIsInstance(tautomer, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(tautomer))

    def test_salts_remover(self):
        smi = "CC(=O).[Na+]"
        expect = "CC=O"
        mol = Chem.MolFromSmiles(smi)
        remover = salts_remover(mol)
        self.assertIsInstance(remover, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(remover))

    def test_uncharge_molecule(self):
        smi = "CC(=O)[O-]"
        expect = "CC(=O)O"
        mol = Chem.MolFromSmiles(smi)
        uncharged_mol = uncharge_molecule(mol)
        self.assertIsInstance(uncharged_mol, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(uncharged_mol))

    def test_fragments_remover(self):
        smi = "CC(=O)[O-].[Na+]"
        expect = "CC(=O)[O-]"
        mol = Chem.MolFromSmiles(smi)
        remover = fragments_remover(mol)
        self.assertIsInstance(remover, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(remover))

    def test_remove_explicit_hydrogens(self):
        smi = "[CH4]"
        expect = "C"
        mol = Chem.MolFromSmiles(smi)
        remover = remove_explicit_hydrogens(mol)
        self.assertIsInstance(remover, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(remover))

    def test_remove_radicals(self):
        smi = "[CH3]"
        expect = "C"
        mol = Chem.MolFromSmiles(smi)
        remover = remove_radicals_and_add_hydrogens(mol)
        self.assertIsInstance(remover, Chem.Mol)
        self.assertEqual(expect, Chem.MolToSmiles(remover))

    def test_remove_isotopes(self):
        # Molecule with isotopic labeling
        smiles = "[13CH3]C([2H])([2H])[17O][18OH]"
        expect = "[H]C([H])(C)OO"
        mol = Chem.MolFromSmiles(smiles)
        result_mol = remove_isotopes(mol)
        for atom in result_mol.GetAtoms():
            self.assertEqual(atom.GetIsotope(), 0, "Isotopes not properly removed")
        self.assertEqual(Chem.MolToSmiles(result_mol), expect)

    def test_clear_stereochemistry(self):
        # Molecule with defined stereochemistry
        smiles = "C[C@H](O)[C@@H](O)C"
        mol = Chem.MolFromSmiles(smiles)
        result_mol = clear_stereochemistry(mol)
        has_stereo = any(atom.HasProp("_CIPCode") for atom in result_mol.GetAtoms())
        self.assertFalse(has_stereo, "Stereochemistry not properly cleared")
