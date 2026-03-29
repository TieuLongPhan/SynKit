from __future__ import annotations

import unittest

from synkit.CRN.Construct.smiles import (
    Chem,
    standardize_smiles_rdkit,
    mol_from_smiles_safe,
    has_atom_maps,
)

from ._case import SEEDS


@unittest.skipIf(Chem is None, "RDKit is required for smiles tests")
class TestSmiles(unittest.TestCase):
    def test_standardize_smiles_rdkit_keep_aam_false(self):
        self.assertEqual(standardize_smiles_rdkit(SEEDS[0], keep_aam=False), "C=O")
        self.assertEqual(standardize_smiles_rdkit(SEEDS[1], keep_aam=False), "O=CCO")

    def test_standardize_smiles_rdkit_strips_atom_maps_when_requested(self):
        mapped = "[CH2:1]=[O:2]"
        self.assertEqual(standardize_smiles_rdkit(mapped, keep_aam=False), "C=O")

    def test_standardize_smiles_rdkit_keeps_or_assigns_atom_maps(self):
        mapped = "[CH2:1]=[O:2]"
        kept = standardize_smiles_rdkit(mapped, keep_aam=True)
        self.assertIsNotNone(kept)
        mol = mol_from_smiles_safe(kept)
        self.assertIsNotNone(mol)
        self.assertTrue(has_atom_maps(mol))
