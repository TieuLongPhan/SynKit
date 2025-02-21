import unittest
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
import numpy as np

from synkit.Chem.Fingerprint.smiles_featurizer import SmilesFeaturizer


class TestSmilesFeaturizer(unittest.TestCase):

    def setUp(self):
        """Set up for tests with a valid smiles string for all tests to use."""
        self.valid_smiles = "CCO"  # Ethanol
        self.mol = SmilesFeaturizer.smiles_to_mol(self.valid_smiles)

    def test_smiles_to_mol_valid(self):
        """Test conversion of valid SMILES to Mol object"""
        smiles = "CCO"  # Ethanol
        mol = SmilesFeaturizer.smiles_to_mol(smiles)
        self.assertIsInstance(mol, Chem.Mol)

    def test_smiles_to_mol_invalid(self):
        """Test handling of invalid SMILES strings"""
        smiles = "CC1"
        with self.assertRaises(ValueError):
            _ = SmilesFeaturizer.smiles_to_mol(smiles)

    def test_get_maccs_keys(self):
        """Test MACCS keys fingerprint generation"""
        fp = SmilesFeaturizer.get_maccs_keys(self.mol)
        self.assertIsInstance(fp, MACCSkeys.GenMACCSKeys(self.mol).__class__)

    def test_get_avalon_fp(self):
        """Test Avalon fingerprint generation with default and custom bit lengths"""
        fp = SmilesFeaturizer.get_avalon_fp(self.mol)
        self.assertEqual(len(fp), 1024)
        fp_custom = SmilesFeaturizer.get_avalon_fp(self.mol, nBits=512)
        self.assertEqual(len(fp_custom), 512)

    def test_get_ecfp(self):
        """Test ECFP fingerprint generation"""
        fp = SmilesFeaturizer.get_ecfp(self.mol, radius=2)
        self.assertEqual(len(fp), 2048)  # Default bit size check

    def test_get_rdk_fp(self):
        """Test RDKit fingerprint generation"""
        fp = SmilesFeaturizer.get_rdk_fp(self.mol, maxPath=5)
        self.assertEqual(len(fp), 2048)  # Check the default size
        fp_custom = SmilesFeaturizer.get_rdk_fp(
            self.mol, maxPath=5, fpSize=1024, nBitsPerHash=1
        )
        self.assertEqual(len(fp_custom), 1024)  # Custom size check

    def test_mol_to_ap(self):
        """Test Atom Pair fingerprint generation"""
        ap_fp = SmilesFeaturizer.mol_to_ap(self.mol)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(ap_fp, ar)
        self.assertEqual(len(ar), 8388608)

    def test_mol_to_pharm2d(self):
        """Test 2D Pharmacophore fingerprint generation"""
        pharm2d_fp = SmilesFeaturizer.mol_to_pharm2d(self.mol)
        ar = np.frombuffer(pharm2d_fp.ToBitString().encode(), "u1") - ord("0")
        self.assertEqual(len(ar), 39972)

    def test_featurize_smiles(self):
        """Test featurization of SMILES strings to numpy arrays and raw bit vectors"""
        smiles = "CCO"
        np_fp = SmilesFeaturizer.featurize_smiles(smiles, "maccs")
        self.assertIsInstance(np_fp, np.ndarray)
        bit_fp = SmilesFeaturizer.featurize_smiles(
            smiles, "maccs", convert_to_array=False
        )
        self.assertNotIsInstance(bit_fp, np.ndarray)  # Should be RDKit ExplicitBitVect

    def test_error_on_unsupported_fingerprint_type(self):
        """Test error handling for unsupported fingerprint types"""
        smiles = "CCO"
        with self.assertRaises(ValueError):
            _ = SmilesFeaturizer.featurize_smiles(smiles, "unsupported_fp_type")


if __name__ == "__main__":
    unittest.main()
