import unittest
import numpy as np
from rdkit.DataStructs import cDataStructs

from synkit.Chem.Fingerprint.transformation_fp import TransformationFP


class TestTransformationFP(unittest.TestCase):

    def test_convert_arr2vec(self):
        """Test conversion of numpy array to RDKit ExplicitBitVect"""
        input_array = np.array([1, 0, 1, 1, 0, 1])
        bit_vect = TransformationFP.convert_arr2vec(input_array)
        self.assertIsInstance(bit_vect, cDataStructs.ExplicitBitVect)
        self.assertEqual(bit_vect.GetNumBits(), len(input_array))

    def test_fit(self):
        """Test the generation of reaction fingerprints from reaction SMILES"""
        reaction_smiles = "CCO.CCN>>CCOC(C)N"
        symbols = ">>"
        fp_type = "maccs"
        abs_val = True

        # Test with return_array=True
        reaction_fp_array = TransformationFP.fit(
            reaction_smiles, symbols, fp_type, abs_val
        )
        self.assertIsInstance(reaction_fp_array, np.ndarray)

        # Test with return_array=False
        reaction_fp_bitvect = TransformationFP.fit(
            reaction_smiles, symbols, fp_type, abs_val, return_array=False
        )
        self.assertIsInstance(reaction_fp_bitvect, cDataStructs.ExplicitBitVect)

    def test_fit_invalid_smiles(self):
        """Test fit method with invalid SMILES that should raise an error in underlying methods"""
        reaction_smiles = "invalid_smiles>>invalid_smiles"
        symbols = ">>"
        fp_type = "maccs"
        abs_val = True
        with self.assertRaises(Exception):
            _ = TransformationFP.fit(reaction_smiles, symbols, fp_type, abs_val)

    def test_fit_reaction_split(self):
        """Test handling of SMILES split by symbols and impact on results"""
        reaction_smiles = "CCO>>CCN"  # Simple reaction split case
        symbols = ">>"
        fp_type = "maccs"
        abs_val = False  # without taking absolute values
        reaction_fp = TransformationFP.fit(reaction_smiles, symbols, fp_type, abs_val)
        self.assertIsInstance(reaction_fp, np.ndarray)


if __name__ == "__main__":
    unittest.main()
