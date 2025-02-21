import numpy as np
from typing import Union, Any
from rdkit.DataStructs import cDataStructs
from synkit.Chem.Fingerprint.smiles_featurizer import SmilesFeaturizer


class TransformationFP:
    """
    A class for handling the transformation of chemical reactions into reaction fingerprints
    based on SMILES strings.
    """

    def __init__(self) -> None:
        """
        Initializes the TransformationFP object. Currently, this constructor does not
        perform any operations.
        """
        pass

    @staticmethod
    def convert_arr2vec(arr: np.ndarray) -> cDataStructs.ExplicitBitVect:
        """
        Converts a numpy array to a RDKit ExplicitBitVect.

        Parameters:
        - arr (np.ndarray): The input array.

        Returns:
        - cDataStructs.ExplicitBitVect: The converted bit vector.
        """
        arr_tostring = "".join(arr.astype(str))
        EBitVect = cDataStructs.CreateFromBitString(arr_tostring)
        return EBitVect

    @staticmethod
    def fit(
        reaction_smiles: str,
        symbols: str,
        fp_type: str,
        abs: bool,
        return_array: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, cDataStructs.ExplicitBitVect]:
        """
        Generates a reaction fingerprint for a given reaction represented by a SMILES string.

        Parameters:
        - reaction_smiles (str): The SMILES string of the reaction, separated by `symbols`.
        - symbols (str): The symbol used to separate reactants and products in the SMILES string.
        - fp_type (str): The type of fingerprint to generate (e.g., 'maccs', 'ecfp').
        - abs (bool): Whether to take the absolute value of the reaction fingerprint difference.
        - return_array (bool): Whether to return the reaction fingerprint as a numpy array or as a bit vector.

        Returns:
        - Union[np.ndarray, cDataStructs.ExplicitBitVect]: The reaction fingerprint either as an array
          or a bit vector, depending on the value of `return_array`.
        """
        react, prod = reaction_smiles.split(symbols)
        react_fps = None
        for s in react.split("."):
            if react_fps is None:
                react_fps = SmilesFeaturizer.featurize_smiles(s, fp_type, **kwargs)
            else:
                react_fps += SmilesFeaturizer.featurize_smiles(s, fp_type, **kwargs)

        prod_fps = None
        for s in prod.split("."):
            if prod_fps is None:
                prod_fps = SmilesFeaturizer.featurize_smiles(s, fp_type, **kwargs)
            else:
                prod_fps += SmilesFeaturizer.featurize_smiles(s, fp_type, **kwargs)

        reaction_fp = np.subtract(prod_fps, react_fps)
        if abs:
            reaction_fp = np.abs(reaction_fp)
        if return_array:
            return reaction_fp
        else:
            return TransformationFP.convert_arr2vec(reaction_fp)
