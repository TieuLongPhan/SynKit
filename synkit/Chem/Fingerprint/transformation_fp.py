"""transformation_fp.py
=======================
Compute reaction‐level fingerprints by combining molecular fingerprints
of reactants and products, with optional absolute mode and bit‐vector conversion.

Quick start
-----------
>>> from synkit.Chem.Fingerprint.transformation_fp import TransformationFP
>>> arr = TransformationFP().fit('CCO>>CC=O', symbols='>>', fp_type='ecfp4', abs=True)
>>> bv = TransformationFP().fit('CCO>>CC=O', symbols='>>', fp_type='ecfp4', abs=True, return_array=False)
"""

from __future__ import annotations
from typing import Any, Union

import numpy as np
from rdkit.DataStructs import cDataStructs

from synkit.Chem.Fingerprint.smiles_featurizer import SmilesFeaturizer


class TransformationFP:
    """Calculate reaction fingerprints by featurizing individual molecules and
    combining them via vector subtraction.

    :cvar None: Stateless utility class.
    """

    def __init__(self) -> None:
        """Initialize TransformationFP.

        This class has no instance state; all methods are static or
        class‐level.
        """
        pass

    @staticmethod
    def convert_arr2vec(arr: np.ndarray) -> cDataStructs.ExplicitBitVect:
        """Convert a NumPy array of bits into an RDKit ExplicitBitVect.

        :param arr: Array of 0/1 values representing a fingerprint.
        :type arr: np.ndarray
        :returns: RDKit bit vector constructed from the bit string.
        :rtype: cDataStructs.ExplicitBitVect
        """
        bitstr = "".join(str(int(x)) for x in arr.flatten())
        return cDataStructs.CreateFromBitString(bitstr)

    def fit(
        self,
        reaction_smiles: str,
        symbols: str,
        fp_type: str,
        abs: bool,
        return_array: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, cDataStructs.ExplicitBitVect]:
        """Generate a reaction fingerprint by subtracting reactant from product
        fingerprints.

        :param reaction_smiles: Reaction SMILES, reactant and product separated by `symbols`.
        :type reaction_smiles: str
        :param symbols: Delimiter between reactants and products in the SMILES string.
        :type symbols: str
        :param fp_type: Fingerprint type to use for individual molecules (e.g., 'ecfp4').
        :type fp_type: str
        :param abs: If True, take absolute value of the difference vector.
        :type abs: bool
        :param return_array: If True, return a NumPy array; otherwise convert to an RDKit bit vector.
        :type return_array: bool
        :param kwargs: Additional keyword arguments passed to `SmilesFeaturizer.featurize_smiles`.
        :type kwargs: Any
        :returns: Reaction fingerprint as a NumPy array or RDKit bit vector.
        :rtype: Union[np.ndarray, cDataStructs.ExplicitBitVect]
        :raises ValueError: If `reaction_smiles` is not correctly formatted.
        """
        if symbols not in reaction_smiles:
            raise ValueError(f"Reaction SMILES must contain separator '{symbols}'")
        react_part, prod_part = reaction_smiles.split(symbols)

        def sum_fps(parts: list[str]) -> np.ndarray:
            total = None
            for smi in parts:
                vec = SmilesFeaturizer.featurize_smiles(smi, fp_type, **kwargs)
                if total is None:
                    total = vec.copy() if isinstance(vec, np.ndarray) else vec
                else:
                    total = total + vec  # type: ignore
            return total  # type: ignore

        react_vec = sum_fps(react_part.split("."))
        prod_vec = sum_fps(prod_part.split("."))

        diff = prod_vec - react_vec  # type: ignore
        if abs:
            diff = np.abs(diff)

        if return_array:
            return diff  # type: ignore
        return TransformationFP.convert_arr2vec(diff)  # type: ignore

    def help(self) -> None:
        """Print usage summary for the TransformationFP class.

        :returns: None
        :rtype: NoneType
        """
        print("TransformationFP: compute reaction fingerprints via vector subtraction.")
        print(
            "  fit(reaction_smiles, symbols, fp_type, abs, return_array=True, **kwargs)"
        )
        print("    reaction_smiles: 'R1.R2>>P1.P2' SMILES string")
        print("    symbols: separator between reactants and products (e.g. '>>')")
        print(
            "    fp_type: one of 'maccs', 'avalon', 'ecfp#', 'fcfp#', 'rdk#', 'ap', 'torsion', 'pharm2d'"
        )
        print("    abs: take absolute difference (True/False)")
        print("    return_array: return NumPy array (True) or RDKit bit vector (False)")
        print("  convert_arr2vec(arr: np.ndarray) -> ExplicitBitVect")
        print("Example:")
        print("  tfp = TransformationFP()")
        print("  arr = tfp.fit('CCO>>CC=O', '>>', 'ecfp4', abs=True)")
        print(
            "  bv = tfp.fit('CCO>>CC=O', '>>', 'ecfp4', abs=True, return_array=False)"
        )

    def __str__(self) -> str:
        """Short description of the transformer.

        :returns: Class name.
        :rtype: str
        """
        return "<TransformationFP>"

    __repr__ = __str__
