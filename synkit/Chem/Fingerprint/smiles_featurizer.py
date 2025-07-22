"""smiles_featurizer.py
=======================
Utility for converting SMILES strings into various cheminformatics fingerprints,
with optional NumPy‐array conversion.

Key features
------------
* **Multi‐fingerprint support** – MACCS, Avalon, ECFP/FCFP, RDKit, AtomPair, Torsion, Pharm2D
* **SMILES validation** – raises on invalid input
* **Array conversion** – output as NumPy arrays for ML pipelines
* **Extensible** – add new methods or override via subclassing

Quick start
-----------
>>> from synkit.Chem.Fingerprint.smiles_featurizer import SmilesFeaturizer
>>> arr = SmilesFeaturizer.featurize_smiles("CCO", "ecfp4", convert_to_array=True)
"""

from __future__ import annotations
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate


class SmilesFeaturizer:
    """Convert SMILES strings into chemical fingerprint vectors.

    :cvar None: This class only provides static/​class methods and holds no state.

    Supported fingerprint methods:
      - MACCS keys
      - Avalon
      - ECFP/FCFP (Morgan)
      - RDKit topological
      - AtomPair
      - Torsion
      - 2D Pharmacophore

    Use `featurize_smiles` for one‑line access.
    """

    def __init__(self) -> None:
        """Initialize SmilesFeaturizer.

        This class has no instance state; all methods are static or
        class‑level.
        """
        pass

    @staticmethod
    def smiles_to_mol(smiles: str) -> Chem.Mol:
        """Convert a SMILES string to an RDKit Mol object.

        :param smiles: The SMILES string to convert.
        :type smiles: str
        :returns: RDKit Mol object corresponding to the SMILES.
        :rtype: Chem.Mol
        :raises ValueError: If the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles!r}")
        return mol

    @staticmethod
    def get_maccs_keys(mol: Chem.Mol) -> Any:
        """Generate the MACCS keys fingerprint for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :returns: MACCS keys fingerprint bit vector.
        :rtype: ExplicitBitVect
        """
        return MACCSkeys.GenMACCSKeys(mol)

    @staticmethod
    def get_avalon_fp(mol: Chem.Mol, nBits: int = 1024) -> Any:
        """Generate the Avalon fingerprint for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :param nBits: Length of the fingerprint vector.
        :type nBits: int
        :returns: Avalon fingerprint bit vector.
        :rtype: ExplicitBitVect
        """
        return fpAvalon.GetAvalonFP(mol, nBits)

    @staticmethod
    def get_ecfp(
        mol: Chem.Mol, radius: int, nBits: int = 2048, useFeatures: bool = False
    ) -> Any:
        """Generate a Morgan fingerprint (ECFP or FCFP) for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :param radius: Radius for the Morgan algorithm.
        :type radius: int
        :param nBits: Length of the fingerprint vector.
        :type nBits: int
        :param useFeatures: If True, generate a Feature‑Class
            fingerprint (FCFP).
        :type useFeatures: bool
        :returns: Morgan fingerprint bit vector.
        :rtype: ExplicitBitVect
        """
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits, useFeatures=useFeatures
        )

    @staticmethod
    def get_rdk_fp(
        mol: Chem.Mol, maxPath: int, fpSize: int = 2048, nBitsPerHash: int = 2
    ) -> Any:
        """Generate an RDKit topological fingerprint for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :param maxPath: Maximum path length (bonds) to include.
        :type maxPath: int
        :param fpSize: Length of the fingerprint vector.
        :type fpSize: int
        :param nBitsPerHash: Bits per hash for path hashing.
        :type nBitsPerHash: int
        :returns: RDKit topological fingerprint bit vector.
        :rtype: ExplicitBitVect
        """
        return Chem.RDKFingerprint(
            mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash
        )

    @staticmethod
    def mol_to_ap(mol: Chem.Mol) -> Any:
        """Generate an Atom Pair fingerprint for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :returns: Atom Pair fingerprint as an integer vector.
        :rtype: ExplicitBitVect
        """
        return Pairs.GetAtomPairFingerprint(mol)

    @staticmethod
    def mol_to_torsion(mol: Chem.Mol) -> Any:
        """Generate a Topological Torsion fingerprint for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :returns: Torsion fingerprint as an integer vector.
        :rtype: ExplicitBitVect
        """
        return Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)

    @staticmethod
    def mol_to_pharm2d(mol: Chem.Mol) -> Any:
        """Generate a 2D Pharmacophore fingerprint for a molecule.

        :param mol: RDKit Mol object.
        :type mol: Chem.Mol
        :returns: 2D pharmacophore fingerprint bit vector.
        :rtype: ExplicitBitVect
        """
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    @classmethod
    def featurize_smiles(
        cls,
        smiles: str,
        fingerprint_type: str,
        convert_to_array: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Featurize a SMILES string into a chosen fingerprint, optionally
        converting to a NumPy array.

        :param smiles: The SMILES string to featurize.
        :type smiles: str
        :param fingerprint_type: One of 'maccs', 'avalon', 'ecfp#', 'fcfp#',
                                 'rdk#', 'ap', 'torsion', 'pharm2d'.
        :type fingerprint_type: str
        :param convert_to_array: If True, convert the result to a NumPy array.
        :type convert_to_array: bool
        :param kwargs: Additional parameters passed to the chosen method:
                       - `nBits` for Avalon/ECFP/FCFP
                       - `radius` for ECFP/FCFP
                       - `maxPath`, `fpSize`, `nBitsPerHash` for RDKit FP
        :type kwargs: dict
        :returns: Fingerprint as a NumPy array (if `convert_to_array`) or RDKit bit vector.
        :rtype: np.ndarray or ExplicitBitVect
        :raises ValueError: If `fingerprint_type` is unsupported.
        """
        mol = cls.smiles_to_mol(smiles)

        ft = fingerprint_type.lower()
        if ft == "maccs":
            fp = cls.get_maccs_keys(mol)
        elif ft == "avalon":
            fp = cls.get_avalon_fp(mol, nBits=kwargs.get("nBits", 1024))
        elif ft.startswith("ecfp") or ft.startswith("fcfp"):
            radius = int(ft[4])
            use_features = ft.startswith("fcfp")
            fp = cls.get_ecfp(
                mol,
                radius,
                nBits=kwargs.get("nBits", 2048),
                useFeatures=use_features,
            )
        elif ft.startswith("rdk"):
            max_path = int(ft[3])
            fp = cls.get_rdk_fp(
                mol,
                maxPath=max_path,
                fpSize=kwargs.get("fpSize", 2048),
                nBitsPerHash=kwargs.get("nBitsPerHash", 2),
            )
        elif ft == "ap":
            fp = cls.mol_to_ap(mol)
        elif ft == "torsion":
            fp = cls.mol_to_torsion(mol)
        elif ft == "pharm2d":
            fp = cls.mol_to_pharm2d(mol)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type!r}")

        if convert_to_array:
            if ft == "pharm2d":
                bitstr = fp.ToBitString()
                return np.array([int(b) for b in bitstr], dtype=np.int8)
            arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr

        return fp

    def __str__(self) -> str:
        """Short description of the featurizer.

        :returns: Class name.
        :rtype: str
        """
        return "<SmilesFeaturizer>"

    def help(self) -> None:
        """Print supported fingerprint types and usage summary.

        :returns: None
        :rtype: NoneType
        """
        print("SmilesFeaturizer supports the following fingerprint types:")
        print("  - maccs, avalon, ecfp#, fcfp#, rdk#, ap, torsion, pharm2d")
        print(
            "Usage: SmilesFeaturizer.featurize_smiles(smiles, fingerprint_type, **kwargs)"
        )
