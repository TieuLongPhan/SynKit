import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate


class SmilesFeaturizer:
    def __init__(self):
        """
        Initializes the SmilesFeaturizer class without any specific parameters for fingerprint generation.
        """
        pass

    @staticmethod
    def smiles_to_mol(smiles: str) -> Chem.Mol:
        """
        Converts a SMILES string to an RDKit Mol object.

        Parameters:
        - smiles (str): The SMILES string to be converted.

        Returns:
        - Chem.Mol: The corresponding RDKit Mol object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string provided.")
        return mol

    @staticmethod
    def get_maccs_keys(mol: Chem.Mol):
        """
        Generates MACCS keys fingerprint from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.

        Returns:
        - RDKit ExplicitBitVect: The MACCS keys fingerprint of the Mol object.
        """
        return MACCSkeys.GenMACCSKeys(mol)

    @staticmethod
    def get_avalon_fp(mol: Chem.Mol, nBits: int = 1024):
        """
        Generates Avalon fingerprint from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.
        - nBits (int): The number of bits in the generated fingerprint.

        Returns:
        - RDKit ExplicitBitVect: The Avalon fingerprint of the Mol object.
        """
        return fpAvalon.GetAvalonFP(mol, nBits)

    @staticmethod
    def get_ecfp(
        mol: Chem.Mol, radius: int, nBits: int = 2048, useFeatures: bool = False
    ):
        """
        Generates Extended-Connectivity Fingerprints (ECFP) or
        Feature-Class Fingerprints (FCFP) from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.
        - radius (int): The radius of the fingerprint.
        - nBits (int): The number of bits in the generated fingerprint.
        - useFeatures (bool): Whether to use atom features instead of atom identities.

        Returns:
        - RDKit ExplicitBitVect: The ECFP or FCFP fingerprint of the Mol object.
        """
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits, useFeatures=useFeatures
        )

    @staticmethod
    def get_rdk_fp(
        mol: Chem.Mol, maxPath: int, fpSize: int = 2048, nBitsPerHash: int = 2
    ):
        """
        Generates RDKit fingerprint from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.
        - maxPath (int): The maximum path length (in bonds) to be included.
        - fpSize (int): The size of the fingerprint.
        - nBitsPerHash (int): The number of bits per hash.

        Returns:
        - RDKit ExplicitBitVect: The RDKit fingerprint of the Mol object.
        """
        return Chem.RDKFingerprint(
            mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash
        )

    @staticmethod
    def mol_to_ap(mol: Chem.Mol) -> np.ndarray:
        """
        Generates an Atom Pair fingerprint as a NumPy array from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.

        Returns:
        - RDKit ExplicitBitVect: The RDKit fingerprint of the Mol object.
        """
        return Pairs.GetAtomPairFingerprint(mol)

    @staticmethod
    def mol_to_torsion(mol: Chem.Mol) -> np.ndarray:
        """
        Generates a Topological Torsion fingerprint as a NumPy array from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.

        Returns:
        - RDKit ExplicitBitVect: The RDKit fingerprint of the Mol object.
        """
        return Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)

    @staticmethod
    def mol_to_pharm2d(mol: Chem.Mol) -> np.ndarray:
        """
        Generates a 2D Pharmacophore fingerprint as a NumPy array from an RDKit Mol object.

        Parameters:
        - mol (Chem.Mol): The Mol object to be featurized.

        Returns:
        - RDKit ExplicitBitVect: The RDKit fingerprint of the Mol object.
        """
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    @classmethod
    def featurize_smiles(
        cls, smiles: str, fingerprint_type: str, convert_to_array: bool = True, **kwargs
    ) -> np.ndarray:
        """
        Featurizes a SMILES string into the specified type of fingerprint, optionally converting it to a NumPy array.

        Parameters:
        - smiles (str): The SMILES string to be featurized.
        - fingerprint_type (str): The type of fingerprint to generate.
        - convert_to_array (bool): Whether to convert the fingerprint to a NumPy array. Defaults to True.
        - **kwargs: Additional keyword arguments for the fingerprint function.

        Returns:
        - np.ndarray or RDKit ExplicitBitVect: The requested type of fingerprint for the SMILES string,
          either as a NumPy array or as an RDKit bit vector, depending on `convert_to_array`.
        """
        mol = cls.smiles_to_mol(smiles)
        if fingerprint_type == "maccs":
            fp = cls.get_maccs_keys(mol)
        elif fingerprint_type == "avalon":
            fp = cls.get_avalon_fp(mol, **kwargs)
        elif fingerprint_type.startswith("ecfp") or fingerprint_type.startswith("fcfp"):
            radius = int(fingerprint_type[4])
            useFeatures = fingerprint_type.startswith("fcfp")
            nBits = kwargs.get("nBits", 2048)
            fp = cls.get_ecfp(mol, radius, nBits=nBits, useFeatures=useFeatures)
        elif fingerprint_type.startswith("rdk"):
            maxPath = int(fingerprint_type[3])
            fp = cls.get_rdk_fp(mol, maxPath, **kwargs)
        elif fingerprint_type == "avalon":
            return cls.mol_to_ap(mol)
        elif fingerprint_type == "torsion":
            return cls.mol_to_torsion(mol)
        elif fingerprint_type == "pharm2d":
            return cls.mol_to_pharm2d(mol)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
        if convert_to_array:
            if fingerprint_type == "pharm2d":
                return np.frombuffer(fp.ToBitString().encode(), "u1") - ord("0")
            else:
                ar = np.zeros((1,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, ar)
                return ar
        else:
            return fp
