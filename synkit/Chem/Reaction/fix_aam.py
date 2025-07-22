from rdkit import Chem
from typing import Optional


class FixAAM:
    """Utilities for incrementing and correcting atom‐atom mapping (AAM)
    numbers in molecules and reaction SMILES.

    Provides methods to:
      - Increment AAM on all atoms of an RDKit Mol.
      - Adjust AAM numbers in a standalone SMILES string.
      - Apply the same adjustment to both sides of a reaction SMILES (RSMI).
    """

    @staticmethod
    def increment_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """Increment the atom‐map number of each atom in an RDKit Mol by 1.

        :param mol: RDKit molecule with existing atom‐map annotations.
        :type mol: Chem.Mol
        :returns: The same Mol object with each atom’s map number
            increased by one.
        :rtype: Chem.Mol
        """
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetAtomMapNum() + 1)
        return mol

    @staticmethod
    def fix_aam_smiles(smiles: str) -> str:
        """Parse a SMILES string, increment all atom map numbers, and return
        updated SMILES.

        :param smiles: SMILES string containing atom‐map annotations.
        :type smiles: str
        :returns: SMILES string with every atom‐map number increased by
            one.
        :rtype: str
        :raises ValueError: If the input SMILES cannot be parsed into an
            RDKit Mol.
        """
        mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles!r}")
        Chem.SanitizeMol(mol)
        FixAAM.increment_atom_mapping(mol)
        return Chem.MolToSmiles(mol)

    @staticmethod
    def fix_aam_rsmi(rsmi: str) -> str:
        """Apply atom‐map increment to both reactant and product sides of a
        reaction SMILES.

        :param rsmi: Reaction SMILES in 'reactants>>products' format
            with atom‐map tags.
        :type rsmi: str
        :returns: New reaction SMILES string where each atom‐map number
            in both halves is increased by one.
        :rtype: str
        """
        react, prod = rsmi.split(">>")
        new_react = FixAAM.fix_aam_smiles(react)
        new_prod = FixAAM.fix_aam_smiles(prod)
        return f"{new_react}>>{new_prod}"
