from rdkit import Chem
from typing import Optional


class FixAAM:
    """
    A class containing methods for manipulating atom mapping numbers (AAM) in molecules.
    It includes functionality for incrementing atom map numbers in a molecule, adjusting
    atom mappings in SMILES strings, and fixing atom mappings in reaction SMILES (RSMI) strings.

    Methods:
    increment_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        Increments the atom map number for each atom in the molecule by 1.

    fix_aam_smiles(smiles: str) -> str:
        Takes a SMILES string, increments all atom mapping numbers by 1, and returns the updated SMILES.

    fix_aam_rsmi(rsmi: str) -> str:
        Adjusts atom mapping numbers in both reactant and product parts of a reaction SMILES (RSMI).
    """

    @staticmethod
    def increment_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """
        Increments the atom mapping number of each atom in the molecule by 1.

        This method iterates through each atom in the molecule and increments its atom map
        number (if it has one).

        Parameters:
        mol (Chem.Mol): The RDKit molecule object that represents the molecule with atom mapping.

        Returns:
        Chem.Mol: The updated RDKit molecule with incremented atom mapping numbers for all atoms.
        """
        # Iterate through all atoms in the molecule
        for atom in mol.GetAtoms():
            # Get the current atom map (if it exists)
            atom_map = atom.GetAtomMapNum()

            atom.SetAtomMapNum(atom_map + 1)
        return mol

    @staticmethod
    def fix_aam_smiles(smiles: str) -> str:
        """
        Takes a SMILES string, increments all atom mapping numbers by 1, and returns the updated SMILES.

        This method converts the SMILES string into an RDKit molecule, increments the atom
        mapping numbers, and returns the updated SMILES string.

        Parameters:
        smiles (str): A SMILES string containing atom mapping numbers.

        Returns:
        str: A new SMILES string with incremented atom mapping numbers for all atoms.

        Raises:
        ValueError: If the input SMILES string is invalid and cannot be parsed into a molecule.
        """
        # Create the molecule from the SMILES string
        mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        Chem.SanitizeMol(mol)
        # Increment atom mapping numbers
        updated_mol = FixAAM.increment_atom_mapping(mol)

        # Return the SMILES string with updated atom mappings
        return Chem.MolToSmiles(updated_mol)

    @staticmethod
    def fix_aam_rsmi(rsmi: str) -> str:
        """
        Adjusts atom mapping numbers in both reactant and product parts of a reaction SMILES (RSMI).

        This method splits the reaction SMILES (RSMI) into its reactant and product components,
        increments the atom mappings for both parts, and returns the updated reaction SMILES string.

        Parameters:
        rsmi (str): A reaction SMILES string with atom mapping numbers.

        Returns:
        str: A new reaction SMILES string with incremented atom mapping numbers in both reactant
             and product parts.
        """
        # Split the reaction SMILES into reactants and products
        r, p = rsmi.split(">>")

        # Update both reactant and product SMILES strings
        return f"{FixAAM.fix_aam_smiles(r)}>>{FixAAM.fix_aam_smiles(p)}"
