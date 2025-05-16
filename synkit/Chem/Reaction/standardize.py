from rdkit import Chem
from typing import List, Optional, Tuple


class Standardize:
    """
    A collection of utilities to normalize and filter reaction and molecule SMILES.
    """

    def __init__(self) -> None:
        """
        Initialize the Standardize helper.
        """
        pass

    @staticmethod
    def remove_atom_mapping(reaction_smiles: str, symbol: str = ">>") -> str:
        """
        Remove atom-map numbers from both sides of a reaction SMILES.

        :param reaction_smiles: A reaction SMILES string with atom mappings.
        :type reaction_smiles: str
        :param symbol: The separator between reactants and products. Defaults to ">>".
        :type symbol: str
        :returns: The reaction SMILES with all atom-map annotations stripped.
        :rtype: str
        :raises ValueError: If the input is not in "reactants>>products" format
                            or contains invalid SMILES.
        """
        parts = reaction_smiles.split(symbol)
        if len(parts) != 2:
            raise ValueError(
                "Invalid reaction SMILES format. Expected 'reactants>>products'."
            )

        def clean_smiles(smi: str) -> str:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smi}")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(mol, canonical=True)

        react, prod = map(clean_smiles, parts)
        return f"{react}{symbol}{prod}"

    @staticmethod
    def filter_valid_molecules(smiles_list: List[str]) -> List[Chem.Mol]:
        """
        Convert a list of SMILES to RDKit Mol objects, keeping only valid molecules.

        :param smiles_list: A list of SMILES strings.
        :type smiles_list: list of str
        :returns: A list of sanitized RDKit Mol objects.
        :rtype: list of rdkit.Chem.Mol
        """
        valid = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    valid.append(mol)
                except Exception:
                    pass
        return valid

    @staticmethod
    def standardize_rsmi(rsmi: str, stereo: bool = False) -> Optional[str]:
        """
        Normalize a reaction SMILES by validating, sorting, and optional stereochemistry.

        :param rsmi: The reaction SMILES to standardize.
        :type rsmi: str
        :param stereo: If True, include stereochemical information. Defaults to False.
        :type stereo: bool
        :returns: The standardized reaction SMILES or None if no valid molecules remain.
        :rtype: str or None
        :raises ValueError: If the input is not in "reactants>>products" format.
        """
        try:
            react_str, prod_str = rsmi.split(">>")
        except ValueError:
            raise ValueError(
                "Invalid reaction SMILES format. Expected 'reactants>>products'."
            )

        react_mols = Standardize.filter_valid_molecules(react_str.split("."))
        prod_mols = Standardize.filter_valid_molecules(prod_str.split("."))

        if not react_mols or not prod_mols:
            return None

        sorted_react = ".".join(
            sorted(Chem.MolToSmiles(m, isomericSmiles=stereo) for m in react_mols)
        )
        sorted_prod = ".".join(
            sorted(Chem.MolToSmiles(m, isomericSmiles=stereo) for m in prod_mols)
        )

        return f"{sorted_react}>>{sorted_prod}"

    def fit(
        self, rsmi: str, remove_aam: bool = True, ignore_stereo: bool = True
    ) -> Optional[str]:
        """
        Full standardization pipeline: remove atom-maps, normalize SMILES, fix H notation.

        :param rsmi: The reaction SMILES to process.
        :type rsmi: str
        :param remove_aam: If True, strip atom-mapping numbers. Defaults to True.
        :type remove_aam: bool
        :param ignore_stereo: If True, drop stereochemistry. Defaults to True.
        :type ignore_stereo: bool
        :returns: The processed reaction SMILES or None if standardization fails.
        :rtype: str or None
        """
        if remove_aam:
            rsmi = self.remove_atom_mapping(rsmi)

        std = self.standardize_rsmi(rsmi, stereo=not ignore_stereo)
        if std is None:
            return None

        # Explicitly format double hydrogens
        return std.replace("[HH]", "[H][H]")

    @staticmethod
    def categorize_reactions(
        reactions: List[str], target_reaction: str
    ) -> Tuple[List[str], List[str]]:
        """
        Partition a list of reaction SMILES into those matching a target and those not.

        :param reactions: List of reaction SMILES strings to categorize.
        :type reactions: list of str
        :param target_reaction: The benchmark reaction SMILES for matching.
        :type target_reaction: str
        :returns: A pair `(matches, non_matches)`:
                  - `matches`: reactions equal to the standardized target.
                  - `non_matches`: all others.
        :rtype: tuple (list of str, list of str)
        """
        tgt = Standardize.standardize_rsmi(target_reaction, stereo=False)
        matches, non_matches = [], []
        for rxn in reactions:
            if rxn == tgt:
                matches.append(rxn)
            else:
                non_matches.append(rxn)
        return matches, non_matches
