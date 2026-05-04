from typing import List, Optional, Tuple
from rdkit import Chem


class Standardize:
    """Utilities to normalize and filter reaction and molecule SMILES.

    This class provides methods to remove atom‑mapping, filter invalid molecules,
    canonicalize reaction SMILES, and a full pipeline via `fit`.

    :ivar None: Stateless helper class.
    """

    def __init__(self) -> None:
        """Initialize the Standardize helper.

        No instance attributes are set.
        """
        pass

    @staticmethod
    def remove_atom_mapping(reaction_smiles: str, symbol: str = ">>") -> str:
        """Remove atom‑map numbers from a reaction SMILES string.

        :param reaction_smiles: Reaction SMILES with atom maps, e.g.
            'C[CH3:1]>>C'.
        :type reaction_smiles: str
        :param symbol: Separator between reactants and products.
            Defaults to '>>'.
        :type symbol: str
        :returns: Reaction SMILES without atom‑mapping annotations.
        :rtype: str
        :raises ValueError: If the input format is invalid or contains
            invalid SMILES.
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
        """Filter and sanitize a list of SMILES, returning only valid Mol
        objects.

        :param smiles_list: List of SMILES strings to validate.
        :type smiles_list: List[str]
        :returns: List of sanitized RDKit Mol objects.
        :rtype: List[rdkit.Chem.Mol]
        """
        valid: List[Chem.Mol] = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    valid.append(mol)
                except Exception:
                    continue
        return valid

    @staticmethod
    def _parse_molecule_fragments(
        smiles_list: List[str],
    ) -> Tuple[List[Chem.Mol], bool]:
        """Parse and sanitize SMILES fragments.

        :param smiles_list: List of SMILES strings to validate.
        :type smiles_list: List[str]
        :returns: Tuple of valid molecules and whether any fragment was invalid.
        :rtype: Tuple[List[rdkit.Chem.Mol], bool]
        """
        valid: List[Chem.Mol] = []
        had_invalid = False
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    valid.append(mol)
                except Exception:
                    had_invalid = True
            else:
                had_invalid = True
        return valid, had_invalid

    @staticmethod
    def standardize_rsmi(
        rsmi: str, stereo: bool = False, remove_invalid: bool = True
    ) -> Optional[str]:
        """
        Normalize a reaction SMILES: validate molecules, sort fragments, optionally keep stereo.

        :param rsmi: Reaction SMILES in 'reactants>>products' format.
        :type rsmi: str
        :param stereo: If True, include stereochemistry in the output. Defaults to False.
        :type stereo: bool
        :param remove_invalid: If True, drop invalid fragments and standardize
            remaining molecules. If False, return None when any invalid fragment
            exists. Defaults to True.
        :type remove_invalid: bool
        :returns: Standardized reaction SMILES or None if no valid molecules remain.
        :rtype: Optional[str]
        :raises ValueError: If the input format is invalid.
        """
        try:
            react_str, prod_str = rsmi.split(">>")
        except ValueError:
            raise ValueError(
                "Invalid reaction SMILES format. Expected 'reactants>>products'."
            )

        react_mols, react_invalid = Standardize._parse_molecule_fragments(
            react_str.split(".")
        )
        prod_mols, prod_invalid = Standardize._parse_molecule_fragments(
            prod_str.split(".")
        )

        if not remove_invalid and (react_invalid or prod_invalid):
            return None

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
        self,
        rsmi: str,
        remove_aam: bool = True,
        ignore_stereo: bool = True,
        remove_invalid: bool = True,
    ) -> Optional[str]:
        """
        Full standardization pipeline: strip atom‑mapping, normalize SMILES, fix hydrogen notation.

        :param rsmi: Reaction SMILES to process.
        :type rsmi: str
        :param remove_aam: If True, remove atom‑mapping annotations. Defaults to True.
        :type remove_aam: bool
        :param ignore_stereo: If True, drop stereochemistry. Defaults to True.
        :type ignore_stereo: bool
        :param remove_invalid: If True, drop invalid fragments and standardize
            remaining molecules. If False, return None when any invalid fragment
            exists. Defaults to True.
        :type remove_invalid: bool
        :returns: The standardized reaction SMILES, or None if standardization fails.
        :rtype: Optional[str]
        """
        std = self.standardize_rsmi(
            rsmi, stereo=not ignore_stereo, remove_invalid=remove_invalid
        )
        if std is None:
            return None

        if remove_aam:
            std = self.remove_atom_mapping(std)

        # Format any double‑hydrogen notation
        return std.replace("[HH]", "[H][H]")

    @staticmethod
    def categorize_reactions(
        reactions: List[str], target_reaction: str
    ) -> Tuple[List[str], List[str]]:
        """Partition reactions into those matching a target and those not.

        :param reactions: List of reaction SMILES to categorize.
        :type reactions: List[str]
        :param target_reaction: Benchmark reaction SMILES for comparison.
        :type target_reaction: str
        :returns: Tuple of (matches, non_matches):
                  - matches: reactions equal to standardized target
                  - non_matches: all others
        :rtype: Tuple[List[str], List[str]]
        """
        tgt = Standardize.standardize_rsmi(target_reaction, stereo=False)
        matches: List[str] = []
        non_matches: List[str] = []
        for rxn in reactions:
            if rxn == tgt:
                matches.append(rxn)
            else:
                non_matches.append(rxn)
        return matches, non_matches
