from typing import List
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.balance_check import BalanceReactionCheck


class Cleaning:
    """Utilities for cleaning and filtering reaction SMILES lists.

    Methods
    -------
    remove_duplicates(smiles_list)
        Remove duplicate SMILES while preserving input order.
    clean_smiles(smiles_list)
        Standardize, balance‑check, and deduplicate a list of reaction SMILES.
    """

    def __init__(self) -> None:
        """Initialize the Cleaning helper.

        No instance attributes are used.
        """
        pass

    @staticmethod
    def remove_duplicates(smiles_list: List[str]) -> List[str]:
        """Remove duplicate SMILES strings, preserving first occurrences.

        :param smiles_list: List of reaction SMILES strings.
        :type smiles_list: List[str]
        :returns: List of unique SMILES in original order.
        :rtype: List[str]
        """
        seen = set()
        return [smi for smi in smiles_list if not (smi in seen or seen.add(smi))]

    @staticmethod
    def clean_smiles(smiles_list: List[str]) -> List[str]:
        """Standardize, balance‑check, and deduplicate reaction SMILES.

        Steps:
          1. Standardize each SMILES via `Standardize.standardize_rsmi`.
          2. Keep only those that pass `BalanceReactionCheck.rsmi_balance_check`.
          3. Remove duplicates preserving order.

        :param smiles_list: List of reaction SMILES strings to clean.
        :type smiles_list: List[str]
        :returns: Cleaned list of standardized, balanced, unique SMILES.
        :rtype: List[str]
        """
        standardizer = Standardize()
        balance_checker = BalanceReactionCheck()

        standardized: List[str] = []
        for smi in smiles_list:
            try:
                std = standardizer.standardize_rsmi(smi, stereo=True)
                if std:
                    standardized.append(std)
            except Exception:
                continue

        balanced = [
            smi for smi in standardized if balance_checker.rsmi_balance_check(smi)
        ]

        return Cleaning.remove_duplicates(balanced)
