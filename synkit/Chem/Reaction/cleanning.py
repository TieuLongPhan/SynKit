from typing import List
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.balance_check import BalanceReactionCheck


class Cleanning:
    def __init__(self) -> None:
        pass

    @staticmethod
    def remove_duplicates(smiles_list: List[str]) -> List[str]:
        """
        Removes duplicate SMILES strings from a list, maintaining the order of
        first occurrences. Uses a set to track seen SMILES for efficiency.

        Parameters:
        - smiles_list (List[str]): A list of SMILES strings representing
        chemical reactions.

        Returns:
        - List[str]: A list with unique SMILES strings, preserving the original order.
        """
        seen = set()
        unique_smiles = [
            smiles for smiles in smiles_list if not (smiles in seen or seen.add(smiles))
        ]
        return unique_smiles

    @staticmethod
    def clean_smiles(smiles_list: List[str]) -> List[str]:
        """
        Cleans a list of SMILES strings by standardizing them, checking their chemical
        balance, and removing duplicates. Each SMILES is first checked for validity and
        then standardized. Only balanced reactions are kept.

        Parameters:
        - smiles_list (List[str]): A list of SMILES strings representing chemical reactions.

        Returns:
        - List[str]: A list of cleaned and standardized SMILES strings.
        """
        # Standardize and check balance in separate list comprehensions
        standardizer = Standardize()
        balance_checker = BalanceReactionCheck()

        standardized_smiles = []
        for smiles in smiles_list:
            try:
                r = standardizer.standardize_rsmi(smiles, True)
                standardized_smiles.append(r)
            except Exception as e:
                print(e)
                pass
        # standardized_smiles = [
        #     standardizer.standardize_rsmi(smiles, True)
        #     for smiles in smiles_list
        #     if smiles
        # ]
        balanced_smiles = [
            smiles
            for smiles in standardized_smiles
            if balance_checker.rsmi_balance_check(smiles)
        ]

        # Remove duplicates from the balanced SMILES list
        clean_smiles = Cleanning.remove_duplicates(balanced_smiles)
        return clean_smiles
