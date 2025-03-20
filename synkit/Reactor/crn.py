from copy import deepcopy
from typing import List, Dict, Any


from synkit.Chem.Reaction.cleanning import Cleanning
from synkit.Chem.utils import count_carbons, get_max_fragment, process_smiles_list
from synkit.Reactor.reactor_utils import _remove_reagent
from synkit.Reactor.core_engine import CoreEngine


class CRN:
    def __init__(
        self,
        rule_list: List[Dict[str, Any]],
        smiles_list: List[str],
        n_repeats: int = 3,
    ) -> None:
        """
        Initializes the CRN class with a list of transformation rules, a list of SMILES strings, and the number of
        expansion repeats to perform on the initial set of molecules.

        Parameters:
        - rule_list (List[Dict[str, Any]]): A list of dictionaries containing rules for molecular transformations.
        - smiles_list (List[str]): A list of SMILES strings representing the initial molecules.
        - n_repeats (int, optional): The number of times to repeat the expansion process. Default is 3.
        """
        self.rule_list = rule_list
        self.smiles_list = smiles_list
        self.n_repeats = n_repeats

    @staticmethod
    def update_smiles(
        list_smiles: List[str],
        solution: List[str],
        prune: bool = True,
        starting_compound: str = None,
        target_compound: str = None,
    ) -> List[str]:
        """
        Updates the list of SMILES strings by extracting products from transformation rules and ensuring uniqueness,
        possibly pruning to the largest fragment and considering carbon count relative to a starting compound.

        Parameters:
        - list_smiles (List[str]): Current list of SMILES strings.
        - solution (List[str]): List of reaction strings from which new SMILES will be extracted.
        - prune (bool, optional): Whether to prune to the largest fragment. Default is True.
        - starting_compound (str, optional): SMILES string of the starting compound for comparison. Default is None.

        Returns:
        - List[str]: An updated list of unique SMILES strings.
        """
        new_list = []
        for r in solution:
            smiles = r.split(">>")[1].split(".")
            if prune:
                smiles = get_max_fragment(smiles)
                if count_carbons(smiles) <= count_carbons(target_compound):
                    if count_carbons(smiles) >= count_carbons(starting_compound):
                        new_list.append(smiles)
            else:
                new_list.extend(smiles)
        new_list = list(set(new_list))
        new_list.extend(list_smiles)
        new_list = list(set(new_list))
        return new_list

    def _expand(
        self, rule_list: List[Dict[str, Any]], smiles_list: List[str]
    ) -> List[str]:
        """
        Private method to expand a list of SMILES strings using provided transformation rules.

        Parameters:
        - rule_list (List[Dict[str, Any]]): List of transformation rules.
        - smiles_list (List[str]): List of SMILES strings to be transformed.

        Returns:
        - List[str]: List of resulting transformation strings after applying the rules.
        """
        solution = []
        process_smiles = process_smiles_list(smiles_list)
        for r in rule_list:
            r = CoreEngine()._inference(r["gml"], process_smiles)
            r = Cleanning().clean_smiles(r)
            r = [_remove_reagent(i) for i in r]
            solution.extend(r)
        return solution

    def _build_crn(
        self, starting_compound: str, target_compound: str
    ) -> List[Dict[str, List[str]]]:
        """
        Private method to build a chemical reaction network by repeatedly
        expanding the list of SMILES strings based on transformation rules.

        Parameters:
        - starting_compound (str): SMILES string of the compound to start the
        reaction network from.

        Returns:
        - List[Dict[str, List[str]]]: A list of dictionaries, each representing the set
        of reactions for each round.
        """
        solutions = []
        solution = []
        smiles = deepcopy(self.smiles_list)
        for i in range(1, self.n_repeats + 1):
            if i > 1:
                smiles = self.update_smiles(
                    smiles,
                    solution,
                    starting_compound=starting_compound,
                    target_compound=target_compound,
                )
            solution = self._expand(self.rule_list, smiles)
            solutions.append({f"Round {i}": solution})

        return smiles, solutions
