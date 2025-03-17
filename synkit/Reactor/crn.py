from typing import List, Dict, Any, Union
from rdkit import Chem
from copy import deepcopy


from synkit.Chem.Reaction.cleanning import Cleanning
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
    def count_carbons(smiles: str) -> int:
        """ "
        Counts the number of carbon atoms in a molecule given a SMILES string.

        Parameters:
        - smiles (str): SMILES representation of the molecule.

        Returns:
        - int: Number of carbon atoms in the molecule if the SMILES string is valid.
        - str: Error message indicating an invalid SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol:
            carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
            return carbon_count
        else:
            return "Invalid SMILES string"

    @staticmethod
    def get_max_fragment(smiles: Union[str, List[str]]) -> str:
        """
        Extracts and returns the SMILES string of the largest fragment from a SMILES string or a list of SMILES strings
        of a compound that may contain multiple fragments. This function determines the largest fragment based on the
        number of atoms.

        Parameters:
        - smiles (Union[str, List[str]]): A single SMILES string or a list of SMILES strings containing potentially
        multiple fragments.

        Returns:
        - str: SMILES string of the largest fragment based on the number of atoms. Returns an empty string if no valid
        fragments can be processed.

        Examples:
        - get_max_fragment("C.CC.CCC") returns "CCC"
        - get_max_fragment(["C.CC", "CCC.C"]) returns "CCC"
        """
        if isinstance(smiles, str):
            fragments = smiles.split(".")
        elif isinstance(smiles, list):
            fragments = [frag for s in smiles for frag in s.split(".")]
        else:
            return ""

        molecules = [Chem.MolFromSmiles(fragment) for fragment in fragments if fragment]
        if not molecules:
            return ""  # Return empty string if no valid molecules are found

        max_mol = max(
            molecules, key=lambda mol: mol.GetNumAtoms() if mol else 0, default=None
        )
        return Chem.MolToSmiles(max_mol) if max_mol else ""

    @staticmethod
    def update_smiles(
        list_smiles: List[str],
        solution: List[str],
        prune: bool = True,
        starting_compound: str = None,
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
        for r in solution:
            smiles = r.split(">>")[1].split(".")
            if prune:
                smiles = CRN.get_max_fragment(smiles)
                if CRN.count_carbons(smiles) >= CRN.count_carbons(starting_compound):
                    list_smiles.extend([smiles])
            else:
                list_smiles.extend(smiles)
        return list(set(list_smiles))

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
        for r in rule_list:
            r = CoreEngine()._inference(r["gml"], smiles_list)
            r = Cleanning().clean_smiles(r)
            r = [_remove_reagent(i) for i in r]
            solution.extend(r)
        return solution

    def _build_crn(self, starting_compound: str) -> List[Dict[str, List[str]]]:
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
                    smiles, solution, starting_compound=starting_compound
                )
            solution = self._expand(self.rule_list, smiles)
            solutions.append({f"Round {i}": solution})

        return solutions
