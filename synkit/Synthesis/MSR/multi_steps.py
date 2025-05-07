from typing import List, Dict, Tuple, Union
from synkit.IO.debug import configure_warnings_and_logs
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Synthesis.reactor_utils import _add_reagent, _find_all_paths
from synkit.Synthesis.Reactor.mod_aam import MODAAM, expand_aam

configure_warnings_and_logs(True, True)


class MultiSteps:
    def __init__(self) -> None:
        """
        Initialize the MultiStep class with a Standardize instance.
        """
        self.std = Standardize()

    @staticmethod
    def _process(
        gml_list: List[str], order: List[int], rsmi: str, exclude_aam: bool = True
    ) -> Tuple[List[List[str]], Dict[str, List[str]]]:
        """
        Process a series of chemical reactions according to given rules and order.

        Parameters:
        - gml_list (List[str]): List of GML format strings representing reaction rules.
        - order (List[int]): Sequence of indices dictating the order of reactions.
        - rsmi (str): Starting reactant SMILES string.
        - exclude_aam (bool, optional): Flag to indicate whether to remove
        atom-atom mapping from the SMILES. Defaults to True.

        Returns:
        - Tuple[List[List[str]], Dict[str, List[str]]]: Tuple containing:
            - List of lists of SMILES strings for each step's products.
            - Dictionary mapping initial reactants to their corresponding products.
        """
        reaction_results = {}
        all_steps: List[List[str]] = []
        result: List[str] = [rsmi]

        for i, index in enumerate(order):
            current_step_gml = gml_list[index]
            new_result = []
            for current_rsmi in result:
                smi_lst = (
                    current_rsmi.split(">>")[0].split(".")
                    if i == 0
                    else current_rsmi.split(">>")[1].split(".")
                )
                reactor = MODAAM(
                    substrate=smi_lst,
                    rule_file=current_step_gml,
                    check_isomorphic=False,
                )
                o = reactor.get_reaction_smiles()
                # o = ReactorEngine()._inference(
                #     smi_lst,
                #     current_step_gml,
                #     complete_aam=False,
                #     check_isomorphic=False,
                # )
                o = [
                    Standardize().fit(product, remove_aam=exclude_aam) for product in o
                ]
                new_result.extend(o)
                if o:
                    reaction_results[current_rsmi] = o
            result = new_result
            all_steps.append(result)

        return all_steps, reaction_results

    @staticmethod
    def _get_aam(
        rsmi_list: List[str], rule_list: List[str], order: List[int]
    ) -> List[str]:
        """
        Apply atom-atom mapping to a series of reaction SMILES strings according to specified rules.

        Parameters:
        - rsmi_list (List[str]): List of reaction SMILES strings.
        - rule_list (List[List[str]]): Nested list where each sublist contains rules for atom-atom mapping.
        - order (List[int]): List of indices specifying which rules apply to each SMILES string.

        Returns:
        - List[str]: List of processed SMILES strings with atom-atom mapping applied.

        Raises:
        - TypeError: If any of the inputs are not of the correct type.
        - IndexError: If an index in 'order' is out of bounds for 'rule_list'.
        """
        if (
            not isinstance(rsmi_list, list)
            or not isinstance(rule_list, list)
            or not isinstance(order, list)
        ):
            raise TypeError("Invalid input types for rsmi_list, rule_list, or order.")
        if any(i >= len(rule_list) for i in order):
            raise IndexError("Index out of bounds in 'order' list.")

        steps = []
        for idx, rsmi in enumerate(rsmi_list):
            rules_to_apply = rule_list[order[idx]]
            new = expand_aam(rsmi, rules_to_apply)[0]
            steps.append(new)
        return steps

    def multi_step(
        self,
        original_rsmi: str,
        list_rule: List[str],
        order: List[int],
        cat: Union[str, List[str]],
    ) -> List[str]:
        """
        Orchestrate a multi-step chemical reaction process using a set of rules and a starting reactant.

        Parameters:
        - original_rsmi (str): Initial reactant SMILES string.
        - list_rule (List[str]): List of GML rules for the reactions.
        - order (List[int]): Order of application of the GML rules.
        - cat (Union[str, List[str]]): Catalysts or additional reagents to be added,
        can be a single string or a list of strings.

        Returns:
        - List[str]: List of reaction SMILES strings with atom-atom mapping applied after all steps.
        """
        if isinstance(cat, str):
            cat = [cat]  # Convert single string to list if necessary
        rsmi = _add_reagent(
            original_rsmi, reagents=cat
        )  # Add reagents to the original SMILES

        results, reaction_tree = self._process(list_rule, order, rsmi, exclude_aam=True)
        target_products = sorted(rsmi.split(">>")[1].split("."))
        max_depth = len(results)
        all_paths = _find_all_paths(reaction_tree, target_products, rsmi, max_depth)
        real_path = all_paths[0][1:]  # remove the original
        real_path = self._get_aam(real_path, list_rule, order)
        return real_path
