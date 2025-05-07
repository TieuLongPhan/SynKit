from typing import List, Any, Dict

# from synkit.Synthesis.Reactor.core_engine import CoreEngine
from synkit.Synthesis.Reactor.mod_reactor import MODReactor


class SinglePredictor:
    """
    A class designed for one-step chemical reaction predictions using transformation rules.

    This class utilizes transformation rules to predict the outcomes of chemical reactions based
    on provided SMILES strings.
    """

    def __init__(self) -> None:
        """
        Initializes the StepPredictor instance.
        """
        pass

    def _single_rule(
        self, smiles_list: List[str], rule: str, invert: bool = False
    ) -> List[Any]:
        """
        Applies a single transformation rule to a list of SMILES strings.

        This function applies the transformation rule to generate potential reaction outcomes from
        given SMILES strings. The results are returned and the memory is cleaned up immediately
        after processing to handle large datasets efficiently.

        Parameters:
        - smiles_list (List[str]): The list of SMILES strings to process.
        - rule (str): The file path to the transformation rule.
        - prediction_type (str, optional): The type of prediction, either 'forward' or 'reverse'. Defaults to 'forward'.

        Returns:
        - List[Any]: The list of reaction outcomes.

        Raises:
        - Exception: If an error occurs during the processing of the rule.
        """
        reactor = MODReactor(smiles_list, rule, invert=invert, strategy="bt")
        reactor.run()
        reactions = reactor.get_reaction_smiles()
        return reactions

    def _multiple_rules(
        self, smiles_list: List[str], rules: List[str], invert: bool = False
    ) -> List[Any]:
        """
        Applies multiple transformation rules to a list of SMILES strings.

        Parameters:
        - smiles_list (List[str]): The list of SMILES strings to process.
        - rules (List[str]): The list of file paths to the transformation rules.
        - prediction_type (str, optional): The type of prediction, either 'forward' or 'reverse'. Defaults to 'forward'.

        Returns:
        - List[Any]: The accumulated list of reaction outcomes from all applied rules.
        """
        reactions = []
        for rule in rules:
            reaction = self._single_rule(smiles_list, rule, invert)
            reactions.extend(reaction)
        return reactions

    def _perform(
        self,
        data: List[Dict[str, Any]],
        rule_data: List[Dict[str, str]],
        reaction_key: str = "rsmi",
        rule_key: str = "gml",
        invert: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Performs prediction for each entry in the data using the specified rules.

        Parameters:
        - data (List[Dict[str, Any]]): The dataset containing chemical reactions.
        - rule_data (List[Dict[str, str]]): Data containing the transformation rules.
        - reaction_key (str): The key in the dataset for reaction SMILES.
        - rule_key (str): The key for the rule file paths.
        - prediction_type (str, optional): The type of prediction, either 'forward' or 'reverse'. Defaults to 'forward'.

        Returns:
        - List[Dict[str, Any]]: The dataset updated with the prediction results.
        """
        rules = [i[rule_key] for i in rule_data]
        for r in data:
            initial_smiles_list = (
                r[reaction_key].split(">>")[1].split(".")
                if invert
                else r[reaction_key].split(">>")[0].split(".")
            )
            r["raw"] = self._multiple_rules(initial_smiles_list, rules, invert)
        return data
