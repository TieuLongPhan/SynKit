from typing import List, Dict, Optional
from synkit.IO.chem_converter import smart_to_gml
from synkit.Rule.Compose.compose_rule import ComposeRule
from synkit.Rule.Compose.rule_mapping import RuleMapping


class SeqComp:
    """
    A class for generating pairwise mappings between sequential chemical reaction rules.

    This class takes a list of reaction SMARTS strings, converts them to their corresponding
    GML representations, composes candidate reaction rules for each consecutive pair, and computes
    a mapping between the rules using a rule mapping algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize an instance of the SeqComp class.
        """
        pass

    @staticmethod
    def sequence_map(smarts: List[str]) -> Dict[str, Optional[dict]]:
        """
        Generate pairwise mapping dictionaries between consecutive reaction SMARTS strings.

        This function processes a list of reaction SMARTS strings by:
          1. Converting each SMARTS string to its GML representation.
          2. For each consecutive pair, composing candidate rules using ComposeRule().get_rule_comp().
          3. Using the first candidate (if available) and the original GMLs to compute a mapping
             using RuleMapping().fit().
          4. Storing the resulting mapping in a dictionary with keys in the format "i:i+1".

        Parameters:
        - smarts (List[str]): The list of reaction SMARTS strings.

        Returns:
        - Dict[str, Optional[dict]]:
            A dictionary where each key is a string "i:i+1" representing the consecutive pair indices,
            and the corresponding value is the mapping dictionary produced by RuleMapping().fit()
            for that pair, or None if no valid mapping could be computed.
        """
        # Convert each SMARTS string to its GML representation.
        gml_list = [smart_to_gml(s, sanitize=True, reindex=False) for s in smarts]
        mappings: Dict[str, Optional[dict]] = {}

        # Process each consecutive pair in the list.
        for i in range(len(gml_list) - 1):
            # Get the consecutive SMARTS and GML representations.
            smart_a = smarts[i]
            smart_b = smarts[i + 1]
            rule_a = gml_list[i]
            rule_b = gml_list[i + 1]

            # Compose candidate rules between smart_a and smart_b.
            candidate_rules = ComposeRule().get_rule_comp(smart_a, smart_b)

            # If no candidate rule is found, assign None for this pair.
            if not candidate_rules:
                mappings[f"{i}:{i+1}"] = None
                continue

            # Try to compute the mapping using the first candidate rule.
            try:
                mapping_result = RuleMapping().fit(rule_a, rule_b, candidate_rules[0])
            except Exception as e:
                print(f"Error computing mapping for pair {i}:{i+1}: {e}")
                mapping_result = None

            mappings[f"{i}:{i+1}"] = mapping_result

        return mappings
