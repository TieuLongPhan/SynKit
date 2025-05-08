import importlib.util
import importlib.resources

from typing import List, Tuple
from synkit.IO.debug import setup_logging
from synkit.IO.data_io import load_database

if importlib.util.find_spec("mod"):
    from mod import BondType
else:
    BondType = None
    print("Optional 'mod' package not found")


logger = setup_logging()


class ValenceConstrain:
    def __init__(self):
        """
        Initialize the ValenceConstrain class by setting up bond type orders and loading
        the maximum valence data.

        Parameters:
        - None

        Returns:
        - None
        """
        self.btToOrder = {
            BondType.Single: 1,
            BondType.Double: 2,
            BondType.Triple: 3,
            BondType.Aromatic: 0,
        }
        maxValence_path = importlib.resources.files("synkit.Rule.Compose").joinpath(
            "MaxValence.json.gz"
        )
        self.maxValence = load_database(maxValence_path)[0]

    def valence(self, vertex) -> int:
        """
        Calculate the valence of a vertex based on its incident edges.

        Parameters:
        - vertex (Vertex): The vertex for which to calculate the valence.

        Returns:
        - int: The total valence of the vertex.
        """
        return sum(self.btToOrder[edge.bondType] for edge in vertex.incidentEdges)

    def check_rule(self, rule, verbose: bool = False, log_error: bool = False) -> bool:
        """
        Check if the rule is chemically valid according to valence rules.

        Parameters:
        - rule (Rule): The rule to check for chemical validity.
        - verbose (bool): If true, logs additional information about the rule
        checking process.
        - log_error (bool): If true, logs additional information about the valence
        checking issue.

        Returns:
        - bool: True if the rule is chemically valid, False otherwise.
        """
        try:
            for vertex_pair in rule.vertices:
                left_valence = self.valence(vertex_pair.left)
                right_valence = self.valence(vertex_pair.right)
                left_label = vertex_pair.left.stringLabel
                right_label = vertex_pair.right.stringLabel

                if left_valence != right_valence:
                    raise ValueError(
                        f"Valence mismatch: left {left_valence} vs right {right_valence}"
                    )

                if left_valence > self.maxValence.get(
                    left_label, 0
                ) or right_valence > self.maxValence.get(right_label, 0):
                    if verbose:
                        logger.info(
                            f"Bad Rule for vertex {left_label} --->"
                            + "Exceeds max chemical valence"
                        )
                    return False
            return True
        except Exception as e:
            if log_error:
                logger.error(f"Error checking rule {rule}: {e}")
            return False

    def split(self, rules: List) -> Tuple[List, List]:
        """
        Split rules into 'good' and 'bad' based on their chemical validity.

        Parameters:
        - rules (List[Rule]): A list of rules to be checked and split.

        Returns:
        - Tuple[List[Rule], List[Rule]]: A tuple containing two lists, one for
        'good' rules and another for 'bad' rules.
        """
        good, bad = [], []
        for rule in rules:
            if self.check_rule(rule):
                good.append(rule)
            else:
                bad.append(rule)
        return good, bad
