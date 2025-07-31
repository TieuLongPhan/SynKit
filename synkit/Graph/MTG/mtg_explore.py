from typing import Optional, List
from synkit.Graph.MTG.mtg import MTG
from synkit.Rule.Apply.rule_matcher import RuleMatcher
from synkit.Graph.MTG.mcs_matcher import MCSMatcher
from networkx import Graph


def find_mtg(
    g1: Graph,
    g2: Graph,
    ground_truth: str,
    node_label_names: Optional[List[str]] = None,
) -> Optional[MTG]:
    """
    Attempt to construct a Mapping Transformation Graph (MTG) for two input graphs
    by finding maximum common substructure mappings and validating against a ground truth.

    :param g1: The first input graph to match.
    :type g1: networkx.Graph
    :param g2: The second input graph to match.
    :type g2: networkx.Graph
    :param ground_truth: A string representation of the expected atom-atom mapping (AAM)
        used to validate candidate mappings.
    :type ground_truth: str
    :param node_label_names: List of node attribute names to use for MCS matching.
        Defaults to ["element", "charge", "hcount"].
    :type node_label_names: list of str, optional
    :returns: An MTG instance if a valid mapping satisfying the ground truth is found;
        otherwise, None.
    :rtype: MTG or None
    :raises ValueError: If input graphs are empty or ground_truth is invalid format.

    :example:
    >>> from networkx import Graph
    >>> g1, g2 = Graph(), Graph()
    >>> # populate g1 and g2 with nodes/edges
    >>> mtg = find_mtg(
    ...     g1,
    ...     g2,
    ...     ground_truth="{0:1, 1:0}",
    ...     node_label_names=["element", "charge", "hcount"]
    ... )
    >>> if mtg:
    ...     print(mtg)
    """
    # Validate inputs
    if not g1 or not g2:
        raise ValueError("Input graphs g1 and g2 must be non-empty.")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("Ground truth mapping must be a non-empty string.")

    # Set default node_label_names if not provided
    if node_label_names is None:
        node_label_names = ["element", "charge", "hcount"]

    # Initialize maximum common substructure matcher
    mcs = MCSMatcher(node_label_names=node_label_names)
    mcs.find_rc_mapping(g1, g2, mcs=False)
    mappings = mcs._mappings

    for mapping in mappings:
        # Construct MTG using current mapping
        mtg = MTG([g1, g2], mappings=[mapping])
        aam = mtg.get_aam()
        try:
            # Validate generated AAM against ground truth
            RuleMatcher(ground_truth, aam, explicit_h=False)
            return mtg
        except AssertionError:
            # Mapping did not satisfy ground truth, try next
            continue

    # No valid mapping found
    return None
