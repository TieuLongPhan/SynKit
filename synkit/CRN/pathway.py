from __future__ import annotations

from dataclasses import dataclass
from typing import List
from collections import Counter

from .network import ReactionNetwork


@dataclass
class Pathway:
    """
    Sequence of reaction applications forming a pathway.

    :param reaction_ids: Reaction indices in forward chronological order.
    :type reaction_ids: List[int]
    :param states: List of canonical states (multisets) from start to end.
    :type states: List[Counter[str]]
    """

    reaction_ids: List[int]
    states: List[Counter[str]]

    def as_original_rsmi_list(self, network: ReactionNetwork) -> List[str]:
        """
        Convert to the original atom-mapped reaction SMILES list.

        :param network: Network containing the source reactions.
        :type network: ReactionNetwork
        :return: Pathway as original reaction SMILES strings.
        :rtype: List[str]
        """
        return [network.reactions[rid].original_raw for rid in self.reaction_ids]

    def __repr__(self) -> str:
        return f"Pathway(steps={len(self.reaction_ids)})"
