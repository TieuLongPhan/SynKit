from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class Rule:
    """
    Abstract rule provenance shared by one or more concrete reactions.

    This does not become a node in the bipartite CRN graph. Instead, reaction
    nodes may carry ``kind="rule"`` in the input graph while still being
    interpreted as concrete reaction instances that reference a rule record.

    :param id: Stable internal rule id such as ``rule_1``.
    :type id: str
    :param rule_index: Optional original rule index.
    :type rule_index: Optional[int]
    :param rule_repr: Optional original rule/template string.
    :type rule_repr: Optional[str]
    :param label: Optional human-readable rule label.
    :type label: Optional[str]
    :param metadata: Extra metadata for the rule.
    :type metadata: Dict[str, Any]
    """

    id: str
    rule_index: Optional[int] = None
    rule_repr: Optional[str] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Signature used to deduplicate rules across reactions.

        :returns: ``(rule_index, rule_repr)``
        :rtype: Tuple[Optional[int], Optional[str]]
        """
        return (self.rule_index, self.rule_repr)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-like dictionary representation.

        :returns: Rule as a dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            "id": self.id,
            "rule_index": self.rule_index,
            "rule_repr": self.rule_repr,
            "label": self.label,
            "metadata": dict(self.metadata),
        }
