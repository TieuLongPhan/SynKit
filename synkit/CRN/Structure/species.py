from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Optional


@dataclass
class Species:
    """
    Canonical species record.

    :param id: Stable internal species id such as ``s_1``.
    :type id: str
    :param source_node_id: Original node id in the source NetworkX graph.
    :type source_node_id: Hashable
    :param label: Human-readable display label.
    :type label: str
    :param smiles: Optional SMILES string.
    :type smiles: Optional[str]
    :param source_attrs: Exact original node attributes from the source graph.
    :type source_attrs: Dict[str, Any]
    :param metadata: Optional extra canonical metadata.
    :type metadata: Dict[str, Any]
    """

    id: str
    source_node_id: Hashable
    label: str
    smiles: Optional[str] = None
    source_attrs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-like dictionary representation.

        :returns: Species as a dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "label": self.label,
            "smiles": self.smiles,
            "source_attrs": dict(self.source_attrs),
            "metadata": dict(self.metadata),
        }
