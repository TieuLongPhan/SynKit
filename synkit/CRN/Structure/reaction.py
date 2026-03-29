from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Tuple


def _clean_counts(counts: Dict[str, int] | None = None) -> Dict[str, int]:
    """
    Normalize a species-to-coefficient mapping.

    :param counts: Raw mapping from species id to stoichiometric coefficient.
    :type counts: Dict[str, int] | None
    :returns: Cleaned mapping with strictly positive integer coefficients.
    :rtype: Dict[str, int]
    """
    out: Dict[str, int] = {}
    if not counts:
        return out

    for species_id, coeff in counts.items():
        if not isinstance(species_id, str):
            raise TypeError(f"Species ids must be str, got {type(species_id).__name__}")
        if not isinstance(coeff, int):
            raise TypeError(
                f"Stoichiometric coefficient for {species_id!r} must be int"
            )
        if coeff < 0:
            raise ValueError(
                f"Stoichiometric coefficient for {species_id!r} must be >= 0"
            )
        if coeff > 0:
            out[species_id] = coeff

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


@dataclass
class RXNSide:
    """
    Stoichiometric multiset for one reaction side.

    :param counts: Mapping ``species_id -> coefficient``.
    :type counts: Dict[str, int]
    """

    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counts = _clean_counts(self.counts)

    def __bool__(self) -> bool:
        return bool(self.counts)

    def __len__(self) -> int:
        return len(self.counts)

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        return iter(self.counts.items())

    def items(self) -> List[Tuple[str, int]]:
        """
        Return the side as a list of ``(species_id, coeff)`` pairs.

        :returns: Side entries.
        :rtype: List[Tuple[str, int]]
        """
        return list(self.counts.items())

    def get(self, species_id: str, default: int = 0) -> int:
        """
        Get the coefficient for one species.

        :param species_id: Internal species id.
        :type species_id: str
        :param default: Value to return when species is absent.
        :type default: int
        :returns: Stoichiometric coefficient.
        :rtype: int
        """
        return self.counts.get(species_id, default)

    def to_dict(self) -> Dict[str, int]:
        """
        Return a JSON-like dictionary representation.

        :returns: Species-to-coefficient mapping.
        :rtype: Dict[str, int]
        """
        return dict(self.counts)


@dataclass
class Reaction:
    """
    Canonical concrete reaction instance.

    :param id: Stable internal reaction id such as ``r_1``.
    :type id: str
    :param source_node_id: Original reaction-node id in the source graph.
    :type source_node_id: Hashable
    :param source_kind: Original source node kind, often ``"rule"``.
    :type source_kind: str
    :param lhs: Reactant multiset.
    :type lhs: RXNSide
    :param rhs: Product multiset.
    :type rhs: RXNSide
    :param label: Optional reaction label.
    :type label: Optional[str]
    :param step: Optional expansion or generation step.
    :type step: Optional[int]
    :param rule_index: Optional original rule index.
    :type rule_index: Optional[int]
    :param app_index: Optional application index.
    :type app_index: Optional[int]
    :param rule_repr: Optional original rule/template string.
    :type rule_repr: Optional[str]
    :param rule_id: Optional associated abstract rule id.
    :type rule_id: Optional[str]
    :param source_attrs: Exact original node attributes from the source graph.
    :type source_attrs: Dict[str, Any]
    :param metadata: Extra canonical metadata.
    :type metadata: Dict[str, Any]
    :param reactant_edge_attrs: Original edge attributes for reactant arcs,
        keyed by internal species id.
    :type reactant_edge_attrs: Dict[str, Dict[str, Any]]
    :param product_edge_attrs: Original edge attributes for product arcs,
        keyed by internal species id.
    :type product_edge_attrs: Dict[str, Dict[str, Any]]
    """

    id: str
    source_node_id: Hashable
    source_kind: str
    lhs: RXNSide
    rhs: RXNSide
    label: Optional[str] = None
    step: Optional[int] = None
    rule_index: Optional[int] = None
    app_index: Optional[int] = None
    rule_repr: Optional[str] = None
    rule_id: Optional[str] = None
    source_attrs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reactant_edge_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    product_edge_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def format_side(
        self,
        side: RXNSide,
        species_token: Callable[[str], str],
    ) -> str:
        """
        Format one reaction side as text.

        :param side: Left or right side.
        :type side: RXNSide
        :param species_token: Function mapping internal species ids to display text.
        :type species_token: Callable[[str], str]
        :returns: Human-readable reaction side.
        :rtype: str
        """
        if not side:
            return "∅"

        chunks: List[str] = []
        for species_id, coeff in side.items():
            token = species_token(species_id)
            chunks.append(token if coeff == 1 else f"{coeff}{token}")
        return " + ".join(chunks)

    def format(
        self,
        species_token: Callable[[str], str],
        *,
        include_id: bool = True,
        include_rule: bool = False,
        include_step: bool = False,
        arrow: str = ">>",
    ) -> str:
        """
        Format the full reaction as text.

        :param species_token: Function mapping internal species ids to display text.
        :type species_token: Callable[[str], str]
        :param include_id: Whether to include the internal reaction id.
        :type include_id: bool
        :param include_rule: Whether to include rule provenance.
        :type include_rule: bool
        :param include_step: Whether to include the step field.
        :type include_step: bool
        :param arrow: Arrow string used between sides.
        :type arrow: str
        :returns: Human-readable reaction string.
        :rtype: str
        """
        lhs = self.format_side(self.lhs, species_token)
        rhs = self.format_side(self.rhs, species_token)
        text = f"{lhs} {arrow} {rhs}"

        if include_id:
            text = f"{self.id}: {text}"

        suffix: List[str] = []
        if include_step and self.step is not None:
            suffix.append(f"step={self.step}")
        if include_rule and self.rule_id is not None:
            suffix.append(f"rule_id={self.rule_id}")
        elif include_rule and self.rule_index is not None:
            suffix.append(f"rule_index={self.rule_index}")
        if include_rule and self.rule_repr is not None:
            suffix.append(f"rule_repr={self.rule_repr}")

        if suffix:
            text += "  (" + ", ".join(suffix) + ")"
        return text

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-like dictionary representation.

        :returns: Reaction as a dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "source_kind": self.source_kind,
            "label": self.label,
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict(),
            "step": self.step,
            "rule_index": self.rule_index,
            "app_index": self.app_index,
            "rule_repr": self.rule_repr,
            "rule_id": self.rule_id,
            "source_attrs": dict(self.source_attrs),
            "metadata": dict(self.metadata),
            "reactant_edge_attrs": {
                k: dict(v) for k, v in self.reactant_edge_attrs.items()
            },
            "product_edge_attrs": {
                k: dict(v) for k, v in self.product_edge_attrs.items()
            },
        }
