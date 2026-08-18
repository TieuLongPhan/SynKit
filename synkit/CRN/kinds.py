"""Shared node-kind vocabulary for SynKit-CRN bipartite graphs.

Every SynKit CRN graph is bipartite over *species* nodes and *reaction* nodes.
Historically the two halves of the package disagreed about how a reaction node
is spelled: the rule-expansion builder emits ``kind="rule"`` (a concrete rule
application), while :meth:`SynCRN.from_digraph` also accepts and round-trips
``kind="reaction"``. Modules that hard-coded a single spelling silently ignored
the other, producing empty stoichiometric matrices rather than an error.

This module is the single source of truth. Any code that classifies CRN graph
nodes must use :func:`is_species_node` and :func:`is_reaction_node` rather than
comparing the ``kind`` attribute directly.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN.kinds import is_reaction_node, REACTION_KINDS

    print(sorted(REACTION_KINDS))
    print(is_reaction_node({"kind": "reaction"}))
    print(is_reaction_node({"kind": "rule"}))
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "KIND_ATTR",
    "SPECIES_KIND",
    "REACTION_KIND",
    "RULE_KIND",
    "SPECIES_KINDS",
    "REACTION_KINDS",
    "ALL_KINDS",
    "node_kind",
    "is_species_node",
    "is_reaction_node",
]

#: Node attribute holding the kind discriminator.
KIND_ATTR = "kind"

#: Canonical spelling for species nodes.
SPECIES_KIND = "species"

#: Canonical spelling for reaction nodes.
REACTION_KIND = "reaction"

#: Legacy spelling for a concrete rule application, still emitted by
#: :class:`~synkit.CRN.Construct.builder.CRNExpand` and preserved on round-trip.
RULE_KIND = "rule"

#: All accepted spellings for species nodes.
SPECIES_KINDS = frozenset({SPECIES_KIND})

#: All accepted spellings for reaction nodes.
#:
#: ``"rule"`` denotes a concrete reaction instance produced by applying a rule;
#: it is *not* an abstract rule template. Abstract rules live in
#: :class:`~synkit.CRN.Structure.rule.Rule` records and never become graph nodes.
REACTION_KINDS = frozenset({REACTION_KIND, RULE_KIND})

#: Every node kind recognised anywhere in the CRN package.
ALL_KINDS = SPECIES_KINDS | REACTION_KINDS


def node_kind(data: Mapping[str, Any]) -> str:
    """Return the normalized node kind for a node-attribute mapping.

    The value is lower-cased and stripped so that ``"Rule"``, ``" rule "`` and
    ``"rule"`` compare equal. A missing ``kind`` yields the empty string.

    :param data:
        Node attribute mapping.
    :type data: Mapping[str, Any]

    :return:
        Normalized kind string, or ``""`` when absent.
    :rtype: str

    .. rubric:: Example

    .. code-block:: python

        node_kind({"kind": " Rule "})
        # 'rule'
    """
    return str(data.get(KIND_ATTR, "")).strip().lower()


def is_species_node(data: Mapping[str, Any]) -> bool:
    """Return whether a node-attribute mapping describes a species node.

    :param data:
        Node attribute mapping.
    :type data: Mapping[str, Any]

    :return:
        ``True`` when the node is a species node.
    :rtype: bool

    .. rubric:: Example

    .. code-block:: python

        is_species_node({"kind": "species"})
        # True
    """
    return node_kind(data) in SPECIES_KINDS


def is_reaction_node(data: Mapping[str, Any]) -> bool:
    """Return whether a node-attribute mapping describes a reaction node.

    Both ``kind="reaction"`` and the legacy ``kind="rule"`` are accepted, since
    both denote a concrete reaction instance.

    :param data:
        Node attribute mapping.
    :type data: Mapping[str, Any]

    :return:
        ``True`` when the node is a reaction node.
    :rtype: bool

    .. rubric:: Example

    .. code-block:: python

        is_reaction_node({"kind": "rule"})
        # True
        is_reaction_node({"kind": "reaction"})
        # True
    """
    return node_kind(data) in REACTION_KINDS
