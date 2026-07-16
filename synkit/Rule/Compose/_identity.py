"""Native identity and clustering helpers for reaction-rule objects."""

from __future__ import annotations

from typing import Any, Iterable, List

import networkx as nx

from synkit.Graph.syn_graph import SynGraph
from synkit.Rule.syn_rule import SynRule


def rule_objects_isomorphic(left: Any, right: Any) -> bool:
    """Compare supported native stereo-bearing rule or graph objects."""
    if isinstance(left, SynRule) or isinstance(right, SynRule):
        return (
            isinstance(left, SynRule) and isinstance(right, SynRule) and left == right
        )
    if isinstance(left, SynGraph) or isinstance(right, SynGraph):
        return (
            isinstance(left, SynGraph) and isinstance(right, SynGraph) and left == right
        )
    if isinstance(left, nx.Graph) or isinstance(right, nx.Graph):
        return (
            isinstance(left, nx.Graph)
            and isinstance(right, nx.Graph)
            and SynGraph(left) == SynGraph(right)
        )
    raise TypeError(
        "Rule identity supports SynRule, SynGraph, and NetworkX graph objects."
    )


def cluster_rule_objects(rules: Iterable[Any]) -> List[Any]:
    """Return one representative from each native rule-identity class."""
    representatives: List[Any] = []
    for candidate in rules:
        if not any(
            rule_objects_isomorphic(candidate, representative)
            for representative in representatives
        ):
            representatives.append(candidate)
    return representatives
