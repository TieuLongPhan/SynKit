"""Identity adapter shared by native and legacy rule-composition paths."""

from __future__ import annotations

from typing import Any

import networkx as nx

from synkit.Graph.syn_graph import SynGraph
from synkit.Rule.syn_rule import SynRule


def rule_objects_isomorphic(left: Any, right: Any) -> bool:
    """Compare native stereo-bearing rules before falling back to MOD."""
    if isinstance(left, SynRule) or isinstance(right, SynRule):
        return isinstance(left, SynRule) and isinstance(right, SynRule) and left == right
    if isinstance(left, SynGraph) or isinstance(right, SynGraph):
        return isinstance(left, SynGraph) and isinstance(right, SynGraph) and left == right
    if isinstance(left, nx.Graph) or isinstance(right, nx.Graph):
        return (
            isinstance(left, nx.Graph)
            and isinstance(right, nx.Graph)
            and SynGraph(left) == SynGraph(right)
        )
    return left.isomorphism(right) == 1
