from __future__ import annotations

from textwrap import wrap
from typing import Dict, Hashable, Iterable, Optional

import networkx as nx


def _truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _wrap(text: str, wrap_at: Optional[int]) -> str:
    if wrap_at is None or wrap_at < 1 or len(text) <= wrap_at:
        return text
    return "\n".join(wrap(text, width=wrap_at))


def _species_text(graph: nx.DiGraph, node: Hashable, mode: str) -> str:
    data = graph.nodes[node]
    if mode == "index":
        return str(node)
    if mode == "label":
        return str(data.get("label", node))
    if mode == "smiles":
        return str(data.get("smiles", data.get("label", node)))
    raise ValueError(
        f"Unsupported species_label={mode!r}. Use 'index', 'label', or 'smiles'."
    )


def _rule_text(graph: nx.DiGraph, node: Hashable, mode: str) -> str:
    data = graph.nodes[node]
    if mode == "index":
        return str(node)
    if mode == "label":
        return str(data.get("label", node))
    if mode == "rule_index":
        return f"r{data.get('rule_index', node)}"
    if mode == "rule_app":
        return f"r@{data.get('rule_index', node)}@{data.get('app_index', '?')}"
    if mode == "step_rule_app":
        return (
            f"s{data.get('step', '?')}:"
            f"r@{data.get('rule_index', node)}@{data.get('app_index', '?')}"
        )
    raise ValueError(
        f"Unsupported rule_label={mode!r}. "
        "Use 'index', 'label', 'rule_index', 'rule_app', or 'step_rule_app'."
    )


def build_node_labels(
    graph: nx.DiGraph,
    *,
    species_nodes: Iterable[Hashable],
    rule_nodes: Iterable[Hashable],
    species_label: str = "label",
    rule_label: str = "label",
    show_species_labels: bool = True,
    show_rule_labels: bool = True,
    max_chars: Optional[int] = 32,
    wrap_at: Optional[int] = None,
) -> Dict[Hashable, str]:
    labels: Dict[Hashable, str] = {}

    for node in species_nodes:
        mode = species_label if show_species_labels else "index"
        labels[node] = _wrap(
            _truncate(_species_text(graph, node, mode), max_chars), wrap_at
        )

    for node in rule_nodes:
        mode = rule_label if show_rule_labels else "index"
        labels[node] = _wrap(
            _truncate(_rule_text(graph, node, mode), max_chars), wrap_at
        )

    return labels


def build_edge_labels(
    graph: nx.DiGraph,
    *,
    mode: str = "none",
    max_chars: Optional[int] = 20,
) -> Dict[tuple[Hashable, Hashable], str]:
    labels: Dict[tuple[Hashable, Hashable], str] = {}
    if mode == "none":
        return labels

    for u, v, d in graph.edges(data=True):
        if mode == "role":
            text = str(d.get("role", ""))
        elif mode == "stoich":
            text = str(d.get("stoich", ""))
        elif mode == "role_stoich":
            text = f"{d.get('role', '')}:{d.get('stoich', '')}"
        elif mode == "rxn_id":
            text = str(d.get("rxn_id", ""))
        elif mode == "step":
            text = str(d.get("step", ""))
        else:
            raise ValueError(
                f"Unsupported edge_label mode {mode!r}. "
                "Use 'none', 'role', 'stoich', 'role_stoich', 'rxn_id', or 'step'."
            )
        labels[(u, v)] = _truncate(text, max_chars)
    return labels
