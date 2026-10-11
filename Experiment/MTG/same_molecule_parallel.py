#!/usr/bin/env python
"""Two grounded electron edits on one molecule, with a global-guard countercase.

This validates representation-level commutation, not simultaneous physical
reaction rates. The whole-material occurrence model is deliberately not used
to assign two consumers to the same molecule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synkit.Graph.Morphism import (  # noqa: E402
    ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import (  # noqa: E402
    DPOError,
    DPOIssue,
    DPOIssueCode,
    RuleSpan,
    apply_dpo,
)
from synkit.Rule.Compose import (  # noqa: E402
    LawError,
    LawIssueCode,
    commute_independent,
)


def _llg(graph):
    graph = graph.copy()
    for _, attrs in graph.nodes(data=True):
        for key in ELECTRON_LLG_SCHEMA.node_derived:
            attrs.pop(key, None)
    for _, _, attrs in graph.edges(data=True):
        for key in ELECTRON_LLG_SCHEMA.edge_derived:
            attrs.pop(key, None)
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(graph), ELECTRON_LLG_SCHEMA
    )


def _initial():
    graph = nx.Graph()
    for node, hydrogen in ((1, 3), (2, 2), (3, 2), (4, 3)):
        graph.add_node(
            node,
            element="C",
            aromatic=False,
            hcount=hydrogen,
            radical=0,
            lone_pairs=0,
            valence_electrons=4,
        )
    for left, right in ((1, 2), (2, 3), (3, 4)):
        graph.add_edge(left, right, sigma_order=1, pi_order=0)
    return _llg(graph)


def _cleave(host, edge):
    left = host.to_networkx().subgraph(edge).copy()
    right = left.copy()
    right.remove_edge(*edge)
    for node in edge:
        right.nodes[node]["radical"] += 1
    return RuleSpan.from_mapping(
        _llg(left),
        _llg(right),
        {node: node for node in edge},
        boundary="closed",
        name=f"homolysis:{edge}",
    )


@dataclass(frozen=True)
class NoRemoteRadicals:
    """A deliberately nonlocal admission condition on the peer's atom states."""

    remote_nodes: tuple[int, ...]

    def validate(self, host, match):
        if any(host.node_labels(node)["radical"] for node in self.remote_nodes):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.APPLICATION_CONDITION,
                    "Remote radicals violate this supplied global guard.",
                )
            )


def run():
    host = _initial()
    first, second = (_cleave(host, edge) for edge in ((1, 2), (3, 4)))
    certificate = commute_independent(first, {1: 1, 2: 2}, second, {3: 3, 4: 4}, host)
    assert certificate.replay()
    left = certificate.first_then_second[-1].result
    right = certificate.second_then_first[-1].result
    assert left.node_ids == right.node_ids == host.node_ids
    assert left.edge_keys == right.edge_keys == frozenset({frozenset({2, 3})})
    for node in host.node_ids:
        assert left.node_labels(node) == right.node_labels(node)
        assert left.node_labels(node)["radical"] == 1
        assert left.node_labels(node)["charge"] == 0
    assert left.edge_labels(frozenset({2, 3})) == right.edge_labels(frozenset({2, 3}))
    assert all(
        rule.resource_delta.element_delta == ()
        and rule.resource_delta.electron_delta == 0
        for rule in (first, second)
    )
    guarded_first = replace(first, condition=NoRemoteRadicals((3, 4)))
    guarded_second = replace(second, condition=NoRemoteRadicals((1, 2)))
    for rule, mapping in (
        (guarded_first, {1: 1, 2: 2}),
        (guarded_second, {3: 3, 4: 4}),
    ):
        assert apply_dpo(rule, host, mapping).certificate.replay().valid
    try:
        commute_independent(
            guarded_first, {1: 1, 2: 2}, guarded_second, {3: 3, 4: 4}, host
        )
    except LawError as error:
        assert any(
            issue.code is LawIssueCode.SEQUENTIAL_APPLICATION for issue in error.issues
        )
        refusal = [issue.to_dict() for issue in error.issues]
    else:
        raise AssertionError("Nonlocal guards must prevent the commuting claim.")
    return {
        "schema": "synkit.mtg-same-molecule-parallel/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_components": nx.number_connected_components(host.to_networkx()),
        "final_components": nx.number_connected_components(left.to_networkx()),
        "closed_applications_checked": 4,
        "anchored_endpoints_equal": True,
        "guarded_events_initially_admitted": 2,
        "nonlocal_guard_refused": refusal,
        "scope": "Grounded finite Lewis-state witness; no rates, likelihood or physical timing claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "closed_applications_checked")}
        )
    )


if __name__ == "__main__":
    main()
