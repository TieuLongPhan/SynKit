"""Tests for the single-process hydrogen-extension comparison."""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx

from Experiment.Lewis.hydrogen_expand import benchmark as benchmark_module
from Experiment.Lewis.hydrogen_expand.benchmark import (
    DATASET,
    load_pickle,
    run_hextend,
    select_reference_cases,
    sha256,
    summarize,
    summarize_by_hcount,
)
from Experiment.Lewis.hydrogen_expand.reference_methods import run_reference_methods
from synkit.Graph.Hyrogen.hextend import HExtend
from synkit.Graph.Matcher.graph_cluster import GraphCluster


def test_hydrogen_corpus_is_the_official_109_reaction_payload() -> None:
    assert DATASET.is_file()
    assert sha256(DATASET) == (
        "d0b64765d9da17f34b983b4293a5bdb02f8928cf91c16a129485996c5dc35cbb"
    )
    assert len(load_pickle(DATASET)) == 109


def test_reference_filters_reproduce_the_published_104_reactions() -> None:
    accepted, excluded = select_reference_cases(load_pickle(DATASET))

    assert len(accepted) == 104
    assert {item["reason"] for item in excluded} == {
        "uneven_aam",
        "no_unmatched_hydrogens",
    }


def test_reference_methods_reproduce_known_class_count() -> None:
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-3666"
    )

    result = run_reference_methods(reaction["ITS"])

    assert result["method_a_classes"] == 3
    assert result["method_b_classes"] == 3
    assert result["backend"] == "rb_nx"
    assert result["unmatched_hydrogens"] == 3


def test_new_hextend_uses_the_same_full_its_class_contract() -> None:
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-3666"
    )

    rows = run_hextend([reaction], repetitions=1, timeout=10)
    new_row = next(row for row in rows if row["method"] == "hextend_new")

    assert new_row["completed_its"] == 3
    assert new_row["unique_classes"] == 3


def test_ambiguous_two_hydrogen_transfer_does_not_collapse() -> None:
    """R-50548 exposes legacy relabeling instead of transfer enumeration."""
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-50548"
    )

    rows = run_hextend([reaction], repetitions=1, timeout=10)
    by_method = {row["method"]: row for row in rows}

    assert by_method["hextend_legacy"]["completed_its"] == 2
    assert by_method["hextend_legacy"]["unique_classes"] == 1
    assert by_method["hextend_new"]["completed_its"] == 2
    assert by_method["hextend_new"]["unique_classes"] == 2

    unique_rc, unique_its, signatures = HExtend.extend_unique_full_its(
        reaction["ITS"]
    )
    assert len(unique_rc) == len(unique_its) == len(signatures) == 2


def test_hydrogen_distance_invariant_splits_symmetric_rc_collision() -> None:
    """R-16362 avoids an expensive exact match after its RC hash collision."""
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-16362"
    )

    _, completed_its, signatures = HExtend.extend_its(reaction["ITS"])
    distance_signatures = {
        HExtend.hydrogen_distance_invariant(its) for its in completed_its
    }
    clusters, _ = HExtend.cluster_full_its(completed_its, signatures)

    assert len(completed_its) == 2
    assert len(set(signatures)) == 1
    assert len(distance_signatures) == 2
    assert len(clusters) == 2


def test_hydrogen_distance_invariant_is_map_independent() -> None:
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-16362"
    )
    _, completed_its, signatures = HExtend.extend_its(reaction["ITS"])
    original = completed_its[0]
    relabeled = nx.relabel_nodes(
        original,
        {node: f"vertex-{index}" for index, node in enumerate(original)},
        copy=True,
    )

    assert HExtend.hydrogen_distance_invariant(
        original
    ) == HExtend.hydrogen_distance_invariant(relabeled)
    clusters, mapping = HExtend.cluster_full_its(
        [original, relabeled],
        [signatures[0], signatures[0]],
    )
    assert clusters == [{0, 1}]
    assert mapping == {0: 0, 1: 0}


def test_rc_anchored_quotient_matches_exact_clustering_on_full_corpus() -> None:
    baseline = GraphCluster()
    payload, _ = select_reference_cases(load_pickle(DATASET))

    for reaction in payload:
        _, completed_its, signatures = HExtend.extend_its(
            reaction["ITS"]
        )
        expected, _ = baseline.iterative_cluster(
            completed_its,
            nodeMatch=baseline.nodeMatch,
            edgeMatch=baseline.edgeMatch,
        )
        observed, _ = HExtend.cluster_full_its(
            completed_its,
            signatures,
        )

        assert observed == expected, reaction["R-id"]


def test_anchored_quotient_does_not_mutate_candidate_graphs() -> None:
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-51355"
    )
    _, completed_its, signatures = HExtend.extend_its(reaction["ITS"])
    before = [
        {node: dict(attributes) for node, attributes in graph.nodes(data=True)}
        for graph in completed_its
    ]

    HExtend.cluster_full_its(completed_its, signatures)

    after = [
        {node: dict(attributes) for node, attributes in graph.nodes(data=True)}
        for graph in completed_its
    ]
    assert after == before


def test_exact_fallback_resolves_a_deliberate_invariant_collision() -> None:
    path = nx.path_graph(4)
    star = nx.star_graph(3)
    for graph in (path, star):
        nx.set_node_attributes(graph, "C", "element")
        nx.set_node_attributes(graph, 0, "charge")
        nx.set_edge_attributes(graph, (1, 1), "order")

    clusters, mapping = HExtend.cluster_full_its(
        [path, star],
        ["deliberate-collision", "deliberate-collision"],
    )

    assert clusters == [{0}, {1}]
    assert mapping == {0: 0, 1: 1}


def test_summary_keeps_capability_failures_separate_from_outputs() -> None:
    rows = [
        {"method": "method_a", "status": "ERROR", "seconds": 0.01},
        {"method": "method_a", "status": "OUTPUT", "seconds": 0.03},
        {
            "method": "hextend_new",
            "status": "OUTPUT",
            "seconds": 0.02,
            "unique_classes": 3,
        },
    ]

    by_method = {item["method"]: item for item in summarize(rows)}

    assert by_method["method_a"]["success_rate"] == 0.5
    assert by_method["method_a"]["errors"] == 1
    assert by_method["hextend_new"]["mean_unique_classes"] == 3


def test_timed_call_runs_without_posix_interval_timers(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "_supports_interval_timer",
        lambda: False,
    )

    _elapsed, result, error = benchmark_module.timed_call(lambda: "ok", 1.0)

    assert result == "ok"
    assert error is None


def test_runner_is_executable() -> None:
    runner = Path(
        "Experiment/Lewis/hydrogen_expand/run_comparison.sh"
    ).resolve()

    assert runner.is_file()
    assert runner.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    if os.name != "nt":
        assert runner.stat().st_mode & 0o111


def test_table_summary_groups_reaction_means_by_hcount() -> None:
    rows = [
        {
            "method": "method_a_rb_nx",
            "record_id": "R-1",
            "unmatched_hydrogens": 2,
            "status": "OUTPUT",
            "seconds": 0.001,
        },
        {
            "method": "method_a_rb_nx",
            "record_id": "R-1",
            "unmatched_hydrogens": 2,
            "status": "OUTPUT",
            "seconds": 0.003,
        },
    ]

    summary = summarize_by_hcount(rows)[0]

    assert summary["reactions"] == 1
    assert summary["repetitions_per_reaction"] == [2]
    assert summary["mean_ms"] == 2
    assert summary["population_std_ms"] == 0
