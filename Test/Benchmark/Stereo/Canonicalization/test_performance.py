"""Controlled-cost measurement boundaries, full equality and censoring rules."""

import json
import os

import pytest
from rdkit import Chem

from Experiment.Stereo.Canonicalization import performance
from Experiment.Stereo.Canonicalization.performance import distribution, isolated, measure, summarize
from Experiment.Stereo.Canonicalization.performance_cases import cases


def _case(name):
    return next(value for value in cases() if value["id"] == name)


def test_public_selection_is_input_only_and_has_twenty_four_unique_constitutions():
    selected = [value for value in cases() if value["kind"] == "public"]
    keys = []
    for value in selected:
        molecule = Chem.MolFromSmiles(value["smiles"])
        Chem.RemoveStereochemistry(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        keys.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    assert len(keys) == len(set(keys)) == 24
    assert [value["input_atom_count"] for value in selected] == sorted(value["input_atom_count"] for value in selected)


def test_generator_and_full_group_outputs_preserve_the_same_complete_code(capsys):
    generator = measure(_case("tetrahedral:symmetric"), "generators", 10)
    group = measure(_case("tetrahedral:symmetric"), "complete_group", 10)
    capsys.readouterr()
    assert generator["status"] == group["status"] == "complete"
    assert generator["canonical_code"] == group["canonical_code"]
    assert not generator["primary_members"][0]["automorphisms_complete"]
    assert group["primary_members"][0]["automorphisms_complete"]
    assert group["primary_members"][0]["returned_witnesses"] == 12
    assert generator["primary_members"][0]["returned_witnesses"] < 12
    for result in (generator, group):
        assert result["certificate_bytes"] == len(result["canonical_code"].encode("utf-8"))
        assert result["canonical_mirror_equal"] == result["pairwise_mirror_equal"]
        assert result["primary_members"][0]["witness_seconds"] >= 0
        assert result["primary_members"][0]["search_seconds"] >= 0


@pytest.mark.parametrize("name,status", [
    ("boundary:unknown_tetrahedral", "incomplete_input"),
    ("boundary:unsupported_registry", "unsupported_input"),
    ("boundary:search_node_budget", "bounded_search"),
])
def test_expected_refusals_cannot_acquire_a_completed_certificate(name, status, capsys):
    result = measure(_case(name), "generators", 10)
    capsys.readouterr()
    assert result["status"] == status
    assert "canonical_code" not in result


def test_censored_runs_are_counted_without_entering_completed_runtime_summaries():
    records = [
        {"mode": "generators", "status": "timeout", "worker_wall_seconds": 30},
        {"mode": "generators", "status": "rss_limit", "worker_wall_seconds": 2},
    ]
    report = summarize(records)["generators"]
    assert report["planned"] == 2
    assert report["outcomes"] == {"timeout": 1, "rss_limit": 1}
    assert report["completed_only"]["worker_wall_seconds"] == distribution([])
    assert report["completed_only"]["worker_wall_seconds"]["median"] is None


def test_distribution_keeps_denominators_and_uses_nearest_rank_p95():
    assert distribution([1, 2, 3]) == {"count": 3, "median": 2, "p95": 3, "maximum": 3}
    assert distribution(list(range(1, 101)))["p95"] == 95
    assert json.loads(json.dumps(distribution([])))["count"] == 0


def test_parent_wall_budget_interrupts_a_worker_and_retains_no_certificate(tmp_path):
    result = isolated(_case("tetrahedral:distinct"), "generators", 0.001, 1024, dict(os.environ), tmp_path)
    assert result["status"] == "timeout"
    assert "canonical_code" not in result
    assert result["worker_returncode"] != 0
    assert result["parent_observed_cpu_seconds"] >= 0
    assert len(result["host_load_average_before"]) == len(result["host_load_average_after"]) == 3


def test_real_worker_records_cpu_and_wall_costs_with_host_contention_visible(tmp_path):
    environment = {**os.environ, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    result = isolated(_case("tetrahedral:distinct"), "generators", 10, 1024, environment, tmp_path)
    assert result["status"] == "complete"
    assert result["worker_cpu_total_seconds"] > 0
    assert 0 <= result["worker_operation_cpu_seconds"] <= result["worker_cpu_total_seconds"]
    assert result["worker_wall_seconds"] >= result["worker_operation_seconds"]
    assert len(result["host_load_average_before"]) == len(result["host_load_average_after"]) == 3


def test_parent_native_memory_budget_interrupts_a_worker(tmp_path):
    result = isolated(_case("tetrahedral:distinct"), "generators", 10, 0, dict(os.environ), tmp_path)
    assert result["status"] == "rss_limit"
    assert result["parent_observed_peak_rss_mib"] > 0
    assert "canonical_code" not in result


def test_worker_output_cap_cannot_turn_a_partial_json_record_into_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, "MAX_WORKER_OUTPUT_BYTES", 128)
    result = isolated(_case("tetrahedral:distinct"), "generators", 10, 1024, dict(os.environ), tmp_path)
    assert result["status"] == "worker_output_limit"
    assert "canonical_code" not in result
