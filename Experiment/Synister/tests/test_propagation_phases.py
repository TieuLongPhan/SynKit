"""Regression controls for post-search censorship and independent rescoring."""

from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.global_milp import doubled_distance
from Experiment.Synister.mapping_check import check_mappings
from Experiment.Synister import propagation_phases as phases


def test_chunked_checker_matches_literal_weighted_and_signed_objectives():
    r = SimpleNamespace(
        atomic_numbers=(6, 6, 6, 8), bonds=((0, 1, -3), (1, 2, 6), (2, 3, 3))
    )
    p = SimpleNamespace(atomic_numbers=(6, 6, 6, 8), bonds=((0, 2, 2), (1, 3, -2)))
    for permutation in permutations(range(3)):
        mapping = [*permutation, 3]
        expected = doubled_distance(r, p, mapping)
        assert check_mappings(r, p, [mapping] * 7, expected, chunk_size=2) == 7
        with pytest.raises(ValueError):
            check_mappings(r, p, [mapping], expected + 1)


@pytest.mark.parametrize(
    "mapping", [[0, 0, 2], [-1, 1, 2], [0, 1], [0, 2, 1], [False, 1, 2], [0.0, 1, 2]]
)
def test_checker_rejects_invalid_bijections_and_atom_types(mapping):
    endpoint = SimpleNamespace(atomic_numbers=(6, 6, 8), bonds=())
    with pytest.raises(ValueError):
        check_mappings(endpoint, endpoint, [mapping], 0)


def test_solver_report_survives_failure_during_output_publication(
    tmp_path, monkeypatch
):
    task = dict(
        benchmark_id="tiny",
        reaction="CC>>CC",
        method="synister_cp",
        prepared_seed=[0, 1],
        seconds=1,
        directory=str(tmp_path),
    )

    def fail_serialization(*args, **kwargs):
        raise RuntimeError("simulated output interruption")

    monkeypatch.setattr(phases.json, "dumps", fail_serialization)
    with pytest.raises(RuntimeError, match="interruption"):
        phases.solve(task)
    report = json.loads(phases.artifact(task, ".solver.json").read_text())
    assert report["methods"]["synister_cp"]["complete"]
    assert report["methods"]["synister_cp"]["minimum_doubled_cd"] == 0
    phases_seconds = report["methods"]["synister_cp"]["phase_seconds"]
    assert phases_seconds["solver_total"] >= phases_seconds["minimum_proof"] > 0
    assert phases_seconds["enumeration"] > 0
    search = report["methods"]["synister_cp"]["statistics"]["search"]
    assert search["minimum_proof_status"] == "proved"
    assert search["incumbent_quarters"] >= search["root_lower_bound_quarters"]
    assert search["root_gap_quarters"] == (
        search["incumbent_quarters"] - search["root_lower_bound_quarters"]
    )
    assert not phases.artifact(task, ".output.json").exists()


def test_specific_cd_solver_and_independent_validation(tmp_path):
    task = dict(
        benchmark_id="specific",
        reaction="CC>>CC",
        method="synister_cp",
        prepared_seed=[0, 1],
        target_mode="specific_cd",
        target_doubled_cd=0,
        seconds=2,
        directory=str(tmp_path),
    )
    report = phases.solve(task)
    method = report["methods"]["synister_cp"]
    assert method["complete"]
    assert method["target_mode"] == "specific_cd"
    assert method["target_doubled_cd"] == 0
    assert method["minimum_doubled_cd"] is None
    validation = phases.validate(task)
    assert validation["complete"]
    assert validation["mapping_count"] == method["mapping_count"]
    assert validation["target_doubled_cd"] == 0


def test_validation_timeout_keeps_real_solver_outcome(tmp_path, monkeypatch):
    task = dict(
        benchmark_id="tiny",
        reaction="CC>>CC",
        method="synister_cp",
        prepared_seed=[0, 1],
        seconds=1,
        directory=str(tmp_path),
    )
    method = dict(
        complete=True,
        minimum_doubled_cd=0,
        termination="complete",
        seconds=0.1,
        mapping_count=2,
        mapping_sha256=None,
        validation_complete=False,
    )
    phases.publish(
        phases.artifact(task, ".solver.json"), {"methods": {"synister_cp": method}}
    )
    phases.publish(phases.artifact(task, ".output.json"), {"mapping_sha256": "binding"})

    def fake_invoke(stage_task, *args):
        return {"parent_timeout": stage_task["stage"] == "validate", "wall_seconds": 1}

    monkeypatch.setattr(phases, "invoke", fake_invoke)
    result = phases.run_method(task, tmp_path, {})
    assert result["complete"] and result["solver_outcome_known"]
    assert result["minimum_doubled_cd"] == 0 and result["seconds"] == 0.1
    assert result["termination"] == "complete"
    assert result["interrupted_phase"] == "validation"
    assert not result["validation_complete"]
