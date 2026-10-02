import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.ablation_worker import enumerate_variant
from Experiment.Synister.all_distance_oracle import literal_sets
from Experiment.Synister.output_scaling import analytic_checks, audit, families, make_tasks, run
from Experiment.Synister.scaling_worker import endpoint


def test_families_separate_atom_count_from_mapping_count():
    rows = families()
    assert len(rows) == 26
    fixed_n = [r for r in rows if r["family"] == "fixed_atom_count_variable_output"]
    assert {len(endpoint(r["reactant"]).atomic_numbers) for r in fixed_n} == {12}
    fixed_l = [r for r in rows if r["family"] == "fixed_output_variable_atom_count"]
    assert {r["expected_indexed_maps"] for r in fixed_l} == {2}
    for row in rows:
        r, p = endpoint(row["reactant"]), endpoint(row["product"])
        assert r == p
        if row["benchmark_id"] in ("fixed12_k2", "fixed12_k3", "edges_k1", "edges_k2", "path_n6"):
            assert len(literal_sets(r, p)[0]) == row["expected_indexed_maps"]
            for variant in ("full", "no_symmetry"):
                observed = enumerate_variant(r, p, variant=variant, target=0)
                assert observed["complete"] and len(observed["mappings"]) == row["expected_indexed_maps"]


def test_repeats_preserve_inputs_and_alternate_variant_order():
    rows = families()
    tasks = make_tasks(rows)
    assert len(tasks) == 156 and len({t["task_id"] for t in tasks}) == 156
    assert [t["variant"] for t in tasks[:2]] == ["full", "no_symmetry"]
    assert [t["variant"] for t in tasks[52:54]] == ["no_symmetry", "full"]
    assert {t["seconds"] for t in tasks} == {15}


def test_closed_form_counts_reject_false_completion_but_allow_partial_output():
    row = {"benchmark_id": "a", "expected_indexed_maps": 24}
    record = {"task": {"task_id": "a", "benchmark_id": "a"}, "mapping_count": 10, "complete": True}
    assert not analytic_checks([row], [record])[0]["consistent"]
    record["complete"] = False
    assert analytic_checks([row], [record])[0]["consistent"]
    record["mapping_count"] = 25
    assert not analytic_checks([row], [record])[0]["consistent"]


def test_small_serial_study_and_independent_saved_output_audit(tmp_path, monkeypatch):
    import Experiment.Synister.output_scaling as module
    selected = [row for row in families() if row["benchmark_id"] == "fixed12_k2"]
    monkeypatch.setattr(module, "families", lambda: selected)
    args = SimpleNamespace(output=tmp_path/"study", after=None, repeats=2,
                           seconds=5, memory_gib=2, max_maps=100)
    report = run(args)
    assert report["all_consistent"] and report["attempts"] == 4
    checked = audit(args.output)
    assert checked["all_verified"] and checked["validated_saved_maps"] == 8
    bad = json.loads((args.output/"summary.json").read_text())
    bad["selected"] = 2
    (args.output/"summary.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="denominator"):
        audit(args.output)
    with pytest.raises(FileExistsError):
        run(args)
