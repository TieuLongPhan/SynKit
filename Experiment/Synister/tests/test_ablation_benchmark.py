import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.ablation_worker import CONFIGURATIONS, enumerate_variant, feasible_seed
from Experiment.Synister.ablation_benchmark import derive_tasks, execute, run, select_inputs
from Experiment.Synister.all_distance_oracle import binary_endpoint, literal_sets, weighted_cases
from Experiment.Synister.global_milp import doubled_distance
from Experiment.Synister.validate_ablations import check
from synkit.Chem.Mapper.identifiability import parse_reaction


def test_every_configuration_matches_literal_sets_at_every_small_case_distance():
    for a, b in ((0, 63), (3, 12), (15, 42), (63, 63)):
        assert check(f"binary_{a}_{b}", binary_endpoint(a), binary_endpoint(b), binary=True)["passed"]
    for name, r, p in weighted_cases(3):
        assert check(name, r, p)["passed"]


def test_seed_is_deterministic_feasible_and_never_privileged_by_optimum():
    for name, r, p in weighted_cases(6):
        seed, swaps = feasible_seed(r, p)
        assert feasible_seed(r, p) == (seed, swaps)
        oracle = literal_sets(r, p)
        assert seed in oracle[doubled_distance(r, p, seed)]
    r, p = parse_reaction("CCO>>CCO")
    complete = enumerate_variant(r, p, variant="full", target="minimal")
    assert complete["seed_mapping"] is not None and complete["seed_seconds"] >= 0
    assert complete["first_map_seconds"] >= complete["seed_seconds"]
    unseeded = enumerate_variant(r, p, variant="no_seed", target="minimal")
    assert unseeded["seed_mapping"] is None and unseeded["seed_doubled_cd"] is None
    basic = enumerate_variant(r, p, variant="basic_bounds", target="minimal")
    assert basic["lower_bound_pruned_branches"] == basic["upper_bound_pruned_branches"] == 0
    assert basic["backend_statistics"]["search"]["profile_bound_calls"] == 0
    assert complete["mappings"] == unseeded["mappings"] == basic["mappings"]
    with pytest.raises(ValueError, match="Unknown"):
        enumerate_variant(r, p, variant="unknown", target=0)


def make_matched(directory, proved=True):
    directory.mkdir()
    (directory/"cases").mkdir()
    rows = [{"benchmark_id": "reaction_000", "reaction": "CC>>CC", "source": "FlowER",
             "size_bin": 0, "selection_sha256": "00"}]
    (directory/"inputs.json").write_text(json.dumps(rows))
    for method in ("synister", "milp"):
        (directory/"cases"/f"reaction_000.minimum.{method}.json").write_text(json.dumps(
            {"minimum_proved": proved, "minimum_doubled_cd": 0 if proved else None}))
    (directory/"summary.json").write_text("{}")
    return rows


def test_input_selection_and_missing_targets_do_not_filter_unsolved_reactions(tmp_path):
    path = tmp_path/"matched"
    rows = make_matched(path, proved=False)
    assert select_inputs(path, 4) == rows
    tasks, unavailable = derive_tasks(rows, path, seconds=1, memory_gib=2, max_maps=100)
    assert len(tasks) == 5 and len(unavailable) == 1
    assert all(t["query"] == "minimum" and not t["minimum_proofs"] for t in tasks)


def test_target_proofs_and_rotating_repeated_configuration_order(tmp_path):
    path = tmp_path/"matched"
    rows = make_matched(path)
    tasks, unavailable = derive_tasks(rows, path, seconds=5, memory_gib=2, max_maps=100, repeats=2)
    assert len(tasks) == 30 and not unavailable
    assert len({t["task_id"] for t in tasks}) == 30
    assert {t["target_doubled_cd"] for t in tasks} == {"minimal", 0, 4}
    for task in tasks:
        assert bool(task["minimum_proofs"]) == (task["query"] != "minimum")
    first = [t["variant"] for t in tasks if t["query"] == "minimum" and t["repeat"] == 0]
    second = [t["variant"] for t in tasks if t["query"] == "minimum" and t["repeat"] == 1]
    assert first != second and set(first) == set(second) == set(CONFIGURATIONS)


def test_isolated_output_cap_and_zero_search_time_are_not_complete(tmp_path):
    (tmp_path/"cases").mkdir()
    (tmp_path/"maps").mkdir()
    task = {"task_id": "example.minimum.full", "reaction": "C.C.C.C>>C.C.C.C",
            "variant": "full", "target_doubled_cd": "minimal", "seconds": 5,
            "memory_gib": 2, "max_maps": 1}
    record = execute(task, tmp_path)
    assert not record["complete"] and record["termination"] == "mapping_limit"
    assert record["minimum_proved"] and record["mapping_count"] == 1
    assert record["peak_rss_kib"] > 0
    with pytest.raises(FileExistsError):
        execute(task, tmp_path)
    r, p = parse_reaction("CCO>>CCO")
    timed = enumerate_variant(r, p, variant="no_seed", target="minimal", seconds=0)
    assert not timed["complete"] and not timed["minimum_proved"] and not timed["mappings"]


def test_small_full_study_saves_all_attempts_and_matching_outputs(tmp_path):
    path = tmp_path/"matched"
    make_matched(path)
    args = SimpleNamespace(matched=path, output=tmp_path/"study", per_bin=1,
                           seconds=5, memory_gib=2, max_maps=100, repeats=1,
                           variants=list(CONFIGURATIONS), workers=2)
    report = run(args)
    assert report["selected"] == 1 and report["attempts"] == 15
    assert report["all_output_comparisons_consistent"]
    assert all(c["both_complete"] for c in report["comparisons"])
    assert report["by_variant"] == {v: {"complete": 3} for v in CONFIGURATIONS}
    from Experiment.Synister.audit_ablations import audit
    audited = audit(args.output)
    assert audited["all_output_comparisons_consistent"] and audited["validated_saved_maps"] == 20
    changed = json.loads((args.output/"summary.json").read_text())
    changed["attempts"] = 14
    (args.output/"summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="accounting"):
        audit(args.output)
    with pytest.raises(FileExistsError):
        run(args)
