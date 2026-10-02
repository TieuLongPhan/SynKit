"""Controls for the publication-facing independent all-distance experiment."""

from types import SimpleNamespace
import json

import pytest

from Experiment.Synister.all_distance_oracle import (
    binary_endpoint, check_case, compatible_count, literal_sets, map_digest,
    run, select_real_cases, targets_for, weighted_cases,
)
from synkit.Chem.Mapper.identifiability import Endpoint


def test_empty_and_complete_graph_have_all_24_maps_at_distance_six():
    r, p = binary_endpoint(0), binary_endpoint(63)
    oracle = literal_sets(r, p)
    assert set(oracle) == {12}
    assert len(oracle[12]) == compatible_count(r, p) == 24
    assert targets_for(oracle, True) == list(range(0, 15, 2))
    result = check_case("complete", r, p, binary=True)
    assert result["passed"] and len(result["queries"]) == 18
    assert sum(q["expected_count"] == 0 for q in result["queries"]) == 14


def test_weighted_reordered_elements_and_half_bonds():
    r = Endpoint((6, 8, 6), (0, 0, 0), (0, 0, 0), ((0, 1, 3),))
    p = Endpoint((6, 6, 8), (0, 0, 0), (0, 0, 0), ((0, 2, 2),))
    oracle = literal_sets(r, p)
    assert oracle == {1: {(0, 2, 1)}, 5: {(1, 2, 0)}}
    assert check_case("half", r, p)["passed"]
    assert check_case("conditioned", r, p, fixed={0: 1})["passed"]


def test_changed_unary_attributes_do_not_restrict_compatible_atom_maps():
    r = Endpoint((6, 6), (0, 1), (4, 3), ())
    assert literal_sets(r, r) == {0: {(0, 1), (1, 0)}}
    assert check_case("unary", r, r)["passed"]


def test_generated_inputs_reproducible_and_full_sets_checked():
    assert list(weighted_cases(8)) == list(weighted_cases(8))
    for name, r, p in weighted_cases(4):
        assert check_case(name, r, p)["passed"]


def test_equal_counts_do_not_hide_wrong_sets_or_duplicates():
    r = Endpoint((6, 8, 6), (0, 0, 0), (0, 0, 0), ((0, 1, 3),))
    p = Endpoint((6, 6, 8), (0, 0, 0), (0, 0, 0), ((0, 2, 2),))
    def wrong(*args, **kwargs):
        return SimpleNamespace(mappings=[(1, 2, 0)], complete=True, status="complete",
                               minimum_cost=0.5, selected_mapping_count=1,
                               visited_nodes=0, symmetry_group_order=1)
    report = check_case("wrong", r, p, enumerator=wrong)
    assert not report["passed"]
    assert report["queries"][0]["missing_count"] == 1
    assert report["queries"][0]["extra_count"] == 1
    assert map_digest({(0, 2, 1)}) != map_digest({(1, 2, 0)})
    def duplicate(*args, **kwargs):
        value = wrong()
        value.mappings = [(0, 2, 1), (0, 2, 1)]
        return value
    assert check_case("duplicate", r, p, enumerator=duplicate)["queries"][0]["duplicate_count"] == 1


def test_small_report_and_no_overwrite(tmp_path):
    target = tmp_path / "results"
    result = run(target, binary_pairs=2, weighted=0, worked=False)
    assert result["all_passed"] and result["cases"] == 2
    assert result["queries"] == 36 and result["compatible_maps"] == 48
    from Experiment.Synister.audit_all_distance_oracle import audit
    verified = audit(target)
    assert verified["all_verified"] and verified["cases"] == 2
    altered = json.loads((target/"summary.json").read_text())
    altered["queries"] = 35
    (target/"summary.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="accounting"):
        audit(target)
    with pytest.raises(FileExistsError):
        run(target, binary_pairs=2, weighted=0, worked=False)


def test_real_selection_is_input_only_and_reports_shortfall(tmp_path):
    path = tmp_path / "inputs.json"
    items = [{"case_id": "a", "reaction": "CO>>CO", "status": "timeout"},
             {"case_id": "b", "reaction": "CCO>>CCO", "status": "complete"},
             {"case_id": "duplicate", "reaction": "CO>>CO"},
             {"case_id": "unbalanced", "reaction": "CO>>CC"},
             {"case_id": "large", "reaction": "CCCCCCCC>>CCCCCCCC"}]
    path.write_text(json.dumps(items))
    selection = select_real_cases([path], count=20, max_maps=10)
    assert selection["eligible"] == 2
    assert selection["requested"] == 20
    assert len(selection["unsupported"]) == 1
    assert {x["case_id"] for x in selection["selected"]} == {"a", "b"}
    assert selection["unique_inputs_examined"] == 4
