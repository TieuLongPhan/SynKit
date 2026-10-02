import json
from types import SimpleNamespace

import pytest

from Experiment.Synister.all_distance_oracle import binary_endpoint, literal_sets
from Experiment.Synister.binary_backend_benchmark import binary_endpoints, prepare, run, solve


def test_both_binary_backends_match_every_literal_distance():
    for a, b in ((0, 63), (3, 12), (15, 42), (63, 63)):
        r, p = binary_endpoint(a), binary_endpoint(b)
        oracle = literal_sets(r, p)
        for target in range(0, 15, 2):
            for backend in ("assignment", "edit_support"):
                result = solve(r, p, target, backend)
                assert result["complete"]
                assert set(result["mappings"]) == oracle.get(target, set())


def test_binary_conversion_preserves_inventory_and_mutable_attributes():
    r, p = binary_endpoints("C=O>>CO")
    assert r.atomic_numbers == p.atomic_numbers == (6, 8)
    assert r.bonds == p.bonds == ((0, 1, 2),)
    assert r.hcounts != p.hcounts
    for backend in ("assignment", "edit_support"):
        result = solve(r, p, 0, backend)
        assert result["complete"] and result["mappings"] == [(0, 1)]


def test_caps_do_not_claim_complete_and_parity_rejection_is_explicit():
    r = binary_endpoint(0)
    for backend in ("assignment", "edit_support"):
        capped = solve(r, r, 0, backend, max_maps=1)
        assert not capped["complete"] and capped["mapping_count"] == 1
        empty = solve(binary_endpoint(3), binary_endpoint(12), 2, backend)
        assert empty["complete"] and empty["precheck"] == "binary_congruence"
    with pytest.raises(ValueError, match="Unknown"):
        solve(r, r, 0, "unknown")


def test_supplied_targets_and_complete_small_study_are_audited(tmp_path):
    selection = tmp_path/"selection.json"
    selection.write_text(json.dumps({"selected": [{"case_id": "a", "reaction": "CC>>CC", "compatible_maps": 2}]}))
    rows, references, tasks = prepare(selection, seconds=5, memory_gib=2, max_maps=100)
    assert len(rows) == 1 and len(tasks) == 6
    assert {t["target_doubled_cd"] for t in tasks} == {0, 2, 4}
    args = SimpleNamespace(selection=selection, output=tmp_path/"study", after=None,
                           seconds=5, memory_gib=2, max_maps=100)
    result = run(args)
    assert result["all_verified"] and result["oracle_replayed"] and result["attempts"] == 6
    assert result["validated_saved_maps"] == 4
    assert result["by_method"] == {m: {"complete": 3} for m in ("assignment", "edit_support")}
    with pytest.raises(FileExistsError):
        run(args)
