import pytest
import hashlib
import json
from pathlib import Path

from Experiment.Synister.closure_profile import FLAGS, checkpoints, summarize


def search(seconds=1):
    return dict(status="complete", parent_seconds=seconds, **dict.fromkeys(FLAGS, True))


def test_terminal_threshold_equality_and_missing_score():
    rows = [checkpoints(search(), {"status": "complete", "parent_seconds": 9}),
            checkpoints(search(10), None),
            checkpoints({"status": "hard_timeout", "parent_seconds": 65}, None)]
    result = summarize(rows)
    assert result["selected"] == 3
    assert result["thresholds"][0]["search_available"] == 1
    assert result["thresholds"][1]["search_available"] == 2
    assert result["thresholds"][1]["search_plus_score_available"] == 1
    assert result["final_available"] == {"search": 2, "search_plus_score": 1}
    for field in ("search_available", "search_plus_score_available"):
        counts = [row[field] for row in result["thresholds"]]
        assert counts == sorted(counts)


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [False, None, 1])
def test_incomplete_or_nonboolean_flags_are_not_closure(flag, value):
    record = search()
    record[flag] = value
    assert checkpoints(record, None) == (None, None)


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_invalid_or_missing_time(value):
    with pytest.raises(ValueError):
        checkpoints(search(value), None)
    with pytest.raises(ValueError):
        checkpoints(search(), {"status": "complete", "parent_seconds": value})


@pytest.mark.parametrize("cohort,total", [("c1", 1000), ("c2", 500)])
def test_full_archive_profile_and_bindings(cohort, total):
    root = Path(__file__).resolve().parents[3]
    evidence = root / "paper/synister/evidence"
    archive = evidence / f"identifiability_{cohort}_primary_v1"
    candidate = evidence / f"{cohort}_terminal_closure_v1.json"
    result = json.loads(candidate.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, filename in [("inputs", "inputs.json"), ("sources", "all_sources.json"),
                           ("manifest", "manifest.json"), ("summary", "summary.json"),
                           ("primary_audit", "audit.json")]:
        assert result[f"{name}_sha256"] == digest(archive / filename)
    assert result["reporter_sha256"] == digest(root / "Experiment/Synister/closure_profile.py")
    assert result["protocol_sha256"] == digest(root / "paper/synister/protocols/TERMINAL_CLOSURE_PROFILE_V1.md")
    resources = json.loads((evidence / f"{cohort}_resources_v1.json").read_text())
    assert result["record_hashes"] == resources["record_hashes"]
    for name, expected in result["record_hashes"].items():
        assert digest(archive / "cases" / name) == expected
    inputs = json.loads((archive / "inputs.json").read_text())
    assert len(inputs) == len({r["case_id"] for r in inputs}) == total
    search_times, combined_times = [], []
    for item in inputs:
        name = item["case_id"]
        record = json.loads((archive / "cases" / f"{name}.exact.json").read_text())
        if record["status"] != "complete":
            continue
        assert all(record[k] is True for k in (
            "minimum_proved", "enumeration_complete", "symmetry_search_complete", "joint_labels_complete"))
        search_times.append(record["parent_seconds"])
        path = archive / "cases" / f"{name}.score.json"
        if path.exists():
            score = json.loads(path.read_text())
            if score["status"] == "complete":
                combined_times.append(record["parent_seconds"] + score["parent_seconds"])
    assert result["profile"]["selected"] == total
    assert result["profile"]["final_available"] == {
        "search": len(search_times), "search_plus_score": len(combined_times)}
    assert [r["seconds"] for r in result["profile"]["thresholds"]] == [1, 10, 30, 60, 65, 100]
    for row in result["profile"]["thresholds"]:
        for label, values in [("search", search_times), ("search_plus_score", combined_times)]:
            available = len([v for v in values if v <= row["seconds"]])
            assert row[f"{label}_available"] == available
            assert row[f"{label}_unavailable"] == total - available
