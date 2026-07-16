import json

from synkit.Mechanism import BenchmarkCase, mechanism_from_legacy_epd
from synkit.Mechanism.benchmark import corrupt_record
from synkit.Mechanism.evidence import collect_evidence, write_evidence_report


def _record():
    return mechanism_from_legacy_epd(
        "[OH-:1].[CH3+:2]>>[CH3:2][OH:1]",
        [["LP-/Sigma+", [1], [1, 2]]],
    )


def test_evidence_runner_reports_replay_corruptions_and_resource_scope(tmp_path):
    case = BenchmarkCase("polar-00001", "polar", _record(), {})
    for partition in ("polar", "radical", "stereo"):
        payload = {
            "schema": f"MechanismBench-{partition}-test-v1",
            "cases": [case.to_dict()] if partition == "polar" else [],
        }
        (tmp_path / f"{partition}.json").write_text(json.dumps(payload))

    report = collect_evidence(tmp_path, repetitions=1)

    assert report["scope"]["typed_replay_cases"] == 1
    assert report["scope"]["partition_counts"] == {"polar": 1}
    assert report["replay"] == {
        "valid": 1,
        "total": 1,
        "final_product_matches": 1,
        "endpoint_stepwise_status_disagreements": 0,
    }
    corruptions = report["corruption_detection"]
    assert corruptions["clean_valid"] == 1
    assert corruptions["corrupted_total"] == corruptions["detected"] == 10
    assert corruptions["expected_issue_matched"] == 10
    assert all(
        value["expected_issue_matched"] == 1
        for value in corruptions["per_corruption"].values()
    )
    assert report["runtime"]["stepwise_replay_ns"]["count"] == 1
    assert report["limitations"]["first_failing_step_localization"].startswith(
        "Not scored"
    )

    output = tmp_path / "evidence.json"
    write_evidence_report(report, output)
    assert json.loads(output.read_text()) == report


def test_every_controlled_corruption_reaches_its_declared_issue_code():
    from synkit.Mechanism.evidence import load_evidence_cases

    mismatches = []
    for case in load_evidence_cases("Data/Mech"):
        for corruption in corrupt_record(case.record):
            if corruption.expected_issue_code not in corruption.observed_issue_codes():
                mismatches.append((case.case_id, corruption.corruption))

    assert not mismatches
