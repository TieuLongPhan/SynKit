from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from synkit.Mechanism import MechanismRecord, MechanismReplayer
from synkit.Mechanism.benchmark import _audit_rule_reapplication

ROOT = Path(__file__).parents[2]


def test_reviewed_radical_manifest_replays_forward_reverse_and_double_reverse():
    manifest = json.loads(
        (ROOT / "Data/MechanismBench/radical_reviewed.json").read_text()
    )

    assert len(manifest["cases"]) == 80
    assert manifest["schema"] == "MechanismBench-radical-reviewed-v1"
    assert all(case["chemistry_reviewed"] for case in manifest["cases"])
    assert all(
        case["provenance"]["chemistry_review"]["automated_replay"]["status"] == "VALID"
        for case in manifest["cases"]
    )
    assert manifest["release_issues"] == [
        "PARTITION_COUNT:polar:0/80",
        "PARTITION_COUNT:stereo:0/80",
    ]
    macro_counts = Counter()
    for case in manifest["cases"]:
        record = MechanismRecord.from_dict(case["record"])
        macro_counts[record.steps[0].groups[0].macro] += 1
        forward = MechanismReplayer().replay(record)
        reverse = MechanismReplayer().replay(record.reversed())

        assert forward.certificate.status == "VALID", case["case_id"]
        assert reverse.certificate.status == "VALID", case["case_id"]
        assert record.reversed().reversed() == record

    assert set(macro_counts) == {
        "HOMOLYSIS",
        "RECOMBINATION",
        "RADICAL_ADDITION",
        "BETA_SCISSION",
        "H_ABSTRACTION",
        "RADICAL_RESONANCE",
    }
    assert max(macro_counts.values()) - min(macro_counts.values()) == 1


def test_each_radical_macro_has_an_executable_grammar_corruption():
    manifest = json.loads(
        (ROOT / "Data/MechanismBench/radical_reviewed.json").read_text()
    )
    representatives = {}
    for case in manifest["cases"]:
        macro = case["record"]["steps"][0]["groups"][0]["macro"]
        representatives.setdefault(macro, case)

    assert len(representatives) == 6
    for macro, case in representatives.items():
        payload = deepcopy(case["record"])
        payload["steps"][0]["groups"][0]["moves"].pop()
        corrupted = MechanismRecord.from_dict(payload)
        result = MechanismReplayer().replay(corrupted)

        assert result.certificate.status == "INVALID", macro
        assert "UNBALANCED_EVENT_GROUP" in {
            issue.code for issue in result.certificate.issues
        }, macro


def test_mechanismbench_json_files_are_executable_case_data_only():
    json_files = sorted(
        path.name for path in (ROOT / "Data/MechanismBench").glob("*.json")
    )

    assert json_files == [
        "radical_reviewed.json",
        "small_rewrite_conformance.json",
        "stereo.json",
    ]
    for filename in json_files:
        payload = json.loads((ROOT / "Data/MechanismBench" / filename).read_text())
        assert payload["cases"]


def test_rule_reapplication_audit_covers_every_unmapped_reactant():
    manifest = json.loads(
        (ROOT / "Data/MechanismBench/radical_reviewed.json").read_text()
    )
    audits = [
        case["provenance"]["chemistry_review"]["rule_reapplication"]
        for case in manifest["cases"]
    ]

    assert len(audits) == 80
    assert sum(audit["status"] == "PASS" for audit in audits) == 80
    assert sum(audit["status"] == "FAIL" for audit in audits) == 0
    assert all(audit["unmapped_reactants"] for audit in audits)


def test_four_review_corrections_preserve_original_and_replay():
    manifest = json.loads(
        (ROOT / "Data/MechanismBench/radical_reviewed.json").read_text()
    )
    corrected = [
        case
        for case in manifest["cases"]
        if case["provenance"]["chemistry_review"]["corrected"]
    ]

    assert {case["case_id"] for case in corrected} == {
        "radical-00004",
        "radical-00026",
        "radical-00053",
        "radical-00210",
    }
    for case in corrected:
        record = MechanismRecord.from_dict(case["record"])
        assert "original_mapped_reaction" in record.provenance
        assert MechanismReplayer().replay(record).certificate.status == "VALID"


def test_allene_regiochemistry_is_retained_as_advisory():
    manifest = json.loads(
        (ROOT / "Data/MechanismBench/radical_reviewed.json").read_text()
    )
    case = next(
        case for case in manifest["cases"] if case["case_id"] == "radical-00034"
    )

    assert "less likely" in case["provenance"]["chemistry_review"]["advisory"]
    assert case["chemistry_reviewed"]


@pytest.mark.parametrize("case_id", ["radical-00002", "radical-00054"])
def test_rule_reapplication_uses_aam_alignment_and_radical_node_state(case_id):
    manifest = json.loads(
        (ROOT / "Data/MechanismBench/radical_reviewed.json").read_text()
    )
    case = next(case for case in manifest["cases"] if case["case_id"] == case_id)

    audit = _audit_rule_reapplication(MechanismRecord.from_dict(case["record"]))

    assert audit["status"] == "PASS", audit
