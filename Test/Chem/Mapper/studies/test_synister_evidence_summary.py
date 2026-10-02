import gzip
import json

import pytest

from scripts.summarize_synister_evidence import _payload_sha256, summarize
from scripts.verify_synister_evidence import audit_campaign


def _sign(payload, digest_field):
    payload[digest_field] = _payload_sha256(payload)
    return payload


def _campaign(tmp_path):
    campaign = tmp_path / "campaign"
    cases = campaign / "cases"
    cases.mkdir(parents=True)
    manifest = _sign(
        {
            "schema_version": 4,
            "kind": "synister_reference_blinded_global_shell_campaign",
            "rows": 1,
            "options": {"mode": "both"},
        },
        "manifest_sha256",
    )
    record = _sign(
        {
            "schema_version": 4,
            "kind": "synister_reference_blinded_global_shell_case",
            "campaign_manifest_sha256": manifest["manifest_sha256"],
            "source_line": 1,
            "reaction_id": "example:1",
            "status": "error",
        },
        "record_sha256",
    )
    (campaign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    case_path = cases / "line_1.json.gz"
    with gzip.open(case_path, "wt", encoding="ascii") as stream:
        json.dump(record, stream)
    return campaign, manifest, record, case_path


def _rewrite_case(path, record):
    with gzip.open(path, "wt", encoding="ascii") as stream:
        json.dump(record, stream)


def _complete_shell(*, structure_complete=True, reference_observed=True):
    return {
        "complete": True,
        "elapsed_seconds": 0.1,
        "labeled_solution_count": 2,
        "minimum_cost": 1.0,
        "reference_cd": 1.0,
        "symmetry_quotient_complete": True,
        "reaction_center": {
            "bond_union": [],
            "bond_intersection": [],
            "atom_union": [],
            "atom_intersection": [],
        },
        "structure": {
            "complete": structure_complete,
            "observed_its_class_count": 2,
            "observed_template_class_count": 2,
            "reference_its_class_observed": reference_observed,
        },
    }


def test_summary_accepts_valid_manifest_payload_and_binding_digests(tmp_path):
    campaign, _, _, _ = _campaign(tmp_path)

    result = summarize(campaign)

    assert result["case_records"] == 1
    assert result["modes"] == {
        "minimal": {"cases": 0},
        "reference_cd": {"cases": 0},
    }


def test_summary_rejects_manifest_payload_tampering(tmp_path):
    campaign, manifest, _, _ = _campaign(tmp_path)
    manifest["rows"] = 2
    (campaign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="campaign manifest digest mismatch"):
        summarize(campaign)


def test_summary_rejects_case_payload_tampering(tmp_path):
    campaign, _, record, case_path = _campaign(tmp_path)
    record["reaction_id"] = "tampered"
    _rewrite_case(case_path, record)

    with pytest.raises(ValueError, match="case payload .* digest mismatch"):
        summarize(campaign)


def test_summary_rejects_rehashed_case_bound_to_another_manifest(tmp_path):
    campaign, _, record, case_path = _campaign(tmp_path)
    record.pop("record_sha256")
    record["campaign_manifest_sha256"] = "0" * 64
    _sign(record, "record_sha256")
    _rewrite_case(case_path, record)

    with pytest.raises(ValueError, match="case manifest binding mismatch"):
        summarize(campaign)


def test_summary_excludes_incomplete_structure_and_null_reference_from_class_counts(
    tmp_path,
):
    campaign, manifest, record, case_path = _campaign(tmp_path)
    record.pop("record_sha256")
    record.pop("status")
    record.update(
        atom_count=4,
        shells={
            "minimal": _complete_shell(reference_observed=None),
            "reference_cd": _complete_shell(structure_complete=False),
        },
    )
    _sign(record, "record_sha256")
    _rewrite_case(case_path, record)

    result = summarize(campaign)

    minimum = result["modes"]["minimal"]
    reference = result["modes"]["reference_cd"]
    assert minimum["structure_complete"] == 1
    assert minimum["reference_its_class_observed"] == 0
    assert minimum["multiple_exact_its_classes"] == 1
    assert reference["cases"] == 1
    assert reference["structure_complete"] == 0
    assert reference["multiple_exact_its_classes"] == 0
    assert reference["alternative_its_application_complete_cases"] == 0


def test_audit_campaign_rejects_promoted_count_drift(tmp_path):
    campaign, manifest, record, case_path = _campaign(tmp_path)
    record.pop("record_sha256")
    record.pop("status")
    record.update(atom_count=4, shells={"minimal": _complete_shell()})
    manifest["options"] = {"mode": "minimal"}
    manifest.pop("manifest_sha256")
    _sign(manifest, "manifest_sha256")
    record["campaign_manifest_sha256"] = manifest["manifest_sha256"]
    _sign(record, "record_sha256")
    (campaign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_case(case_path, record)
    spec = {
        "name": "fixture",
        "path": campaign,
        "manifest_sha256": manifest["manifest_sha256"],
        "case_records": 1,
        "modes": {"minimal": {"cases": 1, "multiple_exact_its_classes": 1}},
    }

    audit = audit_campaign(spec)

    assert audit["modes"]["minimal"] == {
        "cases": 1,
        "multiple_exact_its_classes": 1,
    }
    spec["modes"]["minimal"]["cases"] = 2
    with pytest.raises(ValueError, match="fixture minimal: cases"):
        audit_campaign(spec)
