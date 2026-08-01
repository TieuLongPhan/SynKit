"""Reproducibility gates for native rule-composition and MTG evidence."""

from __future__ import annotations

import json

from Experiment.MTG.validation import validation_report


def test_report_is_passing_bounded_and_claim_typed() -> None:
    report = validation_report(iterations=2)

    assert report["schema"] == "synkit.mtg-validation/1"
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert report["observed"]["raw_overlaps"] == 7
    assert report["observed"]["accepted_witnesses"] == 7
    assert report["observed"]["exact_classes"] == 3
    assert report["observed"]["matrix"] == ((1, 1, 1), (1, 1, 1))
    assert set(report["stage_timings"]) == {
        "rule_construction",
        "extended_match_matrix",
        "overlap_enumeration",
        "composite_construction_family",
        "exact_quotient",
        "certificate_replay_family",
        "process_construction",
        "mtg_derivation",
    }
    assert all(
        item["iterations"] == 2 for item in report["stage_timings"].values()
    )
    assert all(
        not item["formal_proof"] for item in report["case_studies"].values()
    )


def test_case_studies_retain_material_ambiguity_and_curated_scope() -> None:
    report = validation_report(iterations=1)
    cases = report["case_studies"]

    assert cases["aldol"]["step_counts"] == [6, 4, 4]
    assert cases["multistep_synthesis"]["step_count"] == 9
    assert cases["glycolysis_ga3p"]["alternative_count"] == 2
    assert set(cases["glycolysis_ga3p"]["consumed_occurrences"].values()) == {
        "ga3p-direct",
        "ga3p-from-tpi",
    }
    assert cases["native_multistep_rule"]["extended_match_matrix"] == ((2, 1),)


def test_report_is_json_serializable_and_has_stable_non_timing_evidence() -> None:
    first = validation_report(iterations=1)
    second = validation_report(iterations=1)

    json.dumps(first, sort_keys=True)
    assert first["input_sha256"] == second["input_sha256"]
    assert first["checks"] == second["checks"]
    assert first["observed"]["raw_overlaps"] == second["observed"]["raw_overlaps"]
    assert first["observed"]["class_witness_counts"] == [1, 2, 4]
