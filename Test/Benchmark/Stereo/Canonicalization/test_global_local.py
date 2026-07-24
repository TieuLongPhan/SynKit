"""Regression gates for selective global-by-local canonicalization."""

from __future__ import annotations

import json
from pathlib import Path

from Experiment.Stereo.Canonicalization.global_local import (
    _EXPECTED_CONFIGURATION_COUNTS,
    _fixture_catalog,
    _local_arrangement_audit,
    benchmark_global_local_canonicalization,
)
from Experiment.Stereo.Chirality.published import DATASET

FROZEN_REPORT = (
    Path(__file__).resolve().parents[4]
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Canonicalization"
    / "global_local_canonicalization_report.json"
)


def test_frozen_selective_global_local_matrix_covers_every_family_and_class() -> None:
    report = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    summary = report["summary"]

    assert report["task"] == "global_local"
    assert summary["families"] == summary["families_passed"] == 10
    assert summary["expected_configuration_classes"] == 67
    assert summary["exact_certificate_classes"] == 67
    assert summary["raw_local_arrangements_checked"] == 940
    assert summary["class_relabelings_checked"] == 134
    assert summary["representative_relabelings_checked"] == 528
    assert summary["timeouts"] == 0


def test_local_arrangement_audit_covers_all_67_configurations() -> None:
    records = {
        fixture.family: _local_arrangement_audit(fixture)
        for fixture in _fixture_catalog()
    }

    assert sum(record["raw_arrangements_checked"] for record in records.values()) == 940
    assert (
        sum(record["local_configuration_classes"] for record in records.values()) == 67
    )
    assert {
        family: record["local_configuration_classes"]
        for family, record in records.items()
    } == _EXPECTED_CONFIGURATION_COUNTS


def test_reduced_exact_audit_retains_certificate_and_mirror_gates() -> None:
    report = benchmark_global_local_canonicalization(
        DATASET,
        family_names=("tetrahedral",),
        class_relabelings=1,
        representative_samples=2,
        exhaustive_atom_limit=0,
        include_public=False,
    )

    assert report["passed"]
    assert report["totals"]["expected_configuration_classes"] == 2
    assert report["totals"]["exact_certificate_classes"] == 2
    assert report["totals"]["class_relabelings_checked"] == 2
    assert report["totals"]["representative_relabelings_checked"] == 2
    assert report["totals"]["synthetic_timeouts"] == 0
    assert report["families"][0]["expected_mirror_fixed_classes"] == 0
    assert report["families"][0]["mirror_fixed_classes"] == 0


def test_force_exhaustive_checks_every_class_under_every_permutation() -> None:
    report = benchmark_global_local_canonicalization(
        DATASET,
        family_names=("tetrahedral",),
        exhaustive_all_classes=True,
        include_public=False,
    )

    assert report["passed"]
    assert report["protocol"]["exhaustive_all_configuration_classes"]
    assert report["totals"]["class_relabelings_checked"] == 2 * 120
    assert report["totals"]["representative_relabelings_checked"] == 0
    assert {
        record["relabeling_protocol"] for record in report["families"][0]["classes"]
    } == {"exhaustive_n_factorial"}
