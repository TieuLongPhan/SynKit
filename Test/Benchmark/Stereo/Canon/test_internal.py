"""Regression gates for the all-family canonicalization experiment."""

from __future__ import annotations

from Experiment.Stereo.Canonicalization.internal import (
    _EXPECTED_CONFIGURATION_COUNTS,
    _fixture_catalog,
    _local_arrangement_audit,
    benchmark_exact_canonicalization,
)
from Experiment.Stereo.acs_molecular_chirality import DATASET


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
    report = benchmark_exact_canonicalization(
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
    report = benchmark_exact_canonicalization(
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
