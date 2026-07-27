"""Tests for exhaustive configuration-free local canonicalization."""

import json
from pathlib import Path

from Experiment.Stereo.Canonicalization.configuration_free_local import (
    _carrier_result,
    _molecular_record,
)
from Experiment.Stereo.Canonicalization.global_local import _fixture_catalog

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "Experiment" / "Stereo" / "Data" / "Canonicalization"


def _tetrahedral_fixture():
    return next(
        fixture
        for fixture in _fixture_catalog()
        if fixture.family == "tetrahedral"
    )


def test_tetrahedral_enumerates_twenty_four_as_two_classes_of_twelve():
    fixture = _tetrahedral_fixture()
    result = _carrier_result(
        source="internal",
        record_id="tetrahedral",
        carrier_id="tetrahedral:local:0",
        carrier_status="fixture",
        graph=fixture.graph,
        seed=fixture.seed,
        timeout_seconds=10.0,
        internal_colors=True,
    )

    assert result["raw_local_permutations_expected"] == 24
    assert result["raw_local_permutations_checked"] == 24
    assert result["theoretical_configuration_classes"] == 2
    assert result["theoretical_class_multiplicities"] == [12, 12]
    assert result["canonical_classes_observed"] == 2
    assert result["formal_class_separation_passed"]
    assert result["passed"]


def test_supplied_tetrahedral_orientation_does_not_change_local_result():
    clockwise = _molecular_record(
        ("test", "clockwise", "F[C@](Cl)(Br)I", 10.0)
    )
    anticlockwise = _molecular_record(
        ("test", "anticlockwise", "F[C@@](Cl)(Br)I", 10.0)
    )

    left = clockwise["carriers"][0]
    right = anticlockwise["carriers"][0]
    assert clockwise["supplied_configurations_retained"] == 0
    assert anticlockwise["supplied_configurations_retained"] == 0
    assert left["theoretical_class_multiplicities"] == [12, 12]
    assert right["theoretical_class_multiplicities"] == [12, 12]
    assert [
        item["canonical_digests"] for item in left["theoretical_class_results"]
    ] == [
        item["canonical_digests"] for item in right["theoretical_class_results"]
    ]


def test_multiple_centres_are_benchmarked_independently():
    record = _molecular_record(
        ("test", "two-centres", "F[C@H](Cl)[C@@H](Br)I", 10.0)
    )

    assert record["potential_carriers_detected"] == 2
    assert len(record["carriers"]) == 2
    assert all(
        carrier["raw_local_permutations_expected"] == 24
        for carrier in record["carriers"]
    )
    assert sum(
        carrier["raw_local_permutations_expected"]
        for carrier in record["carriers"]
    ) == 48
    assert all(carrier["passed"] for carrier in record["carriers"])


def test_retained_local_matrix_is_complete():
    totals = {
        "records": 0,
        "carriers": 0,
        "representations": 0,
        "theoretical_classes": 0,
        "observed_classes": 0,
        "symmetry_quotients": 0,
    }
    for task in ("internal", "acs", "cip", "rota"):
        path = REPORT_ROOT / f"{task}_local_canonicalization_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report["summary"]
        assert report["schema"] == "synkit.configuration-free-local-canonicalization/1"
        assert summary["complete"]
        assert summary["carriers_failed"] == 0
        assert summary["parse_failures"] == 0
        assert summary["carriers_passed"] == summary["carriers"]
        assert (
            summary["raw_local_permutations_checked"]
            == summary["raw_local_permutations_expected"]
        )
        totals["records"] += summary["records_evaluated"]
        totals["carriers"] += summary["carriers"]
        totals["representations"] += summary["raw_local_permutations_checked"]
        totals["theoretical_classes"] += summary[
            "theoretical_configuration_classes"
        ]
        totals["observed_classes"] += summary["canonical_classes_observed"]
        totals["symmetry_quotients"] += summary["global_symmetry_quotients"]

    assert totals == {
        "records": 1218,
        "carriers": 3476,
        "representations": 66468,
        "theoretical_classes": 6999,
        "observed_classes": 5885,
        "symmetry_quotients": 1114,
    }
