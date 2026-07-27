"""Configuration-free RotA local canonicalization gates."""

from Experiment.Stereo.Canonicalization.rota_local import (
    benchmark_rota_local_canonicalization,
)


def test_rota_axis_raw_orderings_quotient_without_reference_labels() -> None:
    report = benchmark_rota_local_canonicalization(
        record_ids=("RotA-0002",),
        jobs=1,
        timeout_seconds=10.0,
    )

    summary = report["summary"]
    assert report["dataset"]["source_annotations_used"] is False
    assert summary["detected_axis_carriers"] == 1
    assert summary["raw_local_orderings_expected"] == 8
    assert summary["raw_local_orderings_checked"] == 8
    assert summary["canonicalizations_completed"] == 8
    assert summary["complete"] is True
    assert summary["carriers_passed"] == 1
    assert summary["carriers_failed"] == 0

    carrier = report["records"][0]["carriers"][0]
    assert carrier["formal_configuration_classes"] == 2
    assert carrier["canonical_classes_observed"] in {1, 2}
    assert all(
        result["collapsed_within_class"] for result in carrier["configuration_results"]
    )
