"""Regression gates for the pre-benchmark canonicalization inventory."""

from __future__ import annotations

import csv

from Experiment.Stereo.Canonicalization.inventory import (
    build_inventory,
    write_inventory,
)


def test_inventory_preserves_source_specific_task_boundaries() -> None:
    inventory = build_inventory()
    summary = inventory["summary"]

    assert inventory["processing_status"] == (
        "no_permutations_or_certificates_computed"
    )
    assert summary["records"] == {
        "acs_stereomolgraph": 228,
        "cip_validation_suite": 298,
        "chiralfinder_rota": 650,
        "total": 1176,
    }

    acs = summary["acs_stereomolgraph"]
    assert acs["multiplicity"] == {
        "multiple": 164,
        "single": 64,
    }
    assert acs["configured_family_elements"] == {
        "planar_bond": 91,
        "tetrahedral": 919,
    }
    assert acs["configured_certificate_candidates"] == 228
    assert acs["excluded_graph_only_records"] == 30

    cip = summary["cip_validation_suite"]
    assert cip["unit_tag_record_counts_nonexclusive"] == {
        "AT": 7,
        "CT": 65,
        "CT4": 5,
        "HE": 2,
        "TH": 249,
        "TH3": 8,
        "TH5": 2,
    }
    assert cip["excluded_no_stereo_unit_records"] == 2
    assert cip["reference_rs_multiplicity"] == {
        "multiple": 153,
        "no_reference_rs": 58,
        "single": 87,
    }
    assert cip["attached_tetrahedral_multiplicity"] == {
        "multiple": 116,
        "none": 128,
        "single": 54,
    }

    rota = summary["chiralfinder_rota"]
    assert rota["locus_multiplicity"] == {
        "multiple": 40,
        "single": 610,
    }
    assert rota["reference_loci"] == 698
    assert rota["configured_certificate_candidates"] == 0
    assert rota["support_canonicalization_candidates"] == 650


def test_csv_projection_has_one_row_per_source_record(tmp_path) -> None:
    inventory = build_inventory()
    json_path = tmp_path / "inventory.json"
    csv_path = tmp_path / "inventory.csv"

    write_inventory(
        inventory,
        json_path=json_path,
        csv_path=csv_path,
    )

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1176
    assert {row["source"] for row in rows} == {
        "acs_stereomolgraph",
        "cip_validation_suite",
        "chiralfinder_rota",
    }
