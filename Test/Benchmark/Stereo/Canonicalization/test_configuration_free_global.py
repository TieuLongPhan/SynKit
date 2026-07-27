"""Tests for the mixed-family configuration-free global benchmark."""

import json
from math import prod
from pathlib import Path

from rdkit import Chem

from Experiment.Stereo.Canonicalization.configuration_free_global import (
    CASES_BY_BUDGET,
    _carrier_inventory,
    benchmark_budget,
)
from Experiment.Stereo.Chirality.published import load_dataset

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Canonicalization"
    / "Global"
)


def test_abc_registry_contains_only_mixed_cases_in_raw_budget_order():
    limits = {
        "A": lambda value: value <= 1_000,
        "B": lambda value: 1_000 < value <= 1_000_000,
        "C": lambda value: value > 1_000_000,
    }
    for budget, cases in CASES_BY_BUDGET.items():
        assert cases
        assert all(len(case.composition) >= 2 for case in cases)
        assert all(limits[budget](case.raw_product) for case in cases)
        assert [case.raw_product for case in cases] == sorted(
            case.raw_product for case in cases
        )
        assert all(case.formal_assignments <= 4096 for case in cases)


def test_vs191_factorizes_to_128_joint_assignments():
    row = next(row for row in load_dataset() if row["ID"] == "VS191")
    molecule = Chem.MolFromSmiles(row["Input SMILES"])
    assert molecule is not None

    _graph, carriers, _inventory = _carrier_inventory(molecule)

    assert len(carriers) == 7
    assert prod(carrier["raw_local_permutations"] for carrier in carriers) == (
        6_291_456
    )
    assert prod(
        carrier["formal_configuration_classes"] for carrier in carriers
    ) == 128
    assert sum(
        carrier["configuration_dependent_support"] for carrier in carriers
    ) == 1


def test_a_case_canonicalizes_all_carriers_jointly():
    report = benchmark_budget(
        "A",
        case_ids=("VS032",),
        jobs=1,
        timeout_seconds=10.0,
    )
    record = report["records"][0]

    assert record["carrier_composition"] == "planar_bond:1;tetrahedral:1"
    assert record["full_raw_representation_product"] == 192
    assert record["joint_formal_assignments_expected"] == 4
    assert record["enumerated_assignments_expected"] == 4
    assert record["enumerated_assignments_completed"] == 4
    assert record["observed_global_canonical_classes"] == 4
    assert record["passed"]


def test_raw_and_formal_modes_produce_the_same_global_quotient(capsys):
    formal = benchmark_budget(
        "A",
        enumeration_mode="formal",
        case_ids=("VS032",),
        jobs=1,
        timeout_seconds=10.0,
    )["records"][0]
    raw = benchmark_budget(
        "A",
        enumeration_mode="raw",
        case_ids=("VS032",),
        jobs=1,
        timeout_seconds=10.0,
        progress_interval_seconds=10.0,
    )["records"][0]
    progress_output = capsys.readouterr().out

    assert raw["enumerated_assignments_expected"] == 192
    assert raw["enumerated_assignments_completed"] == 192
    assert raw["formal_assignment_tuples_covered"] == 4
    assert raw["formal_assignment_tuples_invariant"] == 4
    assert raw["representation_invariance_passed"]
    assert raw["raw_assignments_per_global_class"] == [48, 48, 48, 48]
    assert {
        result["canonical_digest"] for result in formal["global_classes"]
    } == {
        result["canonical_digest"] for result in raw["global_classes"]
    }
    assert "[raw A acs:VS032] starting 192 assignments" in progress_output
    assert "192/192 (100.00%)" in progress_output
    assert "formal=4/4" in progress_output
    assert "ETA=0.0s" in progress_output
    assert raw["passed"]


def test_retained_abc_matrix_is_complete_without_raw_b_or_c():
    paths = {
        ("A", "formal"): (
            REPORT_ROOT / "A" / "global_formal_canonicalization_report.json"
        ),
        ("A", "raw"): (
            REPORT_ROOT / "A" / "global_raw_canonicalization_report.json"
        ),
        ("B", "formal"): (
            REPORT_ROOT / "B" / "global_formal_canonicalization_report.json"
        ),
        ("C", "formal"): (
            REPORT_ROOT / "C" / "global_formal_canonicalization_report.json"
        ),
    }
    reports = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    expected = {
        ("A", "formal"): (10, 48, 38),
        ("A", "raw"): (10, 2048, 38),
        ("B", "formal"): (11, 192, 95),
        ("C", "formal"): (9, 1920, 752),
    }
    for key, report in reports.items():
        cases, assignments, classes = expected[key]
        summary = report["summary"]
        assert report["budget"] == key[0]
        assert report["enumeration_mode"] == key[1]
        assert summary["complete"]
        assert summary["cases_selected"] == summary["cases_passed"] == cases
        assert summary["cases_failed"] == 0
        assert summary["timeouts"] == 0
        assert (
            summary["enumerated_assignments_completed"]
            == summary["enumerated_assignments_expected"]
            == assignments
        )
        assert summary["observed_global_canonical_classes"] == classes

    assert (
        reports[("A", "formal")]["summary"]["observed_global_canonical_classes"]
        == reports[("A", "raw")]["summary"]["observed_global_canonical_classes"]
    )
    assert not (
        REPORT_ROOT / "B" / "global_raw_canonicalization_report.json"
    ).exists()
    assert not (
        REPORT_ROOT / "C" / "global_raw_canonicalization_report.json"
    ).exists()
