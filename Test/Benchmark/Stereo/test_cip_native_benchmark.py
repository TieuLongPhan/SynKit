"""Native CIP benchmark scoring and limitation-accounting tests."""

import json
from pathlib import Path

from rdkit import Chem

from Experiment.Stereo.Perception.cip_labels import (
    _category_scores,
    _record_result,
    _score_records,
)

ROOT = Path(__file__).resolve().parents[3]
FROZEN_REPORT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "cip_native_report.json"
)


def _record(
    expected: tuple[str, ...],
    *,
    identifier: str = "TEST001",
    units: str = "TH",
) -> dict[str, object]:
    return {
        "ID": identifier,
        "recommended_labels": expected,
        "stereo_units": units,
    }


def test_native_record_scores_an_independent_tetrahedral_label() -> None:
    molecule = Chem.MolFromSmiles("F[C@@](Cl)(Br)I")
    assert molecule is not None

    result = _record_result(_record(("2R",)), molecule)

    assert result["exact"]
    assert result["predicted"] == ["2R"]
    assert result["limitations"] == []
    assert result["primary_limitation"] is None


def test_lowercase_reference_is_a_rule_four_or_five_ranking_limit() -> None:
    molecule = Chem.MolFromSmiles("F[C@@](Cl)(Br)I")
    assert molecule is not None

    result = _record_result(_record(("2r",)), molecule)

    assert not result["exact"]
    assert result["predicted"] == ["2R"]
    assert result["limitations"] == ["ranking_defect"]
    assert result["primary_limitation"] == "ranking_defect"


def test_extended_unit_has_primary_unsupported_class_cause() -> None:
    molecule = Chem.MolFromSmiles("F[C@@](Cl)(Br)I")
    assert molecule is not None

    result = _record_result(
        _record(("2S",), units="TH3"),
        molecule,
    )

    assert not result["exact"]
    assert result["primary_limitation"] == "unsupported_class"
    assert "unsupported_class" in result["limitations"]


def test_absent_descriptor_is_missing_orientation_evidence() -> None:
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None

    result = _record_result(_record(("1R",)), molecule)

    assert result["predicted"] == []
    assert result["limitations"] == ["missing_orientation_evidence"]
    assert result["primary_limitation"] == "missing_orientation_evidence"


def test_micro_and_category_scores_use_exact_label_sets() -> None:
    results = (
        {
            "expected": ["1R"],
            "predicted": ["1R"],
            "exact": True,
            "stereo_units": ["TH"],
        },
        {
            "expected": ["2S", "3E"],
            "predicted": ["2R", "3E"],
            "exact": False,
            "stereo_units": ["TH", "CT"],
        },
    )

    score = _score_records(results)
    categories = _category_scores(results)

    assert score["exact_records"] == 1
    assert score["true_positive_labels"] == 2
    assert score["micro_label_recall"] == 2 / 3
    assert categories["CT"]["records"] == 1
    assert categories["TH"]["records"] == 2


def test_frozen_native_report_covers_all_300_records() -> None:
    report = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))

    assert report["dataset"] == {
        "audited_sha256": (
            "df178635c00b6c41fad820d2609fc4ff18403c63dec4e5c3e1756a6db5858059"
        ),
        "records": 300,
        "structures_vendored": False,
    }
    assert report["parse_failures"] == []
    assert report["schema"] == "synkit.cip-native-validation/2"
    assert report["overall"]["exact_records"] == 175
    assert report["overall"]["true_positive_labels"] == 902
    assert len(report["nonexact_records"]) == 125
    assert report["primary_limitation_counts"] == {
        "disputed_reference": 0,
        "label_projection_defect": 0,
        "missing_orientation_evidence": 32,
        "ranking_defect": 83,
        "unsupported_class": 10,
    }
    assert sum(report["primary_limitation_counts"].values()) == 125


def test_frozen_native_report_contains_no_benchmark_structures() -> None:
    report_text = FROZEN_REPORT.read_text(encoding="utf-8")
    report = json.loads(report_text)

    assert '"SMILES"' not in report_text
    assert report["comparators"]["rdkit"]["exact_records"] == 245
    assert report["comparators"]["rdkit"]["true_positive_labels"] == 1129
    assert report["by_stereo_unit"]["TH"]["records"] == 249
    assert report["by_stereo_unit"]["CT"]["records"] == 65
