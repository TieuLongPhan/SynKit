"""Integrity and task-boundary tests for the stereo benchmark registry."""

import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Test.Benchmark.Stereo.benchmark_stereo_datasets import (  # noqa: E402
    CIP_METADATA,
    ROTA,
    STEREO_ROOT,
    load_rota,
)
from Test.Benchmark.Stereo.benchmark_stereo_backends import (  # noqa: E402
    _classify_rdkit_smiles,
)
from Test.Chem.Molecule.benchmark_molecular_chirality import (  # noqa: E402
    load_dataset,
)

MANIFEST = STEREO_ROOT / "manifest.json"
REPORT = STEREO_ROOT / "benchmark_report.json"
BACKEND_REPORT = STEREO_ROOT / "backend_comparison_report.json"


def test_registry_preserves_task_and_license_boundaries() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    datasets = {entry["id"]: entry for entry in manifest["datasets"]}

    acs = datasets["acs_stereomolgraph_molecular_chirality"]
    assert acs["records"] == 258
    assert acs["license"] == "CC BY-NC 4.0"
    assert acs["protocols"] == {
        "stereo_stripped_four_state": True,
        "supplied_stereo_binary": True,
    }

    rota = datasets["chiralfinder_rota"]
    assert rota["records"] == 650
    assert rota["license"] == "MIT"
    assert not rota["protocols"]["supplied_stereo_binary"]
    assert all(rota["diagnostics"].values())

    cip = datasets["cip_validation_suite"]
    assert cip["records"] == 300
    assert cip["license"] is None
    assert not cip["vendored"]
    assert all(cip["diagnostics"].values())
    assert not (CIP_METADATA.parent / "compounds.smi").exists()


def test_rota_fixture_is_exact_and_registered_for_axial_loci() -> None:
    rows = load_rota(ROTA)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = report["datasets"]["chiralfinder_rota"]

    assert len(rows) == audit["records"] == 650
    assert audit["rdkit_parse_failures"] == []
    assert audit["axis_count_distribution"] == {"1": 610, "2": 32, "3": 8}
    assert sum(audit["chiral_type_counts"].values()) == 650
    assert audit["normal"]["executed"]
    assert audit["stereo_removed"]["executed"]
    assert audit["normal"]["reference_accuracy"] is None
    assert audit["stereo_removed"]["reference_accuracy"] is None


def test_frozen_whole_molecule_protocols_report_separate_conclusions() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    acs = report["datasets"]["acs_stereomolgraph_molecular_chirality"]
    normal = acs["normal"]
    stripped = acs["stereo_removed"]

    assert normal["synkit"]["correct"] == 258
    assert normal["published_stereomolgraph"]["correct"] == 258
    assert normal["published_rdkit_smiles"]["correct"] == 235
    assert stripped["row_outcomes"] == {
        "configuration_dependent": 66,
        "necessarily_achiral": 65,
        "necessarily_chiral": 123,
        "unsupported_or_incomplete": 4,
    }
    assert stripped["definitive_rows"] == 254
    assert stripped["manual_label_in_observed_population"] == 258
    assert [case["ids"] for case in stripped["incomplete_cases"]] == [
        ["VS226"],
        ["VS265"],
        ["VS266"],
        ["VS268"],
    ]


def test_frozen_local_label_diagnostics_cover_both_settings() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rota = report["datasets"]["chiralfinder_rota"]
    cip = report["datasets"]["cip_validation_suite"]

    assert rota["normal"]["evaluated_records"] == 650
    assert rota["normal"]["predictions"] == {"Achiral": 542, "Chiral": 108}
    assert rota["stereo_removed"]["evaluated_records"] == 649
    assert rota["stereo_removed"]["errors"] == [
        {"error_type": "case_timeout", "id": "RotA-0293"}
    ]
    assert rota["stereo_removed"]["row_outcomes"] == {
        "configuration_dependent": 43,
        "necessarily_achiral": 538,
        "necessarily_chiral": 67,
        "unsupported_or_incomplete": 1,
    }

    assert cip["benchmark_run"]
    assert cip["normal"]["evaluated_records"] == 300
    assert cip["normal"]["predictions"] == {"Achiral": 125, "Chiral": 175}
    assert cip["stereo_removed"]["evaluated_records"] == 300
    assert cip["stereo_removed"]["errors"] == []
    assert cip["stereo_removed"]["row_outcomes"] == {
        "configuration_dependent": 68,
        "necessarily_achiral": 94,
        "necessarily_chiral": 134,
        "unsupported_or_incomplete": 4,
    }
    assert cip["stereo_unit_counts"] == {
        "AT": 7,
        "CT": 65,
        "CT4": 5,
        "HE": 2,
        "TH": 249,
        "TH3": 8,
        "TH5": 2,
    }


def test_frozen_live_backends_cover_all_datasets_and_settings() -> None:
    report = json.loads(BACKEND_REPORT.read_text(encoding="utf-8"))
    datasets = report["datasets"]
    expected_records = {
        "acs_stereomolgraph_molecular_chirality": 258,
        "chiralfinder_rota": 650,
        "cip_validation_suite": 300,
    }
    for name, records in expected_records.items():
        dataset = datasets[name]
        assert dataset["records"] == records
        assert dataset["rdkit_parse_failures"] == []
        for setting in ("supplied_stereo", "stereo_removed"):
            backends = dataset[setting]["backends"]
            assert set(backends) == {
                "rdkit_smiles",
                "stereomolgraph",
                "synkit",
            }
            assert all(
                item["evaluated_records"] == records for item in backends.values()
            )
            assert all(item["errors"] == [] for item in backends.values())

    acs = datasets["acs_stereomolgraph_molecular_chirality"]
    supplied = acs["supplied_stereo"]["backends"]
    assert supplied["synkit"]["manual_label_agreement"]["correct"] == 258
    assert supplied["rdkit_smiles"]["manual_label_agreement"]["correct"] == 235
    assert supplied["stereomolgraph"]["manual_label_agreement"]["correct"] == 254

    for dataset in datasets.values():
        removed = dataset["stereo_removed"]["backends"]
        assert removed["rdkit_smiles"]["predictions"] == {"Achiral": dataset["records"]}

    rota_native = datasets["chiralfinder_rota"]["native_task_accuracy"]
    assert not any(item["applicable"] for item in rota_native.values())

    cip_native = datasets["cip_validation_suite"]["native_task_accuracy"]
    assert cip_native["rdkit"]["exact_records"] == 245
    assert cip_native["rdkit"]["exact_record_accuracy"] == 245 / 300
    assert cip_native["rdkit"]["true_positive_labels"] == 1129
    assert cip_native["rdkit"]["expected_labels"] == 1252
    assert not cip_native["synkit"]["applicable"]
    assert not cip_native["stereomolgraph"]["applicable"]

    rota = datasets["chiralfinder_rota"]
    assert rota["binary_reference"]["labels"] == {"Chiral": 650}
    rota_supplied = rota["supplied_stereo"]["backends"]
    assert rota_supplied["synkit"]["manual_label_agreement"]["correct"] == 488
    assert rota_supplied["rdkit_smiles"]["manual_label_agreement"]["correct"] == 87
    assert rota_supplied["stereomolgraph"]["manual_label_agreement"]["correct"] == 109

    cip = datasets["cip_validation_suite"]
    assert cip["binary_reference"]["labels"] == {
        "Achiral": 106,
        "Chiral": 194,
    }
    cip_supplied = cip["supplied_stereo"]["backends"]
    assert cip_supplied["synkit"]["manual_label_agreement"]["correct"] == 298
    assert cip_supplied["rdkit_smiles"]["manual_label_agreement"]["correct"] == 264
    assert cip_supplied["stereomolgraph"]["manual_label_agreement"]["correct"] == 271
    assert cip_supplied["synkit"]["manual_label_agreement"]["disagreement_ids"] == [
        "VS010",
        "VS011",
    ]


def test_live_rdkit_adapter_reproduces_published_acs_column() -> None:
    rows = load_dataset()
    predictions = [
        _classify_rdkit_smiles(Chem.MolFromSmiles(row["Input SMILES"])) for row in rows
    ]
    assert predictions == [row["RDKit SMILES"] for row in rows]
    assert (
        sum(prediction == row["manual"] for prediction, row in zip(predictions, rows))
        == 235
    )


def test_cip_binary_reference_is_complete_and_provenance_tiered() -> None:
    report = json.loads(BACKEND_REPORT.read_text(encoding="utf-8"))
    dataset = report["datasets"]["cip_validation_suite"]
    metadata = dataset["binary_reference"]

    assert dataset["records"] == 300
    assert metadata["labels"] == {"Achiral": 106, "Chiral": 194}
    assert metadata["published_acs_manual_records"] == 258
    assert metadata["independent_atrop_notebook_records"] == 7
    assert metadata["expert_curated_records"] == 35
