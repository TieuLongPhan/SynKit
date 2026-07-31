"""Integrity and task-boundary tests for the stereo benchmark registry."""

import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    CIP_METADATA,
    ROTA,
    STEREO_ROOT,
    load_rota,
)
from Experiment.Stereo.Diagnostics.backend_comparison import (  # noqa: E402
    _classify_rdkit_smiles,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    load_dataset,
)
from Experiment.Stereo.Chirality.exact_mirror import (  # noqa: E402
    _parse_supplied_configured_smiles,
)

MANIFEST = STEREO_ROOT / "manifest.json"
REPORT = STEREO_ROOT / "Diagnostics" / "benchmark_report.json"
BACKEND_REPORT = STEREO_ROOT / "Diagnostics" / "backend_comparison_report.json"
EXACT_MIRROR_REPORT = STEREO_ROOT / "Chirality" / "exact_acs_mirror_report.json"
BENCHMARK_SCRIPT = ROOT / "Experiment" / "Stereo" / "benchmark.sh"


def test_single_stereo_benchmark_script_has_requested_raw_tail() -> None:
    scripts = sorted((ROOT / "Experiment" / "Stereo").rglob("*.sh"))
    text = BENCHMARK_SCRIPT.read_text(encoding="utf-8")
    raw_tail = text.split("# Requested raw stress products", maxsplit=1)[1]

    assert scripts == [BENCHMARK_SCRIPT]
    assert "SYNKIT_BENCHMARK_JOBS:-16" in text
    assert "${RUNNER}" not in text
    assert raw_tail.count("--enumeration-mode raw --allow-expensive-raw") == 2
    assert 'GLOBAL_ABC_RUNNER}" B --cip-path' in raw_tail
    assert 'GLOBAL_ABC_RUNNER}" C --case-id VS146' in raw_tail
    assert raw_tail.index('GLOBAL_ABC_RUNNER}" B --cip-path') < raw_tail.index(
        'GLOBAL_ABC_RUNNER}" C --case-id VS146'
    )


def test_registry_preserves_task_and_license_boundaries() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    datasets = {entry["id"]: entry for entry in manifest["datasets"]}

    acs = datasets["acs_stereomolgraph_molecular_chirality"]
    assert acs["records"] == 258
    assert acs["license"] == "CC BY-NC 4.0"
    assert acs["protocols"]["stereo_stripped_four_state"]
    assert acs["protocols"]["supplied_stereo_binary"]
    assert acs["protocols"]["exact_configured_mirror_audit"]
    assert acs["protocols"]["fixed_graph_local_permutation_audit"]
    assert acs["exact_mirror_report"].endswith("Chirality/exact_acs_mirror_report.json")
    assert acs["canonicalization_report"].endswith(
        "Canonicalization/acs_local_canonicalization_report.json"
    )

    rota = datasets["chiralfinder_rota"]
    assert rota["records"] == 650
    assert rota["license"] == "MIT"
    assert rota["protocols"]["typed_axis_locus_detection"]
    assert not rota["protocols"]["supplied_stereo_binary"]
    assert rota["frozen_report"].endswith("rota_locus_report.json")
    assert all(rota["diagnostics"].values())

    cip = datasets["cip_validation_suite"]
    assert cip["records"] == 300
    assert cip["license"] is None
    assert not cip["vendored"]
    assert cip["protocols"]["native_local_cip_label_assignment"]
    assert cip["protocols"]["configuration_neutral_stereo_element_audit"]
    assert cip["protocols"]["fixed_graph_local_permutation_audit"]
    assert cip["frozen_report"].endswith("cip_native_report.json")
    assert cip["stereo_element_report"].endswith("stereo_element_report.json")
    assert cip["canonicalization_report"].endswith(
        "Canonicalization/cip_local_canonicalization_report.json"
    )
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
        "configuration_dependent": 67,
        "necessarily_achiral": 66,
        "necessarily_chiral": 125,
    }
    assert stripped["definitive_rows"] == 258
    assert stripped["manual_label_in_observed_population"] == 258
    assert stripped["incomplete_cases"] == []


def test_exact_mirror_audit_preserves_supplied_configuration_only() -> None:
    report = json.loads(EXACT_MIRROR_REPORT.read_text(encoding="utf-8"))
    records = {record["id"]: record for record in report["records"]}

    assert report["schema"] == "synkit.exact-acs-mirror-benchmark/3"
    assert report["identity_profile"] == "chemical"
    assert report["outcomes"] == {
        "achiral": 94,
        "chiral": 164,
    }
    assert report["definitive_records"] == 258
    assert report["correct_definitive"] == 256
    assert report["definitive_with_unresolved_loci"] == 0
    assert report["disagreement_ids"] == ["VS170", "VS300"]
    assert records["VS170"]["status"] == "chiral"
    assert records["VS215"]["status"] == "achiral"
    assert records["VS216"]["status"] == "achiral"
    assert records["VS188"]["status"] == "achiral"
    assert records["VS188"]["incomplete_loci"] == []
    assert records["VS042"]["status"] == "achiral"
    assert records["VS044"]["status"] == "achiral"
    assert records["VS298"]["status"] == "chiral"
    assert records["VS298"]["descriptor_count"] == 3
    assert records["VS298"]["incomplete_loci"] == []


def test_exact_mirror_parser_retains_supported_source_declared_cage_center() -> None:
    smiles = "Cl[C@H]1C[C@]2(C1)C[C@H](C2)Cl"

    molecule = _parse_supplied_configured_smiles(smiles)
    assert molecule is not None
    centers = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetChiralTag()
        in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
    }

    assert centers == {1, 3, 6}


def test_frozen_local_label_diagnostics_cover_both_settings() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rota = report["datasets"]["chiralfinder_rota"]
    cip = report["datasets"]["cip_validation_suite"]

    assert rota["normal"]["evaluated_records"] == 650
    assert rota["normal"]["predictions"] == {"Achiral": 542, "Chiral": 108}
    assert rota["stereo_removed"]["evaluated_records"] == 650
    assert rota["stereo_removed"]["errors"] == []
    assert rota["stereo_removed"]["row_outcomes"] == {
        "configuration_dependent": 36,
        "necessarily_achiral": 144,
        "necessarily_chiral": 44,
        "unsupported_or_incomplete": 426,
    }

    assert cip["benchmark_run"]
    assert cip["normal"]["evaluated_records"] == 300
    assert cip["normal"]["predictions"] == {"Achiral": 119, "Chiral": 181}
    assert cip["stereo_removed"]["evaluated_records"] == 300
    assert cip["stereo_removed"]["errors"] == []
    assert cip["stereo_removed"]["row_outcomes"] == {
        "configuration_dependent": 68,
        "necessarily_achiral": 76,
        "necessarily_chiral": 139,
        "unsupported_or_incomplete": 17,
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
    stereomolgraph_correct = supplied["stereomolgraph"]["manual_label_agreement"][
        "correct"
    ]
    assert stereomolgraph_correct == 254

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
    assert cip_native["synkit"]["applicable"]
    assert cip_native["synkit"]["exact_records"] == 175
    assert cip_native["synkit"]["true_positive_labels"] == 902
    assert cip_native["synkit"]["expected_labels"] == 1252
    assert cip_native["synkit"]["primary_limitation_counts"] == {
        "disputed_reference": 0,
        "label_projection_defect": 0,
        "missing_orientation_evidence": 32,
        "ranking_defect": 83,
        "unsupported_class": 10,
    }
    assert not cip_native["stereomolgraph"]["applicable"]

    claim_status = report["claim_status"]
    assert claim_status["current_authority"] == (
        "ACS supplied-stereo global binary and independently assigned CIP "
        "native local labels"
    )
    assert claim_status["status"] == ("native_cip_validation_with_typed_limitations")
    assert claim_status["retracted_claims"] == [
        "RotA 488/650 and 477/650 derived binary scoring",
        "CIP 298/300 and 263/300 derived binary scoring",
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


def test_provisional_cip_binary_reference_is_explicitly_retracted() -> None:
    report = json.loads(BACKEND_REPORT.read_text(encoding="utf-8"))
    assert (
        "CIP 298/300 and 263/300 derived binary scoring"
        in report["claim_status"]["retracted_claims"]
    )
    assert report["datasets"]["cip_validation_suite"]["records"] == 300
