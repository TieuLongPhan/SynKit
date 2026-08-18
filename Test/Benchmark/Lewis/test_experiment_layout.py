"""Architecture gates for executable Lewis benchmark workflows."""

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis import common  # noqa: E402
from Experiment.Lewis.mech_path import audit as mech_audit  # noqa: E402
from Experiment.Lewis.partial_expand import benchmark as partial_benchmark  # noqa: E402
from Experiment.Lewis.partial_expand import plot as partial_plot  # noqa: E402
from Experiment.Lewis.partial_expand import repeat_external, repeat_synkit  # noqa: E402
from Experiment.Lewis.rule_replay import benchmark as replay_benchmark  # noqa: E402
from Experiment.Lewis.rule_replay import plot as replay_plot  # noqa: E402

EXPERIMENT_ROOT = ROOT / "Experiment" / "Lewis"


def test_lewis_experiments_are_split_by_workflow() -> None:
    assert (EXPERIMENT_ROOT / "partial_expand" / "benchmark.py").is_file()
    assert (EXPERIMENT_ROOT / "rule_replay" / "benchmark.py").is_file()
    assert (EXPERIMENT_ROOT / "mech_path" / "evidence.py").is_file()
    assert (EXPERIMENT_ROOT / "mech_path" / "audit.py").is_file()
    assert (EXPERIMENT_ROOT / "run_experiments.sh").is_file()
    assert (EXPERIMENT_ROOT / "partial_expand" / "run_runtime_metadata.sh").is_file()
    assert (EXPERIMENT_ROOT / "hydrogen_expand" / "run_comparison.sh").is_file()


def test_data_are_collocated_without_python_modules() -> None:
    data_directories = (
        EXPERIMENT_ROOT / "partial_expand" / "Data",
        EXPERIMENT_ROOT / "rule_replay" / "Data",
        EXPERIMENT_ROOT / "mech_path" / "Data",
    )
    assert all(path.is_dir() for path in data_directories)
    assert not any(any(path.rglob("*.py")) for path in data_directories)
    assert not (ROOT / "Data" / "Benchmark" / "Lewis").exists()


def test_partial_expansion_paths_target_its_evidence_directory() -> None:
    expected = EXPERIMENT_ROOT / "partial_expand" / "Data"
    assert partial_benchmark.RESULTS_ROOT == expected
    assert repeat_synkit.RESULTS_ROOT == expected
    assert repeat_external.RESULTS_ROOT == expected
    assert partial_plot.RESULTS == expected


def test_rule_replay_paths_target_its_evidence_directory() -> None:
    expected = EXPERIMENT_ROOT / "rule_replay" / "Data"
    assert replay_benchmark.RESULTS_ROOT == expected
    assert replay_plot.DATA_ROOT == expected


def test_shared_dataset_paths_are_collocated_with_lewis_experiments() -> None:
    assert common.POLAR_DATASET == EXPERIMENT_ROOT / "Data" / "benchmark.json.gz"
    assert common.RADICAL_DATASET == EXPERIMENT_ROOT / "Data" / "all.csv"
    assert mech_audit.POLAR_DATASET == (
        EXPERIMENT_ROOT / "Data" / "combinatorial_all.csv"
    )
    assert mech_audit.RADICAL_DATASET == EXPERIMENT_ROOT / "Data" / "all.csv"


def test_reconstruction_audits_retain_ids_only() -> None:
    audit_root = EXPERIMENT_ROOT / "mech_path" / "Data" / "reconstruction_audit"
    expected_counts = {"polar": 0, "radical": 1}
    for name, expected_count in expected_counts.items():
        rows = (audit_root / f"{name}-failures.csv").read_text().splitlines()
        assert rows[0] == "source_row"
        assert len(rows[1:]) == expected_count
        assert all(row.isdigit() for row in rows[1:])


def test_radical_unresolved_registry_matches_retained_ids() -> None:
    path = (
        EXPERIMENT_ROOT
        / "mech_path"
        / "Data"
        / "reconstruction_audit"
        / "radical-failures.csv"
    )
    retained = {int(row) for row in path.read_text().splitlines()[1:]}
    assert retained == mech_audit.RADICAL_UNRESOLVED_IDS


def test_radical_arrow_review_has_ids_and_no_reaction_column() -> None:
    path = (
        EXPERIMENT_ROOT
        / "mech_path"
        / "Data"
        / "reconstruction_audit"
        / "radical-arrow-review.csv"
    )
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    assert rows[0] == [
        "source_row",
        "recorded_arrow",
        "reviewed_arrow",
        "outcome",
    ]
    assert {int(row[0]) for row in rows[1:]} == mech_audit.RADICAL_REVIEW_IDS
    assert all(len(row) == 4 for row in rows)
