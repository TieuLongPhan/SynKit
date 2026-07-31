"""Test policy for optional generated stereo benchmark reports."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "Experiment" / "Stereo" / "Data"
GENERATED_REPORTS = (
    DATA / "Canonicalization" / "internal_local_canonicalization_report.json",
    DATA / "Canonicalization" / "global_local_canonicalization_report.json",
    DATA
    / "Canonicalization"
    / "Global"
    / "A"
    / "global_formal_canonicalization_report.json",
    DATA / "Chirality" / "exact_acs_mirror_report.json",
    DATA / "Diagnostics" / "benchmark_report.json",
    DATA / "Diagnostics" / "backend_comparison_report.json",
    DATA / "Perception" / "cip_native_report.json",
    DATA / "Perception" / "full_detection_report.json",
    DATA / "Perception" / "perception_conformance_report.json",
    DATA / "Perception" / "rota_locus_report.json",
    DATA / "Perception" / "stereo_element_report.json",
)
GENERATED_OUTPUT_TESTS = {
    "Test/Benchmark/Stereo/Canonicalization/test_atom_relabel.py::"
    "test_acs_task_exhaustively_relabels_one_configured_case",
    "Test/Benchmark/Stereo/Canonicalization/test_atom_relabel.py::"
    "test_rota_task_reports_support_accuracy_not_handedness",
    "Test/Benchmark/Stereo/Canonicalization/"
    "test_configuration_free_global.py::"
    "test_retained_abc_matrix_is_complete_without_raw_b_or_c",
    "Test/Benchmark/Stereo/Canonicalization/"
    "test_configuration_free_local.py::test_retained_local_matrix_is_complete",
    "Test/Benchmark/Stereo/Canonicalization/test_global_local.py::"
    "test_frozen_selective_global_local_matrix_covers_every_family_and_class",
    "Test/Benchmark/Stereo/Canonicalization/test_inventory.py::"
    "test_inventory_preserves_source_specific_task_boundaries",
    "Test/Benchmark/Stereo/Canonicalization/test_inventory.py::"
    "test_csv_projection_has_one_row_per_source_record",
    "Test/Benchmark/Stereo/Canonicalization/test_multi_element.py::"
    "test_dataset_extraction_retains_task_specific_boundaries",
    "Test/Benchmark/Stereo/Canonicalization/test_run.py::"
    "test_acs_local_tables_distinguish_tetrahedral_and_planar_counts",
    "Test/Benchmark/Stereo/Canonicalization/test_run.py::"
    "test_parallel_acs_records_match_single_worker_results",
    "Test/Benchmark/Stereo/Perception/test_conformance_data.py::"
    "test_frozen_conformance_report_matches_the_live_stable_result",
    "Test/Benchmark/Stereo/Perception/test_full_detection.py::"
    "test_manifest_registers_full_detection_for_all_source_datasets",
    "Test/Benchmark/Stereo/Perception/test_full_detection.py::"
    "test_frozen_report_covers_every_row_and_every_source_annotation",
    "Test/Benchmark/Stereo/Perception/test_full_detection.py::"
    "test_report_contains_no_cip_structures_or_configuration_predictions",
    "Test/Benchmark/Stereo/test_cip_native_benchmark.py::"
    "test_frozen_native_report_covers_all_300_records",
    "Test/Benchmark/Stereo/test_cip_native_benchmark.py::"
    "test_frozen_native_report_contains_no_benchmark_structures",
    "Test/Benchmark/Stereo/test_stereo_dataset_registry.py::"
    "test_rota_fixture_is_exact_and_registered_for_axial_loci",
    "Test/Benchmark/Stereo/test_stereo_dataset_registry.py::"
    "test_frozen_whole_molecule_protocols_report_separate_conclusions",
    "Test/Benchmark/Stereo/test_stereo_dataset_registry.py::"
    "test_exact_mirror_audit_preserves_supplied_configuration_only",
    "Test/Benchmark/Stereo/test_stereo_dataset_registry.py::"
    "test_frozen_local_label_diagnostics_cover_both_settings",
    "Test/Benchmark/Stereo/test_stereo_dataset_registry.py::"
    "test_frozen_live_backends_cover_all_datasets_and_settings",
    "Test/Benchmark/Stereo/test_stereo_dataset_registry.py::"
    "test_provisional_cip_binary_reference_is_explicitly_retracted",
    "Test/Benchmark/Stereo/test_stereo_element_benchmark.py::"
    "test_frozen_audit_covers_all_300_records_without_promoting_ties",
    "Test/Benchmark/Stereo/test_stereo_element_benchmark.py::"
    "test_frozen_audit_does_not_redistribute_benchmark_structures",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip only assertions that require optional prior-run artifacts."""
    missing = [path for path in GENERATED_REPORTS if not path.is_file()]
    if not missing:
        return
    reason = (
        "generated stereo benchmark reports are absent; run "
        "Experiment/Stereo/benchmark.sh to exercise artifact assertions"
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.nodeid in GENERATED_OUTPUT_TESTS:
            item.add_marker(marker)
