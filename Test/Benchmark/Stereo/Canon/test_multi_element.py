"""Regression gates for multi-element canonicalization composition."""

from __future__ import annotations

from Experiment.Stereo.Canonicalization.multi_element import (
    CIP_ELEMENT_REPORT,
    ROTA,
    ROTA_LOCUS_REPORT,
    _acs_inventory,
    _cip_extraction,
    _rota_extraction,
    benchmark_synthetic_composition,
)
from Experiment.Stereo.acs_molecular_chirality import DATASET


def test_symmetric_two_center_case_has_meso_fixed_class() -> None:
    report = benchmark_synthetic_composition(
        case_names=("symmetric_two_tetrahedral",),
        representative_relabelings=1,
    )

    assert report["passed"]
    assert report["totals"]["raw_assignments"] == 4
    assert report["totals"]["global_classes"] == 3
    assert report["totals"]["mirror_fixed_classes"] == 1
    assert report["totals"]["enantiomer_pairs"] == 1
    assert report["totals"]["diastereomer_pairs"] == 2
    assert report["totals"]["relabelings_checked"] == 1


def test_dataset_extraction_retains_task_specific_boundaries() -> None:
    acs = _acs_inventory(DATASET)
    rota = _rota_extraction(ROTA, ROTA_LOCUS_REPORT)
    cip = _cip_extraction(CIP_ELEMENT_REPORT, cip_path=None)

    assert acs["multi_element_records"] == 164
    assert acs["max_configured_elements"] == 25

    assert rota["multi_locus_records"] == 40
    assert rota["multi_reference_loci"] == 88
    assert rota["multi_records_renumbering_invariant"] == 40

    assert cip["records_with_multiple_reference_rs_positions"] == 153
    assert cip["records_with_multiple_attached_configurations"] == 116
    assert cip["multi_attached_renumbering_invariant"] == 116
    assert not cip["structure_inventory"]["available"]
