"""Smoke tests for the unified canonicalization task runner."""

from __future__ import annotations

from Experiment.Stereo.datasets import ROTA
from Experiment.Stereo.Canonicalization.inventory import (
    build_inventory,
)
from Experiment.Stereo.Canonicalization.atom_relabel import (
    _run_acs,
    _run_rota,
)
from Experiment.Stereo.Chirality.published import DATASET


def test_acs_task_exhaustively_relabels_one_configured_case() -> None:
    report = _run_acs(
        build_inventory(),
        path=DATASET,
        record_ids=("VS060",),
        limit=None,
        max_atoms=None,
        samples=2,
        exhaustive_max_atoms=0,
        force_exhaustive=True,
        seed=42,
        timeout_seconds=5.0,
    )

    assert report["canonicalization_scope"] == "configured_stereograph"
    assert report["summary"]["records_selected"] == 1
    assert report["summary"]["permutations_planned"] == 24
    assert report["summary"]["strict_invariance_accuracy"] == 1.0


def test_rota_task_reports_support_accuracy_not_handedness() -> None:
    report = _run_rota(
        build_inventory(),
        path=ROTA,
        record_ids=("RotA-0000",),
        limit=None,
        max_atoms=None,
        samples=2,
        exhaustive_max_atoms=0,
        force_exhaustive=False,
        seed=42,
        timeout_seconds=5.0,
    )

    assert report["canonicalization_scope"] == "axis_support_only"
    assert report["summary"]["records_selected"] == 1
    assert report["summary"]["permutations_planned"] == 2
    assert "source_locus_recall" in report["summary"]
