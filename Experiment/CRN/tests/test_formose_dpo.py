"""The standalone DPO comparison preserves stoichiometry and failure scope."""

from argparse import Namespace
from copy import deepcopy
import json

import pytest

from Experiment.CRN.studies import formose_dpo as benchmark
from Experiment.CRN.studies.formose_case_study import RULES


def test_identity_audit_normalizes_explicit_hydrogens_and_retains_multiplicity():
    first = benchmark.normalize(["C=O", "OCC=O"], [(["C=O", "C=O"], ["OCC=O"])])
    second = benchmark.normalize(["[H]C(=O)[H]", "O=CCO"], [(["[H]C(=O)[H]", "C=O"], ["O=CCO"])])
    assert first == second
    assert first != benchmark.normalize(["C=O", "OCC=O"], [(["C=O"], ["OCC=O"])])


def test_identity_audit_does_not_cancel_shared_participants():
    participants = ["C=O", "OCC=O", "O"]
    intact = benchmark.normalize(participants, [(["C=O", "O"], ["OCC=O", "O"])])
    cancelled = benchmark.normalize(participants, [(["C=O"], ["OCC=O"])])
    assert intact["sha256"] != cancelled["sha256"]


def test_gml_translation_preserves_mapped_hydrogen_and_rejects_atom_deletion():
    assert 'label "H"' in benchmark.rule_gml(RULES[0], "enol")
    with pytest.raises(ValueError, match="retained atom labels"):
        benchmark.rule_gml("[C:1][O:2]>>[C:1]", "unsupported")


def completed_rows():
    return [
        dict(
            backend=backend,
            depth=4,
            status="complete",
            samples=[0.1, 0.2],
            warmup_samples=[0.3],
            snapshot_sha256="same",
        )
        for backend in ["synreactor-native", "mod", "molfoundry"]
    ]


@pytest.mark.parametrize("failure", ["missing", "duplicates", "partial", "mismatch", "source_changed"])
def test_comparison_cannot_pass_with_incomplete_or_mismatching_evidence(failure):
    rows = completed_rows()
    if failure == "missing":
        rows.pop()
    if failure == "duplicates":
        rows[-1] = deepcopy(rows[0])
    if failure == "partial":
        rows[-1]["samples"].pop()
    if failure == "mismatch":
        rows[-1]["snapshot_sha256"] = "different"
    options = Namespace(external=True, depths=[4], repeats=2, warmups=1)
    report = benchmark.finish_report({"runs": rows}, options, failure != "source_changed")
    assert report["status"] == "FAIL"


def test_comparison_accepts_complete_equal_snapshots():
    options = Namespace(external=True, depths=[4], repeats=2, warmups=1)
    report = benchmark.finish_report({"runs": completed_rows()}, options, True)
    assert report["status"] == "PASS" and report["complete"]


def test_requested_missing_interpreters_are_reported_without_silent_omission(tmp_path, monkeypatch):
    from synkit.CRN.Construct import native_dpo

    monkeypatch.setattr(native_dpo, "dpo_expander_manifest", lambda path: {})
    monkeypatch.setattr(benchmark, "source_hashes", lambda: {"fixture": "constant"})

    # Also prevent a native worker launch: the regression targets orchestration.
    def missing_worker(*args, **kwargs):
        raise FileNotFoundError("test interpreter is absent")

    monkeypatch.setattr(benchmark, "Worker", missing_worker)
    options = benchmark.parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dpo-expander",
            str(tmp_path / "unused"),
            "--external",
            "--depths",
            "4",
            "--warmups",
            "1",
            "--repeats",
            "2",
        ]
    )
    report = benchmark.benchmark(options)
    assert report["status"] == "FAIL" and not report["complete"]
    assert len(report["runs"]) == 3
    assert all(row["status"] == "error" and "FileNotFoundError" in row["failure"] for row in report["runs"])
    assert json.loads((tmp_path / "comparison.json").read_text())["status"] == "FAIL"
