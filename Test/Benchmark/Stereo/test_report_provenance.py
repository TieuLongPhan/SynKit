"""A research report must refuse changed executable sources or supplied inputs."""

import json

import pytest

from Experiment.Stereo import provenance


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    (tmp_path / "synkit").mkdir()
    (tmp_path / "Experiment/Stereo").mkdir(parents=True)
    (tmp_path / "synkit/source.py").write_text("value = 1\n")
    (tmp_path / "Experiment/Stereo/runner.py").write_text("value = 2\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)

    def git(args, **_kwargs):
        return "" if args[1:] == ["status", "--porcelain"] else "immutable-test-commit\n"

    monkeypatch.setattr(provenance.subprocess, "check_output", git)
    return tmp_path


def test_report_captures_deduplicated_inputs_and_proves_they_remain_unchanged(source_tree):
    path = source_tree / "input.json"
    path.write_text('{"input": 1}\n')
    start = provenance.capture_source((path, path))
    assert start["input_sha256"] == {str(path): provenance.sha256(path)}
    assert len(start["source_sha256"]) == 2
    final = provenance.finish_source(start)
    assert final["inputs_unchanged_during_run"] and final["source_unchanged_during_run"]


@pytest.mark.parametrize("change", ["input", "source", "new_source"])
def test_changed_inputs_or_sources_refuse_a_completed_report(source_tree, change):
    path = source_tree / "input.json"
    path.write_text('{}\n')
    start = provenance.capture_source((path,))
    if change == "input":
        path.write_text('{"changed": true}\n')
    elif change == "source":
        (source_tree / "synkit/source.py").write_text("value = 99\n")
    else:
        (source_tree / "synkit/additional.py").write_text("value = 0\n")
    with pytest.raises(RuntimeError, match="changed during report generation"):
        provenance.finish_source(start)


def test_cli_conformance_report_identifies_the_actual_source_and_dataset(tmp_path, monkeypatch):
    from Experiment.Stereo.Perception import conformance

    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["conformance", "--output", str(output)])
    assert conformance.main() == 0
    report = json.loads(output.read_text())
    snapshot = report["provenance"]
    assert snapshot["source_unchanged_during_run"] and snapshot["inputs_unchanged_during_run"]
    expected = {str(conformance.DEFAULT_DATASET): provenance.sha256(conformance.DEFAULT_DATASET)}
    assert snapshot["input_sha256"] == expected
    assert "Experiment/Stereo/Perception/conformance.py" in snapshot["source_sha256"]


def test_protocol_tier_contains_every_missing_report_without_repeating_completed_studies(tmp_path):
    from types import SimpleNamespace
    from Experiment.Stereo.reproduce import stages

    args = SimpleNamespace(output_dir=tmp_path, tier="protocols", jobs=2, cip_path=tmp_path / "input.smi",
                           coordinate_path=tmp_path / "input.sdf", stereomolgraph=None)
    selected = stages(args)
    assert [name for name, _argv in selected] == [
        "relations", "rota_loci", "perception_conformance", "retained_detection", "elements",
        "axial_negatives", "cip_contract",
    ]
