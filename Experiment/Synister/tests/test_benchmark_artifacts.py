"""Keep source snapshots independent of local environments and old runs."""

from hashlib import sha256

import pytest

from Experiment.Synister.benchmark_artifacts import freeze_source, verify_source


def test_snapshot_keeps_project_sources_and_prunes_generated_trees(tmp_path):
    root = tmp_path / "checkout"
    included = (
        "synkit/Chem/Mapper/exact/search.py",
        "Experiment/Synister/emission_benchmark.py",
        "Experiment/Synister/tests/test_example.py",
    )
    excluded = (
        "Experiment/Synister/.venv-confirmation/lib/dependency.py",
        "Experiment/Synister/.venv-confirmation-reproduction/lib/dependency.py",
        "Experiment/Synister/runs/old/frozen_source/search.py",
        "Experiment/Synister/venv/lib/dependency.py",
        "synkit/Chem/Mapper/__pycache__/generated.py",
        "synkit/Chem/Mapper/exact/native_distance.cpp",
    )
    for name in included + excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")
    destination = tmp_path / "snapshot"
    hashes = freeze_source(root, destination)
    assert set(hashes) == set(included)
    assert {str(p.relative_to(destination)) for p in destination.rglob("*.py")} == set(
        included
    )
    for name, digest in hashes.items():
        assert (destination / name).read_bytes() == (root / name).read_bytes()
        assert digest == sha256((root / name).read_bytes()).hexdigest()
    verify_source(destination, hashes)


def test_snapshot_verification_detects_modified_source(tmp_path):
    root = tmp_path / "checkout"
    source = root / "synkit/search.py"
    source.parent.mkdir(parents=True)
    source.write_text("original\n")
    destination = tmp_path / "snapshot"
    hashes = freeze_source(root, destination)
    (destination / "synkit/search.py").write_text("modified\n")
    with pytest.raises(ValueError, match="synkit/search.py"):
        verify_source(destination, hashes)
