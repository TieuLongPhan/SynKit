"""Compatibility checks across the method-oriented mapper source layout."""

import importlib
import pickle
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "historical, canonical",
    [
        ("exact", "solvers.exact"),
        ("exact.pabs.propagation", "solvers.exact.pabs.propagation"),
        ("slap.sequential", "solvers.slap.sequential"),
        ("_native.core", "solvers.native.core"),
        ("_native.build", "solvers.native.build"),
        ("analysis", "reporting.analysis"),
        ("spectrum", "reporting.spectrum"),
        ("native_analysis", "reporting.native_analysis"),
        ("alternatives", "reporting.alternatives"),
        ("aam_validator", "validation.aam_validator"),
        ("evaluation", "validation.bonds"),
        ("annotation_evaluation", "validation.annotations"),
        ("template_adapter", "adapters.template"),
        ("prediction_adapter", "adapters.prediction"),
        ("exact.kernel", "exact.kernel_solvers.kernel"),
        ("exact.exhaustive", "exact.distance_search.exhaustive"),
        ("exact.distance", "exact.distance_search.search"),
        ("exact.distance_records", "exact.distance_search.records"),
        ("exact.propagation", "exact.pabs.propagation"),
        ("exact.assignment_certificate", "exact.assignment.assignment_certificate"),
        ("exact.seed", "exact.seeds.seed"),
        ("exact.native_candidates", "exact.native.candidates"),
        ("exact.native_build", "exact.native.build"),
        ("exact.disk_frontier", "exact.native.disk_frontier"),
    ],
)
def test_historical_and_grouped_imports_share_module_objects(historical, canonical):
    implementation = importlib.import_module(f"synkit.Chem.Mapper.{canonical}")
    flat = importlib.import_module(f"synkit.Chem.Mapper.{historical}")
    legacy = importlib.import_module(f"synkit.Chem.Reaction.Mapper.{historical}")
    legacy_grouped = importlib.import_module(f"synkit.Chem.Reaction.Mapper.{canonical}")

    assert flat is legacy is legacy_grouped is implementation


@pytest.mark.parametrize(
    "historical, symbol",
    [
        ("analysis", "GlobalShellConfig"),
        ("exact.propagation", "PropagationConfig"),
        ("exact.distance_records", "DistanceEnumerationResult"),
        ("exact.kernel", "Kernel"),
        ("template_adapter", "ClassCorrespondence"),
    ],
)
def test_pickles_can_resolve_historical_class_references(historical, symbol):
    expected = getattr(importlib.import_module(f"synkit.Chem.Mapper.{historical}"), symbol)
    for namespace in ("synkit.Chem.Mapper", "synkit.Chem.Reaction.Mapper"):
        global_reference = f"c{namespace}.{historical}\n{symbol}\n.".encode("ascii")
        assert pickle.loads(global_reference) is expected


@pytest.mark.parametrize(
    "module",
    [
        "exact.native_build", "exact.native.build", "solvers.exact.native.build",
        "_native.build", "solvers.native.build",
    ],
)
def test_old_and_grouped_native_build_commands_remain_runnable(module):
    completed = subprocess.run(
        [sys.executable, "-m", f"synkit.Chem.Mapper.{module}", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output-dir" in completed.stdout
    assert "--compiler" in completed.stdout
    assert completed.stderr == ""
