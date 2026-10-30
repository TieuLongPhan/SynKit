"""Core-unit bridge, ABI compatibility and native class/frontend integration."""

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from synkit.Chem.Mapper import Mapper
from synkit.Chem.Mapper._native.build import build_mapper_native
from synkit.Chem.Mapper._native.core import enumerate_core_mappings, map_reaction_core
from synkit.Chem.Mapper._native.runtime import load_core
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    override = os.environ.get("SYNKIT_CORE_LIBRARY")
    if override:
        return Path(override)
    if not shutil.which("g++"):
        pytest.skip("C++17 compiler unavailable")
    return build_mapper_native(tmp_path_factory.mktemp("common-core"))


def pair():
    return (
        LabeledGraph({0: {1: 0.5}, 1: {0: 0.5}}, [6, 8]),
        LabeledGraph({0: {1: 1.0}, 1: {0: 1.0}}, [6, 8]),
    )


def test_additive_abis_and_half_unit_bridge(library):
    lib = load_core(str(library))
    assert lib.synkit_mapper_core_abi() == 1
    assert lib.synkit_mapper_abi() == 1
    assert lib.synkit_distance_abi() == 3
    result = enumerate_core_mappings(pair(), library_path=library)
    assert result.complete and result.minimum_cost == 0.5 and result.distances == [0.5]
    assert result.backend_statistics["minimum2"] == 1
    from synkit.Chem.Mapper.exact.native.search import enumerate_cpp_mappings

    legacy = enumerate_cpp_mappings(
        pair(), library_path=library, binary=False, max_bijections=None
    )
    assert legacy.complete and legacy.minimum_cost == result.minimum_cost


@pytest.mark.parametrize(
    "target,mode,expected",
    [
        (0.25, "exact", []),
        (0.49, "at_most", []),
        (0.5, "exact", [0.5]),
        (0.75, "at_most", [0.5]),
    ],
)
def test_lattice_and_numeric_queries(library, target, mode, expected):
    result = enumerate_core_mappings(
        pair(), library_path=library, CD=target, CD_mode=mode
    )
    assert (
        result.complete and result.minimum_cost is None and result.distances == expected
    )


def test_unsupported_limit_is_not_timeout(library):
    result = enumerate_core_mappings(pair(), library_path=library, max_states=1)
    assert not result.complete and result.status == "unsupported"
    assert result.truncation_reason == "unsupported" and result.minimum_cost is None


def test_public_native_uses_native_class_pipeline(library, monkeypatch):
    import synkit.Chem.Mapper.reaction_search as frontend

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Python class search was called")

    monkeypatch.setattr(frontend, "_exact_code", forbidden)
    result = Mapper("native", library_path=library).map(
        "O=CC=O>>OC#CO", max_mappings=None
    )
    assert result.complete and result.optimality_proven and result.minimum_cost == 8
    assert len(result.mappings) == 2
    assert result.search_result.backend_statistics["classification_complete"] == 1


@pytest.mark.parametrize("backend", ["pabs", "astar", "cpabs"])
def test_native_class_ids_match_reference_frontend(library, backend):
    options = {
        "library_path": library,
        "backend": backend,
        "time_limit_seconds": 10,
        "max_mappings": None,
    }
    actual = map_reaction_core("O=CC=O>>OC#CO", **options)
    reference = map_reaction_core(
        "O=CC=O>>OC#CO", native_classification=False, **options
    )
    assert actual.complete and reference.complete
    assert {(r.class_id, r.distance) for r in actual.reactions} == {
        (r.class_id, r.distance) for r in reference.reactions
    }


def test_class_limit_preserves_proof_and_search_status(library):
    result = map_reaction_core(
        "O=CC=O>>OC#CO",
        library_path=library,
        classification_max_search_nodes=1,
        time_limit_seconds=10,
        max_mappings=None,
    )
    assert result.minimum_cost == 8 and result.search_complete
    assert not result.classification_complete and not result.complete
    assert result.incomplete_reason.startswith("classification:")


def test_compiled_source_snapshot_matches_manifest(library):
    manifest = library.with_suffix(".json")
    if not manifest.exists():
        pytest.skip("External test artifact has no build manifest")
    saved = json.loads(manifest.read_text())
    archive = Path(saved["source_archive"])
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == saved["source_archive_sha256"]
    )
    with tarfile.open(archive) as bundle:
        for name, expected in saved["configuration"]["sources"].items():
            member = bundle.extractfile("Mapper/solvers/native/cpp/" + Path(name).as_posix())
            assert member and hashlib.sha256(member.read()).hexdigest() == expected
        member = bundle.extractfile("Mapper/solvers/exact/native/native_distance.cpp")
        assert (
            member
            and hashlib.sha256(member.read()).hexdigest()
            == saved["configuration"]["legacy_pabs_source_sha256"]
        )
