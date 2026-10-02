"""Disk storage must preserve exact classes across tiny worker batches."""

import gzip
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from synkit.Chem.Mapper.analysis import GlobalShellConfig
from synkit.Chem.Mapper.exact.disk_frontier import (
    DiskOrbitStore,
    analyze_disk_reference_shell,
)
from synkit.Chem.Mapper.native_analysis import analyze_reference_blinded_native_shell
from Test.Chem.Mapper._helpers import graph


def test_exact_store_handles_identifier_collisions_and_big_counts(tmp_path):
    store = DiskOrbitStore(tmp_path / "exact.sqlite")
    observer = SimpleNamespace(count=0, stream_digest=hashlib.sha256())
    accumulator = SimpleNamespace(
        reactant_order=2**80,
        product_order=2**80,
        observer=observer,
        atom_frequencies=Counter(),
        bond_frequencies=Counter(),
    )
    # Same public digest, distinct full certificates. Neither may be discarded.
    key1 = (b"palette", b"SKR2" + bytes(32) + bytes(2) + b"x")
    key2 = (b"palette", b"SKR2" + bytes(32) + bytes(2) + b"y")
    record = (2**80, 1, bytes(32), key1, (), ())
    assert store.add(key1, record, accumulator)
    assert not store.add(key1, record, accumulator)
    assert store.add(key2, record, accumulator)
    assert store.classes == 2 and observer.count == 2**81
    assert store.db.execute("SELECT w FROM templates").fetchone()[0] == str(2**81)
    with pytest.raises(RuntimeError, match="inconsistent exact"):
        store.add(key1, (2**79, 2, bytes(32), key1, (), ()), accumulator)
    with pytest.raises(RuntimeError, match="unproved"):
        store.add(key2, (3, 2, bytes(32), key1, (), ()), accumulator)
    store.add_witnesses([b"one", b"two", b"one"])
    assert store.contains("witnesses", b"one")
    assert not store.contains("witnesses", b"three")
    exported = store.export("templates", tmp_path / "counts.gz")
    assert exported["rows"] == 1 and int(exported["total_weight"]) == 2**81
    store.close()
    with pytest.raises(ValueError, match="overwrite"):
        DiskOrbitStore(tmp_path / "exact.sqlite")


@pytest.mark.parametrize("seed", range(6))
def test_disk_spectra_equal_in_memory_across_batches(
    native_library, tmp_path, monkeypatch, seed
):
    monkeypatch.setenv("SYNKIT_NATIVE_PATTERN_CACHE", "0")
    rng = np.random.default_rng(seed)
    a = nx.to_numpy_array(nx.star_graph(4))
    b = np.triu(rng.choice([0.0, 1.0, 1.5], (5, 5)), 1)
    b += b.T
    if seed == 0:
        b = a.copy()
    lgp = graph(a), graph(b)
    reference = tuple(range(5))
    expected = analyze_reference_blinded_native_shell(
        lgp,
        reference,
        library_path=native_library,
        workers=2,
        config=GlobalShellConfig(time_limit_seconds=60, max_mappings=None),
    ).as_dict()
    actual = analyze_disk_reference_shell(
        lgp,
        reference,
        library_path=native_library,
        output=tmp_path / "run",
        workers=2,
        slice_nodes=3,
        seconds=60,
    )
    assert (
        expected["complete"] and actual["complete"] and actual["structure"]["complete"]
    )
    for name in (
        "representative_solution_count",
        "labeled_solution_count",
        "symmetry_group_order",
        "mapping_hartley_entropy_nats",
        "reference_class_observed",
        "reference_mapping_observed",
    ):
        assert actual[name] == expected[name]
    for name, value in actual["reaction_center"].items():
        if name != "representative_stream_sha256":
            assert json.dumps(value) == json.dumps(expected["reaction_center"][name])
    for table, name in [
        ("its", "its_class_counts"),
        ("templates", "template_class_counts"),
    ]:
        with gzip.open(tmp_path / f"run/{table}_class_counts.jsonl.gz", "rt") as stream:
            assert [json.loads(line) for line in stream] == expected["structure"][name]


def test_disk_expired_search_never_reports_complete(native_library, tmp_path):
    a = nx.to_numpy_array(nx.path_graph(4))
    result = analyze_disk_reference_shell(
        (graph(a), graph(a)),
        tuple(range(4)),
        library_path=native_library,
        output=tmp_path / "expired",
        workers=1,
        seconds=1e-9,
    )
    assert not result["complete"] and not result["structure"]["complete"]
    assert result["truncation_reason"] == "time_limit"
