"""Python/C++ conformance, failure boundaries, and independent graph checks."""

import json
from pathlib import Path

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Morphism.constraints import WildcardConstraint
from synkit.IO import smiles_to_graph
from synkit.Synthesis.Reactor import (
    NativeReactorUnsupportedError,
    WildcardCompletionError,
    WildcardEngine,
    WildcardSearchLimitError,
    build_native_reactor,
)

ROOT = Path(__file__).resolve().parents[3]
CORPUS = json.loads((Path(__file__).parent / "fixtures/wildcard_conformance.json").read_text())
SN2 = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"


@pytest.fixture(scope="session")
def native_reactor():
    return str(build_native_reactor(ROOT / "build/partial/test-native"))


def identity(reaction):
    output = []
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        output.append(Chem.MolToSmiles(molecule))
    return tuple(output)


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_cpp_matches_independent_corpus(native_reactor, case):
    engine = WildcardEngine(
        case["substrate"],
        case["template"],
        backend="cpp",
        library_path=native_reactor,
        partial=case["partial"],
        electron_diagnostics=True,
        wildcards={int(k): WildcardConstraint(**v) for k, v in case.get("wildcards", {}).items()},
    )
    assert len(engine.mappings) == case["mapping_count"]
    assert {identity(s) for s in engine.smarts_list} == {identity(s) for s in case["expected"]}
    assert all(report["mismatch_count"] == 0 for report in engine.diagnostics)
    assert engine.native_statistics["abi"] == 1


@pytest.mark.parametrize(
    "substrate",
    [
        "CBr",
        "CCBr",
        "CCCBr",
        "CC(Br)C",
        "CC(C)(C)Br",
        "BrCCBr",
        "BrCCCBr",
        "C=CCBr",
        "C#CCBr",
        "c1ccccc1CBr",
        "c1ccccc1CCBr",
        "c1ccccc1Br",
        "CBr.[OH-]",
        "CCBr.C[O-]",
        "BrCCBr.[OH-].[OH-]",
        "[CH2-]Br",
    ],
)
@pytest.mark.parametrize("partial", [False, True])
def test_differential_bindings_and_outcomes(native_reactor, substrate, partial):
    python = WildcardEngine(substrate, SN2, partial=partial, dedup_its=False)
    cpp = WildcardEngine(substrate, SN2, partial=partial, dedup_its=False, backend="cpp", library_path=native_reactor)
    assert {tuple(sorted(m.items())) for m in cpp.mappings} == {tuple(sorted(m.items())) for m in python.mappings}
    assert {identity(s) for s in cpp.smarts_list} == {identity(s) for s in python.smarts_list}
    assert len(cpp.its_list) == len(python.its_list)


@pytest.mark.parametrize("substrate", ["CO", "CCO", "CCC(O)C"])
def test_reverse_execution(native_reactor, substrate):
    python = WildcardEngine(substrate, SN2, invert=True)
    cpp = WildcardEngine(substrate, SN2, invert=True, backend="cpp", library_path=native_reactor)
    assert {identity(s) for s in cpp.smarts_list} == {identity(s) for s in python.smarts_list}


def test_native_path_never_calls_python_match_completion_or_rewrite(native_reactor, monkeypatch):
    from synkit.Graph.Matcher.partial_matcher import PartialMatcher
    from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
    from synkit.Synthesis.Reactor.core import completion, rewrite

    def forbidden(*args, **kwargs):
        raise AssertionError("Python kernel invoked by C++ backend")

    monkeypatch.setattr(PartialMatcher, "get_mappings", forbidden)
    monkeypatch.setattr(SubgraphSearchEngine, "find_subgraph_mappings", forbidden)
    monkeypatch.setattr(completion, "complete_rule_host", forbidden)
    monkeypatch.setattr(rewrite, "_glue_graph", forbidden)
    engine = WildcardEngine("CCBr", SN2, backend="cpp", library_path=native_reactor)
    assert engine.smarts_list
    assert all(g.graph["application_provenance"]["backend"] == "cpp" for g in engine.its_list)


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_native_limits_return_no_partial_result(native_reactor, limit):
    engine = WildcardEngine("CBr.[OH-]", SN2, backend="cpp", library_path=native_reactor, embed_threshold=limit)
    for _ in range(2):
        with pytest.raises(WildcardSearchLimitError):
            _ = engine.its_list
        assert engine._native_cache is None and engine._its is None


def test_native_completion_failure_returns_no_partial_result(native_reactor):
    engine = WildcardEngine("CBr.[OH-]", SN2, backend="cpp", library_path=native_reactor)
    # A full mapping is valid, but a later proper subset cannot keep this boundary fixed.
    engine.rule.rc.raw.nodes[3]["charge"] = (-1, 1)
    for _ in range(2):
        with pytest.raises(WildcardCompletionError):
            _ = engine.its_list
        assert engine._native_cache is None and engine._its is None


def test_native_matches_sparse_atom_maps_and_relabelled_vertices(native_reactor):
    host = smiles_to_graph("[CH3:70][Br:90]")
    host = nx.relabel_nodes(host, {node: f"host-{node}" for node in host})
    python = WildcardEngine(host, SN2)
    cpp = WildcardEngine(host, SN2, backend="cpp", library_path=native_reactor)
    assert {identity(s) for s in cpp.smarts_list} == {identity(s) for s in python.smarts_list}
    maps = [attrs["atom_map"][0] for _, attrs in cpp.its_list[0].nodes(data=True)]
    assert {70, 90}.issubset(maps) and len(maps) == len(set(maps))


def test_native_rejects_uncompiled_stereo_instead_of_falling_back(native_reactor):
    engine = WildcardEngine("N[C@H](C)F", "[CH3:1][Br:2]>>[CH3:1][Br:2]", backend="cpp", library_path=native_reactor)
    with pytest.raises(NativeReactorUnsupportedError, match="Stereo"):
        _ = engine.smarts_list


def test_c_abi_rejects_null_request(native_reactor):
    from synkit.Synthesis.Reactor._native.runtime import load

    lib, _ = load(native_reactor)
    result = lib.skp_execute(None)
    try:
        assert lib.skp_status(result) == 1
        assert lib.skp_application_count(result) == 0
        assert b"null request" in lib.skp_error(result)
    finally:
        lib.skp_free(result)


def test_read_only_wildcard_cannot_hide_an_electron_loss():
    engine = WildcardEngine("CO", "[*:1][OH:2]>>[*:1].[OH:2]", partial=False)
    with pytest.raises(ValueError, match="read-only"):
        _ = engine.smarts_list


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_unread_radical_is_preserved_by_an_identity_rule(native_reactor, backend):
    engine = WildcardEngine(
        "[CH2]O",
        "[CH3:1][OH:2]>>[CH3:1][OH:2]",
        partial=False,
        radical_policy="ignore",
        backend=backend,
        library_path=native_reactor,
    )
    assert {identity(s) for s in engine.smarts_list} == {identity("[CH2]O>>[CH2]O")}


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_relaxed_guard_cannot_consume_a_nonexistent_radical(native_reactor, backend):
    engine = WildcardEngine(
        "C.[OH]",
        "[CH3:1].[OH:2]>>[CH3:1][OH:2]",
        partial=False,
        radical_policy="ignore",
        backend=backend,
        library_path=native_reactor,
    )
    assert engine.mappings == []


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_zero_rule_edge_is_not_a_write_to_host_context(native_reactor, backend):
    from synkit.IO import rsmi_to_its

    rule = rsmi_to_its("[CH3:1].[CH3:2]>>[CH3:1].[CH3:2]", format="tuple")
    rule.add_edge(1, 2, order=(0, 0), sigma_order=(0, 0), pi_order=(0, 0))
    engine = WildcardEngine("[CH2][CH2]", rule, partial=False, backend=backend, library_path=native_reactor)
    assert {identity(s) for s in engine.smarts_list} == {identity("[CH2][CH2]>>[CH2][CH2]")}


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_zero_edge_tuple_rule_still_uses_electron_aware_completion(native_reactor, backend):
    engine = WildcardEngine("O", "[CH4:1].[OH2:2]>>[CH4:1].[OH2:2]", backend=backend, library_path=native_reactor)
    expected = "O.C(*)(*)(*)*>>O.C(*)(*)(*)*"
    assert {identity(s) for s in engine.smarts_list} == {identity(expected)}
    assert len(engine.its_list[0].graph["partial_completion"]["attachment_ports"]) == 4
