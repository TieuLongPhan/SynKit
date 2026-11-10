"""Declared aromaticity changes execute natively with a valid Kekule phase."""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from synkit.Rule import WildcardRuleExtractor
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor

CASES = json.loads((Path(__file__).parent / "fixtures/wildcard_aromatic_examples.json").read_text())["cases"]


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    return str(build_native_reactor(tmp_path_factory.mktemp("aromatic-native")))


def identity(reaction):
    result = []
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        result.append(Chem.MolToSmiles(molecule))
    return tuple(result)


def host_for(case, invert, mode):
    expected = identity(case["reaction"])
    source = expected[1 if invert else 0]
    if mode == "partial_largest":
        source = min(source.split("."), key=lambda s: (-Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s))
    return source


def assert_lewis_equations(graph):
    for node, attrs in graph.nodes(data=True):
        if "valence_electrons" not in attrs:
            continue
        for side in (0, 1):
            bonds = sum(a["sigma_order"][side] + a["pi_order"][side] for _, _, a in graph.edges(node, data=True))
            assert attrs["charge"][side] == (
                attrs["valence_electrons"][side]
                - 2 * attrs["lone_pairs"][side]
                - attrs["radical"][side]
                - attrs["hcount"][side]
                - bonds
            )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("invert", [False, True], ids=["fw", "bw"])
@pytest.mark.parametrize("mode", ["full", "partial_all", "partial_largest"])
def test_five_previously_refused_reactions_recover_both_endpoints(native_library, case, invert, mode):
    rule = WildcardRuleExtractor().extract(case["reaction"]).rule
    host = host_for(case, invert, mode)
    options = {"invert": invert, "partial": mode != "full"}
    python = WildcardEngine(host, rule, **options)
    native = WildcardEngine(host, rule, backend="cpp", library_path=native_library, **options)
    expected = identity(case["reaction"])
    actual = {identity(value) for value in native.smarts_list}
    assert expected in actual
    assert actual == {identity(value) for value in python.smarts_list}
    for graph in native.its_list:
        assert_lewis_equations(graph)
    assert native.native_statistics["abi"] == 1


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("invert", [False, True], ids=["fw", "bw"])
def test_native_aromaticity_does_not_use_python_rewrite_or_phase_repair(native_library, monkeypatch, case, invert):
    from synkit.Synthesis.Reactor.core import completion, product, rewrite
    from synkit.Graph.Matcher.partial_matcher import PartialMatcher
    from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine

    rule = WildcardRuleExtractor().extract(case["reaction"]).rule
    host = host_for(case, invert, "full")

    def forbidden(*args, **kwargs):
        raise AssertionError("Python matching, rewrite or Kekule repair was used.")

    monkeypatch.setattr(completion, "complete_rule_host", forbidden)
    monkeypatch.setattr(rewrite, "_glue_graph", forbidden)
    monkeypatch.setattr(product, "_reperceive_product_kekule_phase", forbidden)
    monkeypatch.setattr(PartialMatcher, "get_mappings", forbidden)
    monkeypatch.setattr(SubgraphSearchEngine, "find_subgraph_mappings", forbidden)
    engine = WildcardEngine(
        host, rule, backend="cpp", library_path=native_library, invert=invert, partial=False, dedup_its=False
    )
    assert engine.its_list
    for graph in engine.its_list:
        assert_lewis_equations(graph)
    # Serialization may use RDKit, but cannot repair the native graph phase.
    assert identity(case["reaction"]) in {identity(value) for value in engine.smarts_list}


def test_symbolic_root_without_electron_inventory_is_not_a_charge_mismatch():
    case = json.loads((Path(__file__).parent / "fixtures/wildcard_symbolic_example.json").read_text())
    rule = WildcardRuleExtractor().extract(case["reaction"]).rule
    engine = WildcardEngine(case["host"], rule, partial=True, electron_diagnostics=True)
    assert engine.smarts_list
    symbolic = [a for g in engine.its_list for _, a in g.nodes(data=True) if a["element"] == ("*", "*")]
    assert symbolic
    assert all("valence_electrons" not in a for a in symbolic)
    assert all(report["mismatch_count"] == 0 for report in engine.diagnostics)
    for graph in engine.its_list:
        assert_lewis_equations(graph)
