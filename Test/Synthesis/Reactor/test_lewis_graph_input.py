"""Direct Lewis hosts agree with SMILES execution without losing state."""

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Morphism import ELECTRON_LLG_SCHEMA, LLGError, LewisLabelledGraph
from synkit.IO import smiles_to_graph
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor

SN2 = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    return str(build_native_reactor(tmp_path_factory.mktemp("lewis-input")))


def chemical_endpoints(reactions):
    result = set()
    for reaction in reactions:
        endpoints = []
        for side in reaction.split(">>"):
            molecule = Chem.MolFromSmiles(side)
            assert molecule is not None
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
            endpoints.append(Chem.MolToSmiles(molecule))
        result.add(tuple(endpoints))
    return result


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "smiles,invert,partial",
    [
        ("CBr.[OH-]", False, False),
        ("CO.[Br-]", True, False),
        ("CBr", False, True),
        ("CO", True, True),
    ],
)
def test_direct_lewis_execution_agrees_with_smiles_in_both_directions(native_library, backend, smiles, invert, partial):
    host = LewisLabelledGraph.from_electron_graph(smiles_to_graph(smiles))
    original = host.to_networkx()
    options = dict(
        backend=backend,
        library_path=native_library if backend == "cpp" else None,
        invert=invert,
        partial=partial,
        electron_diagnostics=True,
    )
    direct = WildcardEngine.from_lewis_graph(host, SN2, **options)
    reference = WildcardEngine(smiles, SN2, **options)
    assert direct.smarts_list
    assert chemical_endpoints(direct.smarts_list) == chemical_endpoints(reference.smarts_list)
    assert all(d["mismatch_count"] == 0 for d in direct.diagnostics)
    assert nx.utils.graphs_equal(host.to_networkx(), original)


@pytest.mark.parametrize("frozen", [False, True])
def test_lewis_host_does_not_round_trip_through_smiles(monkeypatch, frozen):
    from synkit.Synthesis.Reactor.core import engine

    host = smiles_to_graph("CBr")
    if frozen:
        host = LewisLabelledGraph.from_electron_graph(host)
    rule = SynRule.from_smart(SN2, format="tuple", implicit_h=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("The host must not be reparsed from SMILES.")

    monkeypatch.setattr(engine, "smiles_to_graph", forbidden)
    assert WildcardEngine.from_lewis_graph(host, rule).its_list


def test_constructor_accepts_a_frozen_lewis_host_directly():
    host = LewisLabelledGraph.from_electron_graph(smiles_to_graph("CBr"))
    assert WildcardEngine(host, SN2).smarts_list


def test_open_patterns_are_not_treated_as_complete_hosts():
    graph = smiles_to_graph("CBr")
    pattern = LewisLabelledGraph.from_networkx(graph, ELECTRON_LLG_SCHEMA, partial=True)
    engine = WildcardEngine.from_lewis_graph(pattern, SN2)
    with pytest.raises(ValueError, match="total concrete"):
        _ = engine.mappings


def test_inconsistent_frozen_host_is_checked_before_execution():
    graph = smiles_to_graph("CBr")
    graph.nodes[1]["charge"] = 7
    host = LewisLabelledGraph.from_networkx(graph, ELECTRON_LLG_SCHEMA)
    engine = WildcardEngine.from_lewis_graph(host, SN2)
    with pytest.raises(LLGError, match="Stored charge"):
        _ = engine.mappings
    assert engine._its is None
