"""Extraction partitions context conservatively and replays both directions."""

from copy import deepcopy

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Morphism import WildcardRole, llg_from_its
from synkit.IO import rsmi_to_its
from synkit.Rule import WildcardRuleExtractionError, WildcardRuleExtractor
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor

ETHYL = "[CH3:1][CH2:2][Br:3].[OH-:4]>>[CH3:1][CH2:2][OH:4].[Br-:3]"
PROPYL = "[CH3:1][CH2:2][CH2:3][Br:4].[OH-:5]>>[CH3:1][CH2:2][CH2:3][OH:5].[Br-:4]"


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    return str(build_native_reactor(tmp_path_factory.mktemp("extracted-wildcard")))


def identity(reaction):
    endpoints = []
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        endpoints.append(Chem.MolToSmiles(molecule))
    return tuple(endpoints)


def test_extraction_keeps_reacting_atoms_and_adds_an_observed_root_predicate():
    result = WildcardRuleExtractor().extract(ETHYL)
    assert result.core_atoms == (2, 3, 4)
    (port,) = result.ports
    assert (port.atom, port.owner, port.context_atoms) == (1, 2, (1,))
    assert port.constraint.role is WildcardRole.ATTACHMENT_PORT
    assert port.constraint.elements == frozenset({"C"})
    assert port.constraint.charges == port.constraint.radicals == frozenset({0})
    assert port.constraint.mapped_identity is None
    assert port.constraint.capacity == port.constraint.resource_budget == 1
    for side in (result.rule.left.raw, result.rule.right.raw):
        assert side.nodes[1]["element"] == "*"
        assert side.nodes[1]["owner"] == 2
        assert side.nodes[1]["elements"] == frozenset({"C"})
    assert result.rule._implicit_h is False
    assert result.rule.rc.raw.nodes[4]["_relative_resource_baselines"]["lone_pairs"] == 2


def test_single_attachment_context_is_abstracted_as_one_root_without_duplicate_atoms():
    result = WildcardRuleExtractor().extract(PROPYL)
    (port,) = result.ports
    assert (port.atom, port.owner, port.context_atoms) == (2, 3, (1, 2))
    assert set(result.rule.rc.raw) == {2, 3, 4, 5}


def test_shared_context_closing_a_ring_is_retained():
    result = WildcardRuleExtractor().extract(
        "[CH:1]1=[CH:2][CH2:3][CH2:4][CH2:5]1>>" "[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1"
    )
    assert result.core_atoms == (1, 2)
    assert result.retained_context_atoms == (3, 4, 5)
    assert result.ports == ()
    assert len(result.rule.rc.raw.edges) == 5


def test_double_bonded_context_is_retained_while_single_bonded_context_generalizes():
    result = WildcardRuleExtractor().extract("[CH2:1]=[CH:2][OH:3]>>[CH2:1]=[C-:2][OH:3]")
    assert result.core_atoms == (2,)
    assert result.retained_context_atoms == (1,)
    assert [port.atom for port in result.ports] == [3]


def test_spectators_are_not_baked_into_the_rule():
    left, right = ETHYL.split(">>")
    result = WildcardRuleExtractor().extract(left + ".[Na+:7]>>" + right + ".[Na+:7]")
    assert result.spectator_atoms == (7,)
    assert 7 not in result.rule.rc.raw


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "host,invert,partial",
    [
        ("CCBr.[OH-]", False, False),
        ("CCO.[Br-]", True, False),
        ("CCBr", False, True),
        ("CCO", True, True),
    ],
)
def test_extracted_wildcard_rule_recovers_both_endpoints(native_library, backend, host, invert, partial):
    extracted = WildcardRuleExtractor().extract(ETHYL)
    engine = WildcardEngine(
        host,
        extracted.rule,
        invert=invert,
        partial=partial,
        backend=backend,
        library_path=native_library if backend == "cpp" else None,
        electron_diagnostics=True,
    )
    assert identity(ETHYL) in {identity(value) for value in engine.smarts_list}
    assert all(report["mismatch_count"] == 0 for report in engine.diagnostics)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("host,invert", [("CCCBr.[OH-]", False), ("CCCO.[Br-]", True)])
def test_observed_root_domains_allow_longer_unseen_context(native_library, backend, host, invert):
    result = WildcardRuleExtractor().extract(ETHYL)
    engine = WildcardEngine(
        host,
        result.rule,
        partial=False,
        invert=invert,
        backend=backend,
        library_path=native_library if backend == "cpp" else None,
    )
    assert identity(PROPYL) in {identity(value) for value in engine.smarts_list}


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_extracted_root_domain_rejects_an_unadmitted_element(native_library, backend):
    result = WildcardRuleExtractor().extract(ETHYL)
    engine = WildcardEngine(
        "NCBr.[OH-]",
        result.rule,
        partial=False,
        backend=backend,
        library_path=native_library if backend == "cpp" else None,
    )
    assert engine.mappings == []


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_missing_interface_remains_symbolic_with_its_original_predicate(native_library, backend):
    result = WildcardRuleExtractor().extract(ETHYL)
    engine = WildcardEngine(
        "[OH-]", result.rule, partial=True, backend=backend, library_path=native_library if backend == "cpp" else None
    )
    assert engine.smarts_list
    for graph in engine.its_list:
        roots = [attrs for _, attrs in graph.nodes(data=True) if attrs.get("element") == ("*", "*")]
        assert len(roots) == 1
        assert roots[0]["elements"] == frozenset({"C"})


def test_lewis_endpoints_align_by_map_and_do_not_mutate_the_sources():
    its = rsmi_to_its(ETHYL, format="tuple")
    left = llg_from_its(its, "left", electron_complete=True).to_networkx()
    right = llg_from_its(its, "right", electron_complete=True).to_networkx()
    left = nx.relabel_nodes(left, {n: f"left-{n}" for n in left})
    right = nx.relabel_nodes(right, {n: f"right-{n}" for n in right})
    before = deepcopy(left), deepcopy(right)
    result = WildcardRuleExtractor().extract_from_lewis_graphs(left, right)
    assert result.core_atoms == (2, 3, 4)
    assert nx.utils.graphs_equal(left, before[0]) and nx.utils.graphs_equal(right, before[1])


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_unchanged_aromatic_context_is_validated_in_its_kekule_coordinates(native_library, backend):
    reaction = (
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7][Br:8].[OH-:9]>>"
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7][OH:9].[Br-:8]"
    )
    result = WildcardRuleExtractor().extract(reaction)
    assert result.core_atoms == (7, 8, 9)
    (port,) = result.ports
    assert port.atom == 6 and port.context_atoms == (1, 2, 3, 4, 5, 6)
    for host, invert in [("c1ccccc1CBr.[OH-]", False), ("c1ccccc1CO.[Br-]", True)]:
        engine = WildcardEngine(
            host,
            result.rule,
            invert=invert,
            partial=False,
            backend=backend,
            library_path=native_library if backend == "cpp" else None,
        )
        assert identity(reaction) in {identity(value) for value in engine.smarts_list}


@pytest.mark.parametrize(
    "reaction,code",
    [
        ("CCBr.[OH-]>>CCO.[Br-]", "INVALID_ATOM_MAPS"),
        ("[CH3:1][I:2].[NH2:3][CH3:4]>>[CH3:1][NH:3][CH3:4]", "UNBALANCED_ATOM_CARRIER"),
        ("[CH4:1]>>[NH3:1]", "CHANGED_ELEMENT"),
        ("[CH4:1]>>[CH4:1]", "EMPTY_CENTER"),
        ("[13CH4:1]>>[13CH4:1]", "UNSUPPORTED_ISOTOPE"),
        ("[H:1][OH:2]>>[H:1][OH:2]", "UNSUPPORTED_EXPLICIT_HYDROGEN"),
        ("[CH3:1][C@H:2]([Br:3])[Cl:4]>>[CH3:1][C@@H:2]([Br:3])[Cl:4]", "UNSUPPORTED_STEREO"),
    ],
)
def test_unsupported_sources_are_refused_without_fabricating_states(reaction, code):
    with pytest.raises(WildcardRuleExtractionError) as error:
        WildcardRuleExtractor().extract(reaction)
    assert error.value.code == code
