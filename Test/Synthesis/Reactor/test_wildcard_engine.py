"""Semantic conformance for the Python kernel and its future native port."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Morphism.constraints import WildcardConstraint, WildcardRole
from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import (
    SynReactor,
    WildcardCompletionError,
    WildcardEngine,
    WildcardSearchLimitError,
)
from synkit.Synthesis.Reactor.core.completion import complete_rule_host
from synkit.Synthesis.Reactor.output.serialization import _tuple_endpoint_graphs

SN2 = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
CORPUS = json.loads((Path(__file__).parent / "fixtures/wildcard_conformance.json").read_text())


def chemical_identity(reaction):
    result = []
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        result.append(Chem.MolToSmiles(molecule))
    return tuple(result)


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_language_independent_conformance(case):
    engine = WildcardEngine(
        case["substrate"],
        case["template"],
        partial=case["partial"],
        wildcards={int(key): WildcardConstraint(**value) for key, value in case.get("wildcards", {}).items()},
        electron_diagnostics=True,
    )
    assert len(engine.mappings) == case["mapping_count"]
    assert {chemical_identity(value) for value in engine.smarts_list} == {
        chemical_identity(value) for value in case["expected"]
    }
    assert all(report["mismatch_count"] == 0 for report in engine.diagnostics)
    for graph in engine.its_list:
        left, right = _tuple_endpoint_graphs(graph)
        assert {attrs["atom_map"] for _, attrs in left.nodes(data=True)} == {
            attrs["atom_map"] for _, attrs in right.nodes(data=True)
        }
    if "port_counts" in case:
        assert (
            sorted(
                len(graph.graph.get("partial_completion", {}).get("attachment_ports", ())) for graph in engine.its_list
            )
            == case["port_counts"]
        )


def test_direct_tuple_partial_rewrite_uses_symbolic_completion():
    reactor = SynReactor("CBr", SN2, partial=True, explicit_h=False, template_format="tuple", electron_diagnostics=True)
    assert {chemical_identity(value) for value in reactor.smarts_list} == {chemical_identity("CBr.*[O-]>>CO*.[Br-]")}
    assert reactor.diagnostics[0]["mismatch_count"] == 0


def test_completion_is_pure_and_preserves_existing_host_context():
    reactor = WildcardEngine("CBr", SN2)
    host = reactor.graph.raw
    rule = reactor.rule.rc.raw
    host_before, rule_before = deepcopy(host), deepcopy(rule)
    mapping = reactor.mappings[0]
    completed, total = complete_rule_host(host, rule, mapping)
    assert nx.utils.graphs_equal(host, host_before)
    assert nx.utils.graphs_equal(rule, rule_before)
    assert mapping == reactor.mappings[0]
    assert set(total) == set(rule)
    assert len(set(total.values())) == len(total)
    for left, right, attrs in host.edges(data=True):
        assert completed.edges[left, right] == attrs
    # The new C--O bond belongs to the product; completion adds no such input edge.
    assert not completed.has_edge(total[1], total[3])


def test_completion_rejects_a_noninjective_map():
    engine = WildcardEngine("CBr", SN2)
    with pytest.raises(WildcardCompletionError, match="injective"):
        complete_rule_host(engine.graph.raw, engine.rule.rc.raw, {1: 1, 2: 1})


def test_persistent_ports_have_typed_owner_capacity_and_identity():
    graph = WildcardEngine("CBr", SN2).its_list[0]
    (port,) = graph.graph["partial_completion"]["attachment_ports"]
    constraint = graph.nodes[port]["_wildcard_constraint"]
    assert constraint.role is WildcardRole.ATTACHMENT_PORT
    assert constraint.capacity == constraint.resource_budget == 1
    assert constraint.owner in graph[port]
    assert graph.edges[port, constraint.owner]["order"] == (1, 1)
    assert "valence_electrons" not in graph.nodes[port]
    json.dumps(graph.graph["partial_completion"])


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_explicit_candidate_limit_fails_without_partial_output(limit):
    engine = WildcardEngine("CBr.[OH-]", SN2, embed_threshold=limit)
    with pytest.raises(WildcardSearchLimitError):
        _ = engine.smarts_list
    assert engine._its is None
    assert engine._smarts is None


def test_threshold_does_not_apply_the_legacy_one_hundredth_cap():
    assert len(WildcardEngine("CBr.[OH-]", SN2, embed_threshold=100).mappings) == 3


def test_reverse_rule_preserves_absolute_electron_baselines():
    rule = SynRule(rsmi_to_its(SN2, format="tuple"))
    reversed_rule = rule.reversed()
    for node in rule.rc.raw:
        assert (
            reversed_rule.rc.raw.nodes[node]["_relative_resource_baselines"]
            == rule.rc.raw.nodes[node]["_relative_resource_baselines"]
        )
    restored = reversed_rule.reversed()
    forward = WildcardEngine("CBr", rule, electron_diagnostics=True)
    roundtrip = WildcardEngine("CBr", restored, electron_diagnostics=True)
    assert {chemical_identity(value) for value in forward.smarts_list} == {
        chemical_identity(value) for value in roundtrip.smarts_list
    }
    backward = WildcardEngine("CO", rule, invert=True, electron_diagnostics=True)
    assert backward.smarts_list
    assert all(report["mismatch_count"] == 0 for report in backward.diagnostics)


def test_missing_component_with_unequal_endpoint_budget_is_rejected():
    engine = WildcardEngine("CBr", SN2)
    rule = deepcopy(engine.rule.rc.raw)
    rule.nodes[3]["charge"] = (-1, 1)
    with pytest.raises(WildcardCompletionError, match="differs"):
        complete_rule_host(engine.graph.raw, rule, engine.mappings[0])


def test_missing_component_with_fractional_budget_is_rejected():
    engine = WildcardEngine("CBr", SN2)
    rule = deepcopy(engine.rule.rc.raw)
    rule.nodes[3]["valence_electrons"] = (6.5, 6.5)
    with pytest.raises(WildcardCompletionError, match="integer"):
        complete_rule_host(engine.graph.raw, rule, engine.mappings[0])


def test_query_charge_domain_is_exact():
    template = "[*:1][CH3:2]>>[*:1][CH3:2]"
    constraints = {1: WildcardConstraint("attachment_port", owner=2, elements={"N"}, charges={1})}
    assert WildcardEngine("C[NH3+]", template, partial=False, wildcards=constraints).smarts_list
    assert WildcardEngine("CN", template, partial=False, wildcards=constraints).smarts_list == []


def test_bare_query_keeps_both_incident_edges():
    template = "[CH3:1][*:2][CH3:3]>>[CH3:1][*:2][CH3:3]"
    assert WildcardEngine("CO", template, partial=False).mappings == []


def test_attachment_port_requires_one_owner_edge():
    engine = WildcardEngine(
        "COC", "[CH3:1][*:2][CH3:3]>>[CH3:1][*:2][CH3:3]", wildcards={2: WildcardConstraint("attachment_port", owner=1)}
    )
    with pytest.raises(ValueError, match="exactly one"):
        _ = engine.mappings


def test_factory_retains_typed_defaults():
    engine = WildcardEngine.from_smiles("CBr", SN2)
    assert engine.template_format == "tuple"
    assert engine.partial
    assert engine.smarts_list


def test_completion_errors_do_not_cache_empty_or_partial_results():
    engine = WildcardEngine("CBr", SN2)
    engine.rule.rc.raw.nodes[3]["valence_electrons"] = (6.5, 6.5)
    for _ in range(2):
        with pytest.raises(WildcardCompletionError):
            _ = engine.its_list
        assert engine._its is None


def test_attachment_ports_cannot_change_their_owner_edge():
    engine = WildcardEngine(
        "CO",
        "[*:1][OH:2]>>[*:1].[OH:2]",
        wildcards={1: WildcardConstraint("attachment_port", owner=2)},
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="owner|exactly one|read-only"):
            _ = engine.rule


def test_completion_uses_unique_atom_maps_with_sparse_host_labels():
    from synkit.IO import smiles_to_graph

    host = smiles_to_graph("[CH3:70][Br:90]")
    host = nx.relabel_nodes(host, {node: f"host-{node}" for node in host})
    engine = WildcardEngine(host, SN2)
    graph = engine.its_list[0]
    left, right = _tuple_endpoint_graphs(graph)
    maps = [attrs["atom_map"] for _, attrs in left.nodes(data=True)]
    assert len(maps) == len(set(maps))
    assert {70, 90}.issubset(maps)
    assert set(maps) == {attrs["atom_map"] for _, attrs in right.nodes(data=True)}


def test_relative_resource_baselines_are_part_of_rule_identity():
    rule = SynRule(rsmi_to_its(SN2, format="tuple"))
    altered_graph = deepcopy(rule.rc.raw)
    altered_graph.nodes[3]["_relative_resource_baselines"]["lone_pairs"] += 1
    altered = SynRule(altered_graph, format="tuple")
    assert rule != altered


@pytest.mark.parametrize("embed_threshold", [None, 1])
def test_explicit_h_extension_does_not_require_the_missing_component(embed_threshold):
    engine = WildcardEngine(
        "CCO",
        "[CH2:1]([H:3])[OH:2].[OH:4]>>[CH:1]=[O:2].[OH:4].[H:3]",
        embed_threshold=embed_threshold,
    )
    assert engine.smarts_list
    assert all(graph.graph["partial_completion"]["introduced_atoms"] for graph in engine.its_list)


def test_symbolic_host_requires_an_explicit_unification_implementation():
    engine = WildcardEngine("CO*", "[*:1][OH:2]>>[*:1][O-:2]")
    with pytest.raises(ValueError, match="concrete"):
        _ = engine.mappings


def test_typed_serialization_failure_is_not_reported_as_no_application(monkeypatch):
    engine = WildcardEngine("CBr", SN2)
    monkeypatch.setattr(engine, "_to_smarts", lambda graph: None)
    with pytest.raises(ValueError, match="typed ITS"):
        _ = engine.smarts_list
    assert engine.serialization_failure_indices == (0,)
    assert engine._smarts is None


@pytest.mark.parametrize("substituent", ["C", "CC", "CC(C)C"])
def test_grounding_a_port_commutes_with_full_rule_application(substituent):
    symbolic = WildcardEngine("CBr", SN2).smarts_list[0]
    left, right = chemical_identity(symbolic)
    grounded_left = left.replace("*", substituent)
    grounded_right = right.replace("*", substituent)
    expected = chemical_identity(f"{grounded_left}>>{grounded_right}")
    concrete = WildcardEngine(grounded_left, SN2, partial=False)
    assert {chemical_identity(value) for value in concrete.smarts_list} == {expected}


def test_rule_vertex_relabeling_preserves_partial_application():
    rule = SynRule(rsmi_to_its(SN2, format="tuple"))
    relabeled = nx.relabel_nodes(rule.rc.raw, {node: f"rule-{node}" for node in rule.rc.raw})
    original = WildcardEngine("CBr", rule)
    renamed = WildcardEngine("CBr", relabeled)
    assert {chemical_identity(value) for value in original.smarts_list} == {
        chemical_identity(value) for value in renamed.smarts_list
    }


def test_unbound_stereo_boundary_is_explicitly_outside_the_prototype_scope():
    engine = WildcardEngine(
        "CBr",
        "[CH3:1][Br:2].[F:4][C@H:3]([Cl:5])[O-:6]>>" "[CH3:1][O:6][C@H:3]([F:4])[Cl:5].[Br-:2]",
    )
    with pytest.raises(WildcardCompletionError, match="stereo boundary"):
        _ = engine.smarts_list
    assert engine._its is None


@pytest.mark.parametrize(
    "constraint",
    [
        WildcardConstraint("attachment_port", owner=4, capacity=0),
        WildcardConstraint("attachment_port", owner=4, resource_budget=0),
        WildcardConstraint("attachment_port", owner=4, bond_orders={2}),
        WildcardConstraint("attachment_port", owner=4, side="product"),
    ],
)
def test_missing_wildcard_cannot_supply_an_unsatisfiable_binding(constraint):
    engine = WildcardEngine(
        "CBr",
        "[CH3:1][Br:2].[*:3][O-:4]>>[CH3:1][O:4][*:3].[Br-:2]",
        wildcards={3: constraint},
    )
    with pytest.raises(WildcardCompletionError):
        _ = engine.smarts_list
