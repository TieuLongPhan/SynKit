"""Independent finite-carrier and Lewis-equation checks of native applications.

These tests check particular pushout diagrams in the typed-incidence ambient
category. They do not assert adhesivity of chemical graphs or predicates.
"""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor
from synkit.Graph.Morphism.constraints import WildcardConstraint
from synkit.Rule import SynRule

EXAMPLES = json.loads((Path(__file__).parent / "fixtures/wildcard_real_examples.json").read_text())["cases"]


def local_balance(graph):
    """Check both endpoints independently of the native charge-refresh code."""

    def pick(value, side):
        return value[side] if isinstance(value, tuple) and len(value) == 2 else value

    for node, attrs in graph.nodes(data=True):
        if "valence_electrons" not in attrs:
            continue
        for side in (0, 1):
            bond_sum = sum(
                pick(e["sigma_order"], side) + pick(e["pi_order"], side) for _, _, e in graph.edges(node, data=True)
            )
            assert pick(attrs["charge"], side) == (
                pick(attrs["valence_electrons"], side)
                - 2 * pick(attrs["lone_pairs"], side)
                - pick(attrs["radical"], side)
                - pick(attrs["hcount"], side)
                - bond_sum
            )


def incidence(vertices, bonds):
    """Atom, bond, incidence sorts with concrete shared carrier identities."""
    return (
        set(vertices),
        set(bonds),
        {(bond, endpoint) for bond in bonds for endpoint in bond[:2]},
    )


def carrier_dpo(graph, rule, mapping):
    """Check union/intersection characterizations of both finite pushouts."""
    completed = dict(mapping)
    completed.update(graph.graph.get("partial_completion", {}).get("introduced_mapping", ()))
    active = set(completed.values())

    def block(u, v, order):
        left, right = sorted((u, v), key=repr)
        return left, right, order

    host_bonds = {block(u, v, attrs["order"][0]) for u, v, attrs in graph.edges(data=True) if attrs["order"][0]}
    product_bonds = {block(u, v, attrs["order"][1]) for u, v, attrs in graph.edges(data=True) if attrs["order"][1]}
    left_bonds, right_bonds = set(), set()
    for u, v, attrs in rule.edges(data=True):
        if attrs["order"][0]:
            left_bonds.add(block(completed[u], completed[v], attrs["order"][0]))
        if attrs["order"][1]:
            right_bonds.add(block(completed[u], completed[v], attrs["order"][1]))
    interface_bonds = left_bonds & right_bonds
    context_bonds = host_bonds - (left_bonds - interface_bonds)
    L = incidence(active, left_bonds)
    K = incidence(active, interface_bonds)
    R = incidence(active, right_bonds)
    D = incidence(graph, context_bonds)
    G = incidence(graph, host_bonds)
    H = incidence(graph, product_bonds)
    for left, middle, right, context, host, product in zip(L, K, R, D, G, H):
        assert left & context == middle
        assert right & context == middle
        assert left | context == host
        assert right | context == product


def resource_updates(graph, rule, mapping):
    """Check declared field deltas independently of charge and serialization."""
    completed = dict(mapping)
    completed.update(graph.graph.get("partial_completion", {}).get("introduced_mapping", ()))

    def pick(value, side):
        return value[side] if isinstance(value, tuple) and len(value) == 2 else value

    for source, target in completed.items():
        for field in ("hcount", "lone_pairs", "radical"):
            observed = graph.nodes[target].get(field, 0)
            declared = rule.nodes[source].get(field, 0)
            assert pick(observed, 1) - pick(observed, 0) == pick(declared, 1) - pick(declared, 0)


def outcome_identity(value):
    result = []
    for side in value.split(">>"):
        mol = Chem.MolFromSmiles(side)
        assert mol is not None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        result.append(Chem.MolToSmiles(mol))
    return tuple(result)


@pytest.fixture(scope="module")
def library():
    root = Path(__file__).resolve().parents[3]
    return str(build_native_reactor(root / "build/partial/test-native"))


@pytest.mark.parametrize("case", EXAMPLES, ids=lambda case: case["id"])
def test_real_examples_have_valid_local_balances_and_carrier_pushouts(library, case):
    template = case["template"]
    if case.get("hydrogen_policy") == "absolute":
        template = SynRule.from_smart(template, format="tuple", implicit_h=False)
    engine = WildcardEngine(
        case["substrate"],
        template,
        partial=case["partial"],
        backend="cpp",
        library_path=library,
        dedup_its=False,
        wildcards={int(k): WildcardConstraint(**v) for k, v in case.get("wildcards", {}).items()},
    )
    assert len(engine.mappings) == case["mapping_count"]
    for graph, mapping in zip(engine.its_list, engine.mappings):
        local_balance(graph)
        carrier_dpo(graph, engine.rule.rc.raw, mapping)
        resource_updates(graph, engine.rule.rc.raw, mapping)
        maps = [attrs["atom_map"][0] for _, attrs in graph.nodes(data=True)]
        assert len(maps) == len(set(maps))
        for node, attrs in graph.nodes(data=True):
            assert attrs["atom_map"][0] == attrs["atom_map"][1]
            assert attrs["element"][0] == attrs["element"][1]
    assert {outcome_identity(s) for s in engine.smarts_list} == {outcome_identity(s) for s in case["expected"]}


def test_budget_two_does_not_claim_completeness_for_double_bond_contexts():
    # Two unit ports have distinct roots; they cannot stand for one double-bond root.
    from synkit.Graph.Morphism.constraints import WildcardConstraint

    engine = WildcardEngine(
        "CO",
        "[*:1][O:2][*:3]>>[*:1][O:2][*:3]",
        partial=False,
        wildcards={
            1: WildcardConstraint("attachment_port", owner=2),
            3: WildcardConstraint("attachment_port", owner=2),
        },
    )
    assert engine.mappings == []
