"""Completion factorization and private-record DPO instances.

The ambient diagrams below retain resource values and sigma/pi coordinates.
These are finite examples, not a proof of adhesivity of molecular graphs.
"""

from copy import deepcopy
from pathlib import Path

import networkx as nx
import pytest

from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import (
    WildcardCompletionError,
    WildcardEngine,
    build_native_reactor,
)
from synkit.Synthesis.Reactor.core.completion import (
    complete_rule_host,
    validate_lewis_endpoints,
)
from synkit.Synthesis.Reactor.output.serialization import _tuple_endpoint_graphs

SN2 = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
FIELDS = ("element", "valence_electrons", "lone_pairs", "radical", "hcount", "charge")


def pick(value, side):
    return value[side] if isinstance(value, tuple) and len(value) == 2 else value


def test_completion_refuses_a_cut_inside_a_source_component_without_mutation():
    engine = WildcardEngine("CBr", SN2)
    host, rule = engine.graph.raw, engine.rule.rc.raw
    before = deepcopy(host), deepcopy(rule)
    with pytest.raises(WildcardCompletionError, match="whole source components"):
        complete_rule_host(host, rule, {1: 1})
    assert nx.utils.graphs_equal(host, before[0])
    assert nx.utils.graphs_equal(rule, before[1])


def test_legacy_completion_can_explicitly_keep_its_within_component_behavior():
    engine = WildcardEngine("CBr", SN2)
    completed, total = complete_rule_host(
        engine.graph.raw,
        engine.rule.rc.raw,
        {1: 1},
        componentwise=False,
    )
    assert completed.has_edge(total[1], total[2])
    assert total[2] not in engine.graph.raw


def test_completion_refuses_binding_a_source_absent_vertex():
    host, rule = large_inventory_completion()
    rule.nodes[1]["present"] = (False, True)
    with pytest.raises(WildcardCompletionError, match="source-present"):
        complete_rule_host(host, rule, {1: 10})


def large_inventory_completion():
    """A small formal inventory example outside native numeric bounds."""
    rule = nx.Graph()
    rule.add_node(
        1, element=("O", "O"), valence_electrons=(6, 6), lone_pairs=(2, 2), hcount=(2, 2), radical=(0, 0), charge=(0, 0)
    )
    rule.add_node(
        2,
        element=("C", "C"),
        valence_electrons=(2**53 + 1,) * 2,
        lone_pairs=(2**52,) * 2,
        hcount=(0, 0),
        radical=(0, 0),
        charge=(0, 0),
    )
    host = nx.Graph()
    host.add_node(10, element="O", valence_electrons=6, lone_pairs=2, hcount=2, radical=0, charge=0)
    return host, rule


def test_completion_uses_exact_integer_resource_budgets():
    host, rule = large_inventory_completion()
    completed, total = complete_rule_host(host, rule, {1: 10})
    (port,) = completed.graph["partial_completion"]["attachment_ports"]
    assert completed.has_edge(total[2], port)
    validate_lewis_endpoints(completed)


def test_endpoint_validation_detects_one_unit_error_above_float_precision():
    graph = nx.Graph()
    graph.add_node(1, valence_electrons=2**53 + 1, lone_pairs=2**52, radical=0, hcount=0, charge=1)
    validate_lewis_endpoints(graph)
    graph.nodes[1]["charge"] = 0
    with pytest.raises(WildcardCompletionError, match="local Lewis model"):
        validate_lewis_endpoints(graph)


def test_declared_charge_exchange_is_compared_as_an_exact_integer():
    graph, rule = nx.Graph(), nx.Graph()
    graph.add_node(1, charge=(2**53, 2**53 + 1))
    rule.add_node(1, charge=(0, 0))
    with pytest.raises(WildcardCompletionError, match="net charge exchange"):
        validate_lewis_endpoints(graph, rule)


@pytest.mark.parametrize("value", [True, -1, 0.5, float("nan"), float("inf"), "1"])
def test_completion_rejects_noninteger_or_negative_resources(value):
    host, rule = large_inventory_completion()
    rule.nodes[2]["valence_electrons"] = (value, value)
    with pytest.raises(WildcardCompletionError, match="integer"):
        complete_rule_host(host, rule, {1: 10})


@pytest.fixture(scope="module")
def library():
    root = Path(__file__).resolve().parents[3]
    return str(build_native_reactor(root / "build/partial/test-native"))


def private_records(graph, side, atoms=None, pairs=None):
    """Five sorts: atoms, bonds, incidences, resource records and ownership."""
    atoms = set(graph) if atoms is None else set(atoms)
    bonds = set()
    for u, v, attrs in graph.edges(data=True):
        pair = frozenset((u, v))
        if pairs is not None and pair not in pairs:
            continue
        if pick(attrs["order"], side):
            bonds.add((pair, pick(attrs["sigma_order"], side), pick(attrs["pi_order"], side)))
    resources = {(node, field, pick(graph.nodes[node][field], side)) for node in atoms for field in FIELDS}
    return (
        atoms,
        bonds,
        {(bond, atom) for bond in bonds for atom in bond[0]},
        resources,
        {(record, record[0]) for record in resources},
    )


def check_completed_coproduct(graph, original):
    """Fresh completion has no source bond to the original host."""
    source, _ = _tuple_endpoint_graphs(graph)
    old = set(original)
    new = set(source) - old
    assert new
    assert not any((u in old) != (v in old) for u, v in source.edges)
    for node, attrs in original.nodes(data=True):
        assert all(source.nodes[node][key] == attrs[key] for key in FIELDS)
    assert set(source.subgraph(old).edges) == set(original.edges)
    for u, v, attrs in original.edges(data=True):
        assert all(source.edges[u, v][key] == attrs[key] for key in ("order", "sigma_order", "pi_order"))
    whole = private_records(graph, 0)
    left = private_records(graph.subgraph(old), 0)
    right = private_records(graph.subgraph(new), 0)
    for union, first, second in zip(whole, left, right):
        assert not first & second
        assert first | second == union


def ground_unit_ports(graph):
    """Choose a separated methyl substituent for each fresh symbolic root."""
    grounded = deepcopy(graph)
    for root in graph.graph["partial_completion"]["attachment_ports"]:
        attrs = grounded.nodes[root]
        attrs.update(
            element=("C", "C"),
            valence_electrons=(4, 4),
            lone_pairs=(0, 0),
            radical=(0, 0),
            hcount=(3, 3),
            charge=(0, 0),
        )
        for key in ("wildcard_role", "_wildcard_constraint", "owner"):
            attrs.pop(key, None)
    return grounded


def check_private_record_pushouts(graph, rule, mapping):
    """Test both pushouts by exact intersections and unions in every sort."""
    total = dict(mapping)
    total.update(graph.graph["partial_completion"]["introduced_mapping"])
    assert set(total) == set(rule)
    assert len(set(total.values())) == len(total)
    active = set(total.values()) | set(graph.graph["partial_completion"]["attachment_ports"])
    pairs = {frozenset((total[u], total[v])) for u, v in rule.edges}
    for root in graph.graph["partial_completion"]["attachment_ports"]:
        pairs.update(frozenset((root, neighbor)) for neighbor in graph[root])
    L, R = (private_records(graph, side, active, pairs) for side in (0, 1))
    G, H = (private_records(graph, side) for side in (0, 1))
    K = tuple(left & right for left, right in zip(L, R))
    D = tuple(source - (left - interface) for source, left, interface in zip(G, L, K))
    for atoms, bonds, incidences, resources, ownership in (L, K, R, G, D, H):
        assert all(bond in bonds and atom in atoms for bond, atom in incidences)
        assert all(record in resources and atom in atoms for record, atom in ownership)
        assert all((bond, atom) in incidences for bond in bonds for atom in bond[0])
        assert all((record, record[0]) in ownership for record in resources)
    for left, right, source, target in zip(L, R, G, H):
        interface = left & right
        context = source - (left - interface)
        assert left & context == interface
        assert right & context == interface
        assert left | context == source
        assert right | context == target


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "host,invert,absolute",
    [
        ("CBr", False, False),
        ("CO", True, False),
        ("CBr.O", False, False),
        ("CBr", False, True),
    ],
)
def test_partial_application_factors_through_completion_and_grounded_dpo(
    library,
    backend,
    host,
    invert,
    absolute,
):
    rule = SynRule.from_smart(SN2, format="tuple", implicit_h=not absolute)
    engine = WildcardEngine(host, rule, invert=invert, backend=backend, library_path=library, dedup_its=False)
    assert engine.its_list
    for graph, mapping in zip(engine.its_list, engine.mappings):
        assert graph.graph["partial_completion"]["introduced_mapping"]
        domain = set(mapping)
        for u, v, attrs in engine.rule.rc.raw.edges(data=True):
            if pick(attrs["order"], 0):
                assert (u in domain) == (v in domain)
        # Ground first, so all resource records in the diagrams are concrete.
        grounded = ground_unit_ports(graph)
        check_completed_coproduct(grounded, engine.graph.raw)
        validate_lewis_endpoints(grounded, engine.rule.rc.raw)
        check_private_record_pushouts(grounded, engine.rule.rc.raw, mapping)
