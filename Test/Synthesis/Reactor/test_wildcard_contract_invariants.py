"""Persistent variables retain predicates and every authoritative bond coordinate."""

from copy import deepcopy

import pytest

from synkit.Graph.Morphism import WildcardConstraint
from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor
from synkit.Synthesis.Reactor.matching.wildcard import validate_wildcard_rule

IDENTITY = "[CH3:1][*:2]>>[CH3:1][*:2]"


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    return str(build_native_reactor(tmp_path_factory.mktemp("wildcard-invariants")))


@pytest.mark.parametrize(
    "field,value",
    [
        ("elements", {"N"}),
        ("charges", {1}),
        ("radicals", {1}),
        ("bond_orders", {2}),
        ("side", "product"),
        ("capacity", 0),
        ("resource_budget", 0),
        ("mapped_identity", 17),
        ("materialization", "symbolic"),
        ("wildcard_role", "attachment_port"),
    ],
)
def test_endpoint_predicate_changes_are_not_silently_ignored(field, value):
    rule = SynRule.from_smart(IDENTITY, format="tuple", implicit_h=False, canon=False)
    rule.right.raw.nodes[2][field] = value
    before = deepcopy(rule.right.raw.nodes[2])
    for _ in range(2):
        engine = WildcardEngine("CC", rule, partial=False)
        with pytest.raises(ValueError, match="shared endpoint contract"):
            _ = engine.mappings
        assert engine._rule is None
        assert engine._its is None
    assert rule.right.raw.nodes[2] == before


def test_disjoint_endpoint_domains_are_rejected_instead_of_using_left_only():
    rule = SynRule.from_smart(IDENTITY, format="tuple", implicit_h=False, canon=False)
    rule.left.raw.nodes[2]["elements"] = {"C"}
    rule.right.raw.nodes[2]["elements"] = {"N"}
    with pytest.raises(ValueError, match="shared endpoint contract"):
        validate_wildcard_rule(rule.left.raw, rule.right.raw)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("substrate,accepted", [("CN", True), ("CC", False), ("CO", False)])
def test_explicit_predicate_meet_is_retained_by_both_backends(native_library, backend, substrate, accepted):
    left = WildcardConstraint("query_atom", elements={"C", "N"})
    right = WildcardConstraint("query_atom", elements={"N", "O"})
    result = left.intersect(right)
    assert result.valid and result.constraint.elements == frozenset({"N"})
    engine = WildcardEngine(
        substrate,
        IDENTITY,
        partial=False,
        backend=backend,
        library_path=native_library if backend == "cpp" else None,
        wildcards={2: result.constraint},
    )
    assert bool(engine.mappings) is accepted
    assert bool(engine.smarts_list) is accepted


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("coordinate", ["sigma_order", "pi_order", "kekule_order"])
def test_zero_total_bond_delta_cannot_hide_boundary_edits(backend, coordinate):
    graph = rsmi_to_its(IDENTITY, format="tuple")
    graph.edges[1, 2][coordinate] = (1, 0)
    rule = SynRule(graph, canon=False, implicit_h=False)
    engine = WildcardEngine("CC", rule, partial=False, backend=backend)
    # Rule validation precedes loading a native library or enumerating matches.
    with pytest.raises(ValueError, match="read-only"):
        _ = engine.mappings
    assert engine._its is None and engine._native_cache is None


@pytest.mark.parametrize("field", ["charge", "hcount", "lone_pairs", "radical", "aromatic"])
def test_all_persistent_variable_atom_states_are_read_only(field):
    graph = rsmi_to_its(IDENTITY, format="tuple")
    graph.nodes[2][field] = (0, 1)
    engine = WildcardEngine("CC", SynRule(graph, canon=False, implicit_h=False), partial=False)
    with pytest.raises(ValueError, match="read-only"):
        _ = engine.mappings


def test_unchanged_boundary_bonds_can_use_implicit_default_coordinates():
    rule = SynRule.from_smart(IDENTITY, format="tuple", implicit_h=False, canon=False)
    for field in ("sigma_order", "pi_order", "kekule_order"):
        rule.right.raw.edges[1, 2].pop(field)
    validate_wildcard_rule(rule.left.raw, rule.right.raw)
