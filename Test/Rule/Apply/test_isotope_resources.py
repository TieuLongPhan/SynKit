"""Isotope identity, boundary inventory and natural-square regressions."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LLGError,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import (
    DPOError,
    DPOIssueCode,
    EnvironmentToken,
    RuleSpan,
    apply_dpo,
)
from synkit.Rule.Compose import RuleOverlap, compose_rules, reverse_rule


def _oxygen(node, isotope, lone_pairs=3):
    graph = nx.Graph()
    graph.add_node(
        node,
        element="O",
        isotope=isotope,
        aromatic=False,
        hcount=0,
        radical=0,
        lone_pairs=lone_pairs,
        valence_electrons=6,
    )
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(graph), ISOTOPE_ELECTRON_LLG_SCHEMA
    )


def _exchange(left=18, right=16, *, token=None, boundary="open"):
    return RuleSpan.from_mapping(
        _oxygen(1, left),
        _oxygen(2, right),
        {},
        boundary=boundary,
        environment=token,
    )


def test_isotope_identity_is_preserved_under_maps_and_carrier_renaming():
    source = _oxygen(1, 18)
    assert source.is_isomorphic(source.relabel({1: "renamed"}))
    assert not source.is_isomorphic(_oxygen(1, 16))
    with pytest.raises(DPOError, match="identity labels"):
        RuleSpan.from_mapping(source, _oxygen(2, 16), {1: 2})


@pytest.mark.parametrize("mass", [-1, 18.5, True, "18"])
def test_invalid_isotope_mass_is_refused(mass):
    with pytest.raises(LLGError, match="isotope mass"):
        _oxygen(1, mass)


def test_closed_isotope_exchange_cannot_hide_behind_element_balance():
    with pytest.raises(DPOError) as error:
        _exchange(boundary="closed")
    assert DPOIssueCode.CLOSED_ISOTOPE in {item.code for item in error.value.issues}


def test_open_isotope_exchange_requires_the_exact_resolved_token():
    with pytest.raises(DPOError) as error:
        _exchange(token=EnvironmentToken("element-only"))
    assert error.value.issues[0].code is DPOIssueCode.OPEN_TOKEN_MISMATCH
    token = EnvironmentToken(
        "oxygen exchange", isotope_delta=((("O", 16), 1), (("O", 18), -1))
    )
    rule = _exchange(token=token)
    application = apply_dpo(rule, rule.left, {1: 1})
    assert application.certificate.replay().valid
    assert rule.resource_delta.element_delta == ()
    assert rule.resource_delta.isotope_delta == token.isotope_delta
    assert reverse_rule(rule).environment.isotope_delta == (
        (("O", 16), -1),
        (("O", 18), 1),
    )


def test_open_composition_transports_and_cancels_isotope_tokens():
    first = _exchange(
        token=EnvironmentToken(
            "18 to 16", isotope_delta=((("O", 16), 1), (("O", 18), -1))
        )
    )
    second = _exchange(
        16,
        18,
        token=EnvironmentToken(
            "16 to 18", isotope_delta=((("O", 16), -1), (("O", 18), 1))
        ),
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {2: 1})
    result = compose_rules(first, second, overlap)
    assert result.rule.environment.isotope_delta == ()
    assert result.certificate.replay().valid


def test_isotope_schema_recomputes_derived_electron_state_after_rewrite():
    left, right = _oxygen(1, 18, 3), _oxygen(2, 18, 2)
    rule = RuleSpan.from_mapping(left, right, {1: 2})
    application = apply_dpo(rule, left, {1: 1})
    assert application.result.node_labels(1)["charge"] == 2
    assert application.result.node_labels(1)["isotope"] == 18
    assert application.certificate.replay().valid
    forged = application.result.to_networkx()
    forged.nodes[1]["charge"] = 0
    invalid_result = LewisLabelledGraph.from_networkx(
        forged, ISOTOPE_ELECTRON_LLG_SCHEMA
    )
    assert not replace(application.certificate, result=invalid_result).replay().valid
