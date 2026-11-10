"""Absolute DPO representatives retain the resources hidden by relative rules."""

from copy import deepcopy

import networkx as nx
import pytest

from synkit.Graph.Morphism import WildcardConstraint, WildcardRole
from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Rule.Apply import (
    DPOError,
    DPOIssueCode,
    apply_dpo,
    rule_from_its,
    rule_from_synrule,
)
from synkit.Rule.Compose import compose_synrules

EXAMPLES = (
    ("[OH:1][CH3:2]>>[O:1]=[CH2:2]", (2, 2)),
    ("[NH2:1][CH3:2]>>[NH:1]=[CH2:2]", (1, 1)),
    ("[OH-:1]>>[OH2:1]", (3, 2)),
)


@pytest.mark.parametrize("reaction,lone_pairs", EXAMPLES)
def test_absolute_adaptation_restores_heteroatom_resources_without_mutation(reaction, lone_pairs):
    source = rsmi_to_its(reaction, format="tuple")
    expected = rule_from_its(source, electron_complete=True)
    rule = SynRule(source, format="tuple", implicit_h=False)
    before = tuple(deepcopy(graph) for graph in (rule.rc.raw, rule.left.raw, rule.right.raw))

    adapted = rule.to_rule_span()

    assert adapted.left.is_isomorphic(expected.left)
    assert adapted.right.is_isomorphic(expected.right)
    assert (
        adapted.left.node_labels(1)["lone_pairs"],
        adapted.right.node_labels(1)["lone_pairs"],
    ) == lone_pairs
    for original, current in zip(before, (rule.rc.raw, rule.left.raw, rule.right.raw)):
        assert nx.utils.graphs_equal(original, current)
    application = apply_dpo(adapted, expected.left, {node: node for node in expected.left.node_ids})
    assert application.result.is_isomorphic(expected.right)
    assert application.certificate.replay().valid


@pytest.mark.parametrize("reaction,lone_pairs", EXAMPLES)
def test_composition_and_reverse_retain_the_absolute_representative(reaction, lone_pairs):
    first = SynRule.from_smart(reaction, format="tuple", implicit_h=False)
    second = first.reversed()
    overlap = {node: node for node in first.right.raw}

    composition = compose_synrules(first, second, overlap)

    assert composition.certificate.replay().valid
    assert composition.rule.left.is_isomorphic(composition.rule.right)
    assert composition.rule.left.node_labels(1)["lone_pairs"] == lone_pairs[0]
    assert composition.certificate.first_application.result.node_labels(1)["lone_pairs"] == lone_pairs[1]


@pytest.mark.parametrize("electron_complete", (False, True))
@pytest.mark.parametrize("contract", ("bare", "role", "predicate", "rc_predicate", "rc_wildcard", "rc_mixed"))
def test_direct_adapter_refuses_to_reinterpret_symbolic_wildcards(electron_complete, contract):
    if contract == "bare":
        rule = SynRule.from_smart("[*:1][CH3:2]>>[*:1][CH2:2]", format="tuple", implicit_h=False)
    else:
        rule = SynRule.from_smart("[CH3:1][CH3:2]>>[CH2:1]=[CH2:2]", format="tuple", implicit_h=False)
        endpoints = (
            (rule.rc.raw,)
            if contract in {"rc_predicate", "rc_wildcard", "rc_mixed"}
            else (rule.left.raw, rule.right.raw)
        )
        for endpoint in endpoints:
            if contract == "role":
                endpoint.nodes[1]["wildcard_role"] = "query_atom"
            elif contract in {"rc_wildcard", "rc_mixed"}:
                endpoint.nodes[1]["element"] = ("*", "*") if contract == "rc_wildcard" else ("C", "*")
            else:
                endpoint.nodes[1]["_wildcard_constraint"] = WildcardConstraint(WildcardRole.QUERY_ATOM)

    with pytest.raises(DPOError) as error:
        rule_from_synrule(rule, electron_complete=electron_complete)

    assert error.value.issues[0].code is DPOIssueCode.SYMBOLIC_RULE_UNSUPPORTED


@pytest.mark.parametrize("electron_complete", (False, True))
def test_discarded_common_hydrogen_is_not_treated_as_an_absolute_endpoint(
    electron_complete,
):
    rule = SynRule.from_smart("[CH3:1][OH:2]>>[CH2:1]=[O:2]", format="tuple", implicit_h=True)
    assert rule.left.raw.nodes[1]["hcount"] == 1

    with pytest.raises(DPOError) as error:
        rule_from_synrule(rule, electron_complete=electron_complete)

    assert error.value.issues[0].code is DPOIssueCode.NORMALIZED_RULE_UNSUPPORTED
    assert "implicit_h=False" in str(error.value)
