"""Composition proofs must certify the applications actually retained."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Rule.Apply import RuleSpan
from synkit.Rule.Compose import RuleOverlap, compose_rules


def composition():
    schema = LabelSchema(node_identity=("kind",), node_state=("state",))

    def graph(state):
        value = nx.Graph()
        value.add_node(1, kind="X", state=state)
        return LewisLabelledGraph.from_networkx(value, schema)

    first = RuleSpan.from_mapping(graph(0), graph(1), {1: 1})
    second = RuleSpan.from_mapping(graph(1), graph(2), {1: 1})
    return compose_rules(
        first, second, RuleOverlap.from_mapping(first.right, second.left, {1: 1})
    )


@pytest.mark.parametrize(
    "field", ["first_application", "second_application", "outer_application"]
)
def test_application_result_cannot_disagree_with_its_certificate(field):
    c = composition().certificate
    wrong = replace(getattr(c, field), result=c.first.left)
    assert not replace(c, **{field: wrong}).replay().valid


def test_stripping_the_sequential_condition_invalidates_the_composition_certificate():
    c = composition().certificate
    stripped = replace(c.composite, condition=None)
    outer = replace(
        c.outer_application,
        certificate=replace(c.outer_application.certificate, rule=stripped),
    )
    assert not replace(c, composite=stripped, outer_application=outer).replay().valid


def test_wrong_second_match_is_rejected():
    c = composition().certificate
    assert not replace(c, second_match=((1, "outside"),)).replay().valid
