import networkx as nx
import pytest

from synkit.Chem.Mapper import AAMValidator as MapperAAMValidator
from synkit.Chem.Reaction.aam_validator import AAMValidator as LegacyAAMValidator


def test_mapper_aam_validator_is_legacy_compatible():
    assert MapperAAMValidator is LegacyAAMValidator
    assert MapperAAMValidator().strip_unbalanced_maps is True


def _its_graph():
    graph = nx.Graph()
    carbon = ("C", False, 3, 0, ("O",))
    oxygen = ("O", False, 1, 0, ("C",))
    graph.add_node(1, typesGH=(carbon, carbon))
    graph.add_node(2, typesGH=(oxygen, oxygen))
    graph.add_edge(1, 2, order=(1, 2))
    return graph


def test_its_equivalence_ignores_node_identifiers():
    graph = _its_graph()
    relabeled = nx.relabel_nodes(graph, {1: 20, 2: 10})

    assert MapperAAMValidator.check_equivariant_graph([graph, relabeled]) == (
        [(0, 1)],
        1,
    )


@pytest.mark.parametrize(
    "attribute_index, replacement",
    [(0, "N"), (1, True), (2, 2), (3, 1), (4, ("N",))],
)
@pytest.mark.parametrize("side", [0, 1])
def test_its_equivalence_checks_every_attribute_on_both_sides(
    attribute_index, replacement, side
):
    graph = _its_graph()
    changed = graph.copy()
    states = list(changed.nodes[1]["typesGH"])
    attributes = list(states[side])
    attributes[attribute_index] = replacement
    states[side] = tuple(attributes)
    changed.nodes[1]["typesGH"] = tuple(states)

    assert MapperAAMValidator.check_equivariant_graph([graph, changed]) == ([], 0)


def test_its_equivalence_checks_bond_changes():
    graph = _its_graph()
    changed = graph.copy()
    changed.edges[1, 2]["order"] = (1, 1)

    assert MapperAAMValidator.check_equivariant_graph([graph, changed]) == ([], 0)
