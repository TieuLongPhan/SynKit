from synkit.Chem.Mapper.exact.core import reduce_to_reference_its_components
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def _graph(edges, hcounts):
    adjacency = {index: {} for index in range(5)}
    for left, right, order in edges:
        adjacency[left][right] = order
        adjacency[right][left] = order
    graph = LabeledGraph(adjacency, [6, 6, 8, 6, 8])
    graph.set_prop("atomic numbers", [6, 6, 8, 6, 8])
    graph.set_prop("hcounts", hcounts)
    graph.set_prop("charges", [0] * 5)
    return graph


def test_reference_its_component_reduction_and_lift():
    # Atoms 0--2 react; disconnected 3--4 is unchanged spectator context.
    reactant = _graph([(0, 1, 1), (1, 2, 1), (3, 4, 1)], [3, 2, 1, 3, 1])
    product = _graph([(0, 1, 2), (1, 2, 1), (3, 4, 1)], [2, 1, 1, 3, 1])
    pair = [reactant, product]

    reduced = reduce_to_reference_its_components(pair, list(range(5)))

    assert reduced.scope == "reference_active_its_component_space"
    assert reduced.changed_atoms == (0, 1)
    assert reduced.reactant_atoms == (0, 1, 2)
    assert reduced.product_atoms == (0, 1, 2)
    assert reduced.inactive_atoms == (3, 4)
    assert reduced.reference_mapping == (0, 1, 2)
    assert reduced.heavy_distance == chemical_distance(
        reduced.lgp, reduced.reference_mapping, binary=False
    )
    assert reduced.heavy_distance == chemical_distance(
        pair, list(range(5)), binary=False
    )
    assert reduced.hydrogen_distance == 2
    assert reduced.lgp[0].props["hcounts"] == [3, 2, 1]
    assert reduced.lift([0, 1, 2]) == list(range(5))


def test_hcount_only_change_selects_its_component():
    reactant = _graph([(0, 1, 1), (1, 2, 1), (3, 4, 1)], [3, 2, 1, 3, 1])
    product = _graph([(0, 1, 1), (1, 2, 1), (3, 4, 1)], [2, 3, 1, 3, 1])

    reduced = reduce_to_reference_its_components([reactant, product], list(range(5)))

    assert reduced.changed_atoms == (0, 1)
    assert reduced.reactant_atoms == (0, 1, 2)
    assert reduced.heavy_distance == 0
    assert reduced.hydrogen_distance == 2


def test_unchanged_reaction_has_empty_active_core_and_identity_lift():
    graph = _graph([(0, 1, 1), (1, 2, 1), (3, 4, 1)], [3, 2, 1, 3, 1])

    reduced = reduce_to_reference_its_components([graph, graph.copy()], list(range(5)))

    assert reduced.reactant_atoms == ()
    assert reduced.changed_atoms == ()
    assert reduced.lift([]) == list(range(5))
