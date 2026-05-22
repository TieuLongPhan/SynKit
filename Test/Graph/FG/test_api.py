import pytest

from synkit.Graph.FG import smiles_to_graph_and_functional_groups


def test_smiles_to_graph_and_functional_groups_supports_unmapped_smiles():
    graph, groups = smiles_to_graph_and_functional_groups("CC(=O)O")

    assert tuple(graph.nodes) == (1, 2, 3, 4)
    assert groups == [("carboxylic_acid", (2, 3, 4))]


def test_smiles_to_graph_and_functional_groups_preserves_atom_map_node_ids():
    graph, groups = smiles_to_graph_and_functional_groups(
        "[CH3:10][C:20](=[O:30])[OH:40]"
    )

    assert tuple(graph.nodes) == (10, 20, 30, 40)
    assert groups == [("carboxylic_acid", (20, 30, 40))]


def test_smiles_to_graph_and_functional_groups_rejects_invalid_smiles():
    with pytest.raises(ValueError):
        smiles_to_graph_and_functional_groups("not_smiles")
