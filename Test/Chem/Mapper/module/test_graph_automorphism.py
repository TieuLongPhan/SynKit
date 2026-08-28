from synkit.Chem.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Mapper.graph.automorphism import (
    bounded_automorphism_permutations,
    node_orbits,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def test_automorphism_node_orbits_cover_all_nodes():
    graph = smiles2lgp("CC>>CC", add_Hs=False)[0]

    covered = set().union(*node_orbits(graph))

    assert covered == set(range(len(graph.labels)))


def test_symmetry_node_properties_refine_safe_subgroup():
    graph = LabeledGraph({0: {}, 1: {}}, [6, 6])
    graph.set_prop("atomic numbers", [6, 6])
    graph.set_prop("hcounts", [4, 0])
    graph.set_prop("charges", [0, 0])

    plain, _ = bounded_automorphism_permutations(graph, limit=8)
    hydrogen_refined, _ = bounded_automorphism_permutations(
        graph,
        limit=8,
        node_properties=("hcounts", "charges"),
    )

    assert len(plain) == 2
    assert hydrogen_refined == ((0, 1),)
