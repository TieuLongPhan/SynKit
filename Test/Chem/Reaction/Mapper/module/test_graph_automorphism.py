from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.graph.automorphism import node_orbits


def test_automorphism_node_orbits_cover_all_nodes():
    graph = smiles2lgp("CC>>CC", add_Hs=False)[0]

    covered = set().union(*node_orbits(graph))

    assert covered == set(range(len(graph.labels)))
