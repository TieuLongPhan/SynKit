from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.graph.synkit_adapter import graph_to_nx


def test_synkit_adapter_converts_mapper_graph_to_networkx():
    graph = smiles2lgp("CC>>CC", add_Hs=False)[0]

    nx_graph = graph_to_nx(graph)

    assert nx_graph.number_of_nodes() == len(graph.labels)
    assert nx_graph.number_of_edges() == 1
