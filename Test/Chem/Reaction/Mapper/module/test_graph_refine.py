from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.graph.refine import wl_node_colors


def test_refine_wl_node_colors_matches_node_count():
    graph = smiles2lgp("CCO>>CC=O", add_Hs=False)[0]

    colors = wl_node_colors(graph.graph, len(graph.labels))

    assert len(colors) == len(graph.labels)
