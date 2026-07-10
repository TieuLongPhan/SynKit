from synkit.Chem.Reaction.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Reaction.Mapper.graph.blockcut import block_cut_tree


def test_block_cut_tree_contains_original_nodes():
    graph = smiles2lgp("CCC>>CCC", add_Hs=False)[0]

    tree = block_cut_tree(graph.graph, len(graph.labels))

    assert set().union(*tree.blocks) == set(range(len(graph.labels)))
