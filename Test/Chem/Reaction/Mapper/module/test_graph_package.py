import importlib


def test_graph_package_exports_graph_helpers():
    module = importlib.import_module("synkit.Chem.Reaction.Mapper.graph")

    assert module.LabeledGraph is not None
    assert callable(module.wl_node_colors)
    assert callable(module.block_cut_tree)
