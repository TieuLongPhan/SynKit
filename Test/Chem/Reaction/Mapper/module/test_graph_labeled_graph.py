from synkit.Chem.Reaction.Mapper.graph.labeled_graph import LabeledGraph


def test_labeled_graph_builds_label_index():
    graph = LabeledGraph({0: {1: 1}, 1: {0: 1}}, [6, 6])

    assert graph.label2idxs[6] == [0, 1]
    assert graph.copy().labels == graph.labels
