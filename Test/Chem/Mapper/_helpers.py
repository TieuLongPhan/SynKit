"""Small graph constructors shared by independent Mapper search oracles."""

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def graph(matrix):
    """Construct a carbon-labeled graph from a symmetric bond matrix."""
    return LabeledGraph(
        {
            i: {j: float(value) for j, value in enumerate(row) if i != j and value}
            for i, row in enumerate(matrix)
        },
        [6] * len(matrix),
    )
