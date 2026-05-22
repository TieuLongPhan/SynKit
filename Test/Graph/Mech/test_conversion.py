import networkx as nx

from synkit.Graph.Mech.conversion import (
    extract_atom_maps_from_smiles,
    typed_convert_arrow_code,
)


class CountingGraph(nx.Graph):
    def __init__(self):
        super().__init__()
        self.nodes_calls = 0

    def nodes(self, *args, **kwargs):
        self.nodes_calls += 1
        return super().nodes(*args, **kwargs)


def test_extract_atom_maps_from_smiles():
    assert extract_atom_maps_from_smiles("[CH:10][N+:61]") == [10, 61]


def test_typed_convert_arrow_code_reuses_atom_map_index():
    its = CountingGraph()
    its.add_node("a", atom_map=1)
    its.add_node("b", atom_map=2)
    its.add_edge("a", "b", order=(1.0, 2.0))

    assert typed_convert_arrow_code("1=1,2;1,2=1", its) == [
        ["LP-/Pi+", [1], [1, 2]],
        ["Sigma-/LP+", [1, 2], [1]],
    ]
    assert its.nodes_calls == 1
