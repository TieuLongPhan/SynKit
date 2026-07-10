import networkx as nx

from synkit.Graph.Mech.conversion import (
    ef_smirks_to_epd,
    epd_to_ef_smirks,
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


def test_ef_smirks_round_trip_completes_aam_and_preserves_flow_code():
    ef_smirks = (
        "C[C:10](C)(CO[N+]([O-])=O)[CH2:20][O:21]>>"
        "C[C:10](CO[N+]([O-])=O)C.[CH2:20]=[O:21] "
        "21-20,21;10,20-20,21;10,20-10"
    )

    result = ef_smirks_to_epd(ef_smirks)

    assert result["complete_aam"] == result["expanded_rsmi"]
    assert result["epd_lw"] == [
        ["LP-/Pi+", [21], [20, 21]],
        ["Sigma-/Pi+", [10, 20], [20, 21]],
        ["Sigma-/LP+", [10, 20], [10]],
    ]
    assert epd_to_ef_smirks(result["complete_aam"], result["epd"]) == (
        f"{result['complete_aam']} 21-20,21;10,20-20,21;10,20-10"
    )


def test_ef_smirks_conversion_is_exposed_from_io_conversion():
    from synkit.IO import ef_smirks_to_epd
    from synkit.IO.conversion import epd_to_ef_smirks

    result = ef_smirks_to_epd("[CH3:1][OH:2]>>[CH3:1][O-:2] 2-1,2")

    assert result["epd"] == [["LP-/B+", [2], [1, 2]]]
    assert epd_to_ef_smirks(result["complete_aam"], result["epd"]).endswith("2-1,2")
