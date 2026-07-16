import networkx as nx

from synkit.Graph.Stereo import StereoChange, TetrahedralStereo
from synkit.Graph.utils import print_graph_attributes


def test_print_graph_attributes_includes_its_stereo_metadata(capsys):
    graph = nx.Graph()
    graph.add_node(2, element=("C", "C"), atom_map=(2, 2))
    graph.add_node(4, element=("Cl", "Cl"), atom_map=(4, 4))
    graph.add_node(5, element=("O", "O"), atom_map=(5, 5))
    graph.add_edge(2, 4, order=(1.0, 0.0))
    graph.add_edge(2, 5, order=(0.0, 1.0))

    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1, "rdkit")
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1, "rdkit")
    graph.graph["stereo_descriptors"] = {
        "reactant": {"atom:2": before},
        "product": {"atom:2": after},
    }
    graph.graph["stereo_changes"] = {"atom:2": StereoChange("INVERTED", before, after)}
    graph.graph["stereo_outcomes"] = {}
    graph.graph["stereo_branch_weight"] = 1.0

    print_graph_attributes(graph)
    output = capsys.readouterr().out

    assert "Graph-level attributes" in output
    assert "stereo_descriptors:" in output
    assert "reactant:" in output and "product:" in output
    assert "center=2 ordered_refs=(1, 3, 4, '@H:2') parity=-1" in output
    assert "atom:2: INVERTED" in output
    assert "reference_delta: removed=[4] added=[5]" in output
    assert "stereo_outcomes:\n    (none)" in output
    assert "stereo_branch_weight:\n    1.0" in output


def test_print_graph_attributes_reports_absent_graph_metadata(capsys):
    graph = nx.Graph()
    graph.add_node(1, element="C")

    print_graph_attributes(graph)
    output = capsys.readouterr().out

    assert "Graph-level attributes:\n  (none)" in output
