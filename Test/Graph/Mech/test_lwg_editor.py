import networkx as nx

from synkit.Graph.Mech.lwg_editor import LWGEditor
from synkit.Graph.Mech.lwg_ops import change_edge_order, normalize_lwg_graph


def test_normalize_lwg_graph_derives_sigma_pi_from_kekule_order():
    graph = nx.Graph()
    graph.add_node(1, atom_map=1)
    graph.add_node(2, atom_map=2)
    graph.add_edge(1, 2, kekule_order=2.0)

    normalized = normalize_lwg_graph(graph)

    assert normalized.edges[1, 2]["sigma_order"] == 1.0
    assert normalized.edges[1, 2]["pi_order"] == 1.0
    assert normalized.edges[1, 2]["order"] == 2.0


def test_change_edge_order_decrements_pi_and_increments_target_pi():
    graph = nx.Graph()
    for atom_map in range(1, 5):
        graph.add_node(atom_map, atom_map=atom_map)
    graph.add_edge(1, 2, sigma_order=1.0, pi_order=1.0)
    graph.add_edge(2, 3, sigma_order=1.0, pi_order=0.0)
    normalize_lwg_graph(graph, in_place=True)

    minus = change_edge_order(graph, [1, 2], field="pi_order", delta=-1.0)
    plus = change_edge_order(graph, [2, 3], field="pi_order", delta=1.0)

    assert minus.previous_value == 1.0
    assert minus.new_value == 0.0
    assert plus.previous_value == 0.0
    assert plus.new_value == 1.0
    assert graph.edges[1, 2]["kekule_order"] == 1.0
    assert graph.edges[2, 3]["kekule_order"] == 2.0


def test_lwg_editor_noop_reaction_matches_product():
    result = LWGEditor().apply("[CH3:1][Br:2]>>[CH3:1][Br:2]", [])

    assert result.structural_match
    assert result.charge_match
    assert result.smiles_match
    assert result.matches_product


def test_lwg_editor_lp_sigma_edits_lone_pair_and_charge_converges():
    result = LWGEditor().apply(
        "[OH-:1].[CH3+:2]>>[CH3:2][OH:1]",
        [["LP-/Sigma+", [1], [1, 2]]],
    )

    assert result.structural_match
    assert result.charge_match
    assert result.matches_product
    assert result.final_graph.has_edge(1, 2)
    assert result.final_graph.edges[1, 2]["sigma_order"] == 1.0
    assert result.final_graph.nodes[1]["lone_pairs"] == 2
    assert result.final_graph.nodes[1]["charge"] == 0
    assert result.final_graph.nodes[2]["charge"] == 0


def test_lwg_editor_two_step_substitution_uses_local_charge_deltas():
    result = LWGEditor().apply(
        "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]",
        [
            ["LP-/Sigma+", [1], [1, 2]],
            ["Sigma-/LP+", [2, 3], [3]],
        ],
    )

    first, second = result.intermediates

    assert result.matches_product
    assert first.nodes[1]["charge"] == 1
    assert first.nodes[2]["charge"] == -1
    assert first.nodes[3]["charge"] == 0
    assert first.nodes[1]["lone_pairs"] == 0
    assert first.has_edge(1, 2)
    assert first.has_edge(2, 3)

    assert second.nodes[1]["charge"] == 1
    assert second.nodes[2]["charge"] == 0
    assert second.nodes[3]["charge"] == -1
    assert second.nodes[3]["lone_pairs"] == 4
    assert second.has_edge(1, 2)
    assert not second.has_edge(2, 3)

    assert result.step_reports[0].charge_changes[0].delta == 2
    assert [change.delta for change in result.step_reports[0].charge_changes[1:]] == [
        -1,
        -1,
    ]
    assert [change.delta for change in result.step_reports[1].charge_changes] == [
        1,
        1,
        -2,
    ]
