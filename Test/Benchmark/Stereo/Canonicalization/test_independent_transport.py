"""Validate rooted transport truth against exhaustive maps and witness replay."""

import networkx as nx
import pytest
from dataclasses import replace

from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
from Experiment.Stereo.Canonicalization.independent_model import formal_states, state_key, transport
from Experiment.Stereo.Canonicalization.independent_mutations import run_mutations
from Experiment.Stereo.Canonicalization.independent_orbits import color_signature, reference_partition
from Experiment.Stereo.Canonicalization.independent_transport import _Queries

SINGLE = tuple(value for value in fixtures() if len(value.seeds) == 1 and value.seeds[0].family != "framework")
DIRECT = "direct_single_carrier_transport"


def _report(fixture, method, **options):
    return reference_partition(fixture.graph, (formal_states(fixture.seeds[0]),),
                               atom_color="color", bond_color="color", method=method, **options)


def test_real_configuration_erasure_and_locus_omission_fail_the_direct_gate():
    report = run_mutations(single_carrier_method=DIRECT)
    assert report["passed"]
    assert len(report["records"]) == 7
    assert all(value["rejected"] for value in report["records"])


@pytest.mark.parametrize("fixture", SINGLE, ids=lambda value: value.name)
def test_direct_partition_and_mirror_equal_all_exhaustive_maps(fixture):
    """Cover all ten local families, virtual H/LP and symmetric plane reflections."""
    old = _report(fixture, "exhaustive_constitutional_automorphisms")
    new = _report(fixture, DIRECT)
    assert old["status"] == new["status"] == "complete", new
    assert new["partition"] == old["partition"]
    assert new["mirror_classes"] == old["mirror_classes"]
    assert new["mirror_assignment_images"] == old["mirror_assignment_images"]
    assert new["constitutional_automorphisms"] is None
    assert not new["automorphism_group_enumerated"]
    assert new["composition"]["failures"] == new["mirror_commutation"]["failures"] == 0
    choices = formal_states(fixture.seeds[0])
    assert len(new["merge_witnesses"]) == len(choices)
    for witness in new["merge_witnesses"]:
        mapping = dict(witness["atom_mapping"])
        assert set(mapping) == set(mapping.values()) == set(fixture.graph)
        assert all(color_signature(fixture.graph.nodes[node].get("color")) ==
                   color_signature(fixture.graph.nodes[image].get("color")) for node, image in mapping.items())
        assert all(fixture.graph.has_edge(mapping[a], mapping[b]) and
                   color_signature(data.get("color")) ==
                   color_signature(fixture.graph.edges[mapping[a], mapping[b]].get("color"))
                   for a, b, data in fixture.graph.edges(data=True))
        source, target = witness["source_assignment"][0], witness["target_assignment"][0]
        assert state_key(transport(choices[source], mapping)) == state_key(choices[target])


@pytest.mark.parametrize("values,classes", [
    ((True, 1), 2), ((0.0, -0.0), 2), (({"a": 1, "b": 2}, {"b": 2, "a": 1}), 1),
])
def test_direct_queries_preserve_original_typed_attributes(values, classes):
    fixture = next(value for value in SINGLE if value.name == "tetrahedral:distinct")
    graph = fixture.graph.copy()
    graph.nodes[1]["color"], graph.nodes[2]["color"] = values
    new = reference_partition(graph, (formal_states(fixture.seeds[0]),),
                              atom_color="color", bond_color="color", method=DIRECT)
    assert new["status"] == "complete"
    assert len(new["partition"]) == classes


def test_rooted_copies_use_a_shared_palette_for_different_anchor_positions():
    """Moving an anchor must allow a real symmetry despite different node order."""
    graph = nx.cycle_graph(6)
    query = _Queries(graph, "color", "color", float("inf"), 1024, 50000)
    for target in range(6):
        mapping = query.query({0: target, 1: (target + 1) % 6})
        assert mapping is not None
        assert mapping[0] == target and mapping[1] == (target + 1) % 6


def test_full_original_edge_colors_decide_rooted_query_truth():
    """Vertex refinement alone cannot approve a map that changes bond types."""
    graph = nx.cycle_graph(4)
    for a, b in graph.edges:
        graph.edges[a, b]["color"] = "red" if frozenset((a, b)) in ({0, 1}, {2, 3}) else "blue"
    query = _Queries(graph, "color", "color", float("inf"), 1024, 50000)
    assert query.query({0: 1, 1: 2}) is None
    assert query.query({0: 2, 1: 3}) is not None


def test_large_trivial_kernel_can_be_decided_without_enumerating_the_group():
    """Independent peripheral swaps should not exhaust a local-orbit truth budget."""
    fixture = next(value for value in SINGLE if value.name == "tetrahedral:distinct")
    graph = fixture.graph.copy()
    # Fourteen attached sibling pairs give at least 2**14 symmetries, all
    # fixing the configured frame. Color-distinct branch roots isolate pairs.
    for index in range(14):
        root, first, second = 100 + index * 3, 101 + index * 3, 102 + index * 3
        graph.add_node(root, color=f"kernel-root:{index}")
        graph.add_nodes_from(((first, {"color": f"kernel-leaf:{index}"}),
                              (second, {"color": f"kernel-leaf:{index}"})))
        graph.add_edges_from(((1, root), (root, first), (root, second)), color="bond")
    choices = (formal_states(fixture.seeds[0]),)
    old = reference_partition(graph, choices, atom_color="color", bond_color="color", max_automorphisms=20)
    new = reference_partition(graph, choices, atom_color="color", bond_color="color", method=DIRECT)
    assert old["status"] == "oracle_incomplete"
    assert "automorphism_limit" in old["reason"]
    assert new["status"] == "complete", new
    assert new["partition"] == (((0,),), ((1,),))
    assert 1 <= new["query_counts"]["isomorphism_map_proposals"] <= 2


@pytest.mark.parametrize("options,reason", [
    ({"max_queries": 0}, "frame_bindings_limit"), ({"max_assignments": 1}, "bounded_carrier"),
    ({"timeout_seconds": 0}, "oracle_timeout"),
])
def test_every_direct_budget_refusal_remains_incomplete(options, reason):
    fixture = next(value for value in SINGLE if value.name == "tetrahedral:distinct")
    report = _report(fixture, DIRECT, **options)
    assert report["status"] == "oracle_incomplete"
    assert reason in report["reason"]


def test_direct_method_refuses_a_joint_carrier_product():
    fixture = next(value for value in fixtures() if value.name == "tetrahedral:two:meso")
    report = reference_partition(fixture.graph, tuple(formal_states(value) for value in fixture.seeds), method=DIRECT)
    assert report["status"] == "oracle_incomplete"
    assert report["reason"] == "direct_transport_requires_one_bounded_carrier"


@pytest.mark.parametrize("family", ("planar_bond", "atrop_bond", "cumulene_axis", "extended_cis_trans"))
def test_terminal_exchange_transports_virtual_references_through_their_owners(family):
    fixture = next(value for value in SINGLE if value.seeds[0].family == family)
    graph = fixture.graph.copy()
    seed = fixture.seeds[0]
    graph.remove_nodes_from((seed.atoms[1], seed.atoms[-1]))
    nx.set_node_attributes(graph, "carbon", "color")
    atoms = (seed.atoms[0], f"@H:{seed.atoms[2]}", *seed.atoms[2:5], f"@H:{seed.atoms[3]}")
    seed = replace(seed, atoms=atoms)
    choices = (formal_states(seed),)
    options = {"atom_color": "color", "bond_color": "color"}
    old = reference_partition(graph, choices, **options)
    new = reference_partition(graph, choices, method=DIRECT, **options)
    assert old["status"] == new["status"] == "complete", new
    assert new["partition"] == old["partition"]
    assert new["mirror_classes"] == old["mirror_classes"]
