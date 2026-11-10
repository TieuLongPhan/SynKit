"""Independent orbit evidence, replayable witnesses and defective-gate regressions."""

from itertools import product
import json

import pytest

from Experiment.Stereo.reference.fixtures import fixtures
from Experiment.Stereo.reference.model import (
    assignment_key, formal_states, proper_group, transport,
)
from Experiment.Stereo.reference.orbits import partition_agreement, reference_partition
from synkit.Graph.Stereo.orbits import SHAPE_DEFINITIONS


def _fixture(name):
    return next(fixture for fixture in fixtures() if fixture.name == name)


@pytest.mark.parametrize("family,order", [
    ("tetrahedral", 12), ("square_planar", 8), ("trigonal_bipyramidal", 6), ("octahedral", 24),
    ("planar_bond", 4), ("atrop_bond", 4), ("cumulene_axis", 4), ("extended_cis_trans", 4),
])
def test_geometry_groups_agree_without_supplying_oracle_truth(family, order):
    """Compare independently constructed geometries with the implementation tables."""
    reference = set(proper_group(family))
    assert len(reference) == order
    assert reference == {element.image for element in SHAPE_DEFINITIONS[family].preserving_group.elements}


@pytest.mark.parametrize("name,classes,fixed,pairs", [
    ("tetrahedral:two:meso", 3, 1, 1),
    ("tetrahedral:two:asymmetric", 4, 0, 2),
    ("framework:coupled:same", 2, 0, 1),
    ("framework:coupled:opposite", 1, 1, 0),
    ("plane:reflection_symmetric", 1, 1, 0),
])
def test_global_symmetry_and_mirror_counts_from_direct_transport(name, classes, fixed, pairs):
    """Cover carrier exchange, framework coupling and plane reflection collapse."""
    fixture = _fixture(name)
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    report = reference_partition(fixture.graph, choices, atom_color="color", bond_color="color")
    assert report["status"] == "complete"
    assert len(report["partition"]) == classes
    assert len(report["mirror_classes"]["mirror_fixed_classes"]) == fixed
    assert len(report["mirror_classes"]["enantiomer_pairs"]) == pairs


def test_every_merge_has_a_constitutional_map_replaying_its_assignment():
    """Check the actual stored witness for every assignment in a meso orbit."""
    fixture = _fixture("tetrahedral:two:meso")
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    report = reference_partition(fixture.graph, choices, atom_color="color", bond_color="color")
    assignments = tuple(product(*(range(len(local)) for local in choices)))
    assert len(report["merge_witnesses"]) == len(assignments)
    for witness in report["merge_witnesses"]:
        mapping = dict(witness["atom_mapping"])
        assert set(mapping) == set(mapping.values()) == set(fixture.graph)
        assert all(fixture.graph.nodes[node]["color"] == fixture.graph.nodes[mapping[node]]["color"]
                   for node in fixture.graph)
        assert all(fixture.graph.has_edge(mapping[left], mapping[right]) for left, right in fixture.graph.edges)
        source = tuple(local[index] for local, index in zip(choices, witness["source_assignment"], strict=True))
        target = tuple(local[index] for local, index in zip(choices, witness["target_assignment"], strict=True))
        assert assignment_key(tuple(transport(value, mapping) for value in source)) == assignment_key(target)


@pytest.mark.parametrize("options", [
    {"max_assignments": 1}, {"max_automorphisms": 1}, {"timeout_seconds": 0},
])
def test_an_incomplete_reference_cannot_pass_a_partition_gate(options):
    """Prevent a capped or interrupted reference from becoming correctness evidence."""
    fixture = _fixture("tetrahedral:symmetric")
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    report = reference_partition(fixture.graph, choices, atom_color="color", bond_color="color", **options)
    assert report["status"] == "oracle_incomplete"
    assert not partition_agreement(report, ((0,), (1,)), ("constant", "constant"))


def test_missing_formal_state_refuses_instead_of_validating_a_smaller_product():
    """The oracle checks exhaustive local choices independently of their supplier."""
    fixture = _fixture("tetrahedral:distinct")
    report = reference_partition(fixture.graph, ((fixture.seeds[0],),), atom_color="color", bond_color="color")
    assert report["status"] == "oracle_incomplete"
    assert "not_independently_exhaustive" in report["reason"]


@pytest.mark.parametrize("left,right,classes", [
    (1, True, 2), (0.0, -0.0, 2), (float("nan"), float("nan"), 1),
    ({"x": 1, "y": 2}, {"y": 2, "x": 1}, 1),
])
def test_constitutional_colors_preserve_types_and_normalize_declared_containers(left, right, classes):
    """Python bool/int equality must not admit an invalid constitutional map."""
    fixture = _fixture("tetrahedral:distinct")
    graph = fixture.graph.copy()
    graph.nodes[1]["color"], graph.nodes[2]["color"] = left, right
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    report = reference_partition(graph, choices, atom_color="color", bond_color="color")
    assert report["status"] == "complete"
    assert len(report["partition"]) == classes
    from synkit.Graph.Stereo import canonicalize_configured_stereograph
    from Experiment.Stereo.reference.model import to_descriptor

    labels = [canonicalize_configured_stereograph(
        graph, (to_descriptor(value),), atom_color="color", bond_color="color",
        enumerate_automorphism_group=False,
    ).canonical_code for value in choices[0]]
    assert partition_agreement(report, report["assignments"], labels)


def test_partition_gate_accepts_a_lossless_json_report_round_trip():
    """Serialized list containers retain the exact partition and assignment IDs."""
    fixture = _fixture("tetrahedral:distinct")
    choices = tuple(formal_states(seed) for seed in fixture.seeds)
    report = reference_partition(fixture.graph, choices, atom_color="color", bond_color="color")
    recovered = json.loads(json.dumps(report))
    assert partition_agreement(recovered, recovered["assignments"], ("first-complete-code", "second-complete-code"))
