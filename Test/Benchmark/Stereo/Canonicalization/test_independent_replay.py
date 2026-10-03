"""Require archived positive maps and mirror relations to replay exactly."""

import json

import pytest

from Experiment.Stereo.Canonicalization.configuration_free_local import _carrier_result
from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
from Experiment.Stereo.Canonicalization.independent_model import to_descriptor
from Experiment.Stereo.Canonicalization.independent_replay import replay_carrier


def _basis(name="tetrahedral:distinct"):
    fixture = next(value for value in fixtures() if value.name == name)
    seed = to_descriptor(fixture.seeds[0])
    report = _carrier_result(source="test", record_id=name, carrier_id=name, carrier_status="fixture",
                             graph=fixture.graph, seed=seed, timeout_seconds=10, internal_colors=True)
    return fixture.graph, seed, json.loads(json.dumps(report))


def test_archived_positive_maps_replay_every_original_configuration():
    graph, seed, report = _basis()
    assert replay_carrier(graph, seed, report, "color", "color") == 2


@pytest.mark.parametrize("defect", ("node_color", "missing_vertex", "configuration", "mirror"))
def test_replay_rejects_corrupted_maps_and_mirror_relations(defect):
    graph, seed, report = _basis()
    reference = report["independent_reference"]
    witness = reference["merge_witnesses"][0]
    if defect == "node_color":
        witness["atom_mapping"] = [[node, 2 if node == 1 else 1 if node == 2 else node] for node in graph]
    elif defect == "missing_vertex":
        witness["atom_mapping"] = witness["atom_mapping"][:-1]
    elif defect == "configuration":
        witness["target_assignment"] = [1]
    else:
        reference["mirror_classes"]["class_images"] = [0, 1]
    with pytest.raises(ValueError):
        replay_carrier(graph, seed, report, "color", "color")


def test_original_bond_color_is_checked_even_if_all_ligand_colors_match():
    graph, seed, report = _basis("tetrahedral:symmetric")
    graph = graph.copy()
    graph.edges[0, 1]["color"] = "distinct-bond"
    report["independent_reference"]["merge_witnesses"][0]["atom_mapping"] = [
        [node, 2 if node == 1 else 1 if node == 2 else node] for node in graph
    ]
    with pytest.raises(ValueError, match="edge color"):
        replay_carrier(graph, seed, report, "color", "color")
