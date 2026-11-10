"""Test independently frozen chemical witnesses and coupled configurations."""

from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from Experiment.Stereo.reference.model import assignment_key, transport
from Experiment.Stereo.reference.chemical_annotation import align_inventory, annotate_coordination
from Experiment.Stereo.reference.coordination import read_coordination, signed_tetrahedral_volume
from Experiment.Stereo.reference.biaryl import replay_negative

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def negative():
    manifest = ROOT / "Experiment/Stereo/data/paper/core/inputs/negative_inputs.json"
    return deepcopy(json.loads(manifest.read_text())["records"][0])


def test_all_frozen_negative_witnesses_are_full_typed_branch_exchanges():
    path = ROOT / "Experiment/Stereo/data/paper/core/inputs/negative_inputs.json"
    records = json.loads(path.read_text())["records"]
    assert len(records) == 48
    for record in records:
        _molecule, truth = replay_negative(record)
        assert truth["full_typed_bijection_replayed"]
        assert truth["terminal_branches_exchanged"]


@pytest.mark.parametrize("mutation", ["not_bijection", "identity", "branch_claim", "axis_claim", "ring_claim"])
def test_false_negative_witnesses_are_rejected(negative, mutation):
    witness = negative["independent_negative_witness"]
    if mutation == "not_bijection":
        witness["atom_map"][0][1] = witness["atom_map"][1][1]
    elif mutation == "identity":
        witness["atom_map"] = [[source, source] for source, _target in witness["atom_map"]]
    elif mutation == "branch_claim":
        witness["swapped_terminal_references"] = [0, 3]
    elif mutation == "axis_claim":
        witness["axis_endpoints"] = [0, 1]
    else:
        witness["ring_order"][1] = witness["ring_order"][0]
    with pytest.raises(ValueError):
        replay_negative(negative)


def test_signed_volume_convention_agrees_with_independent_rdkit_geometry_assignment():
    molecule = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
    points = np.array(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=float)
    for reflect in (False, True):
        if reflect:
            points[:, 0] *= -1
        conformer = Chem.Conformer(5)
        conformer.SetAtomPosition(0, (0, 0, 0))
        for index, point in enumerate(points, start=1):
            conformer.SetAtomPosition(index, point)
        molecule.RemoveAllConformers()
        molecule.AddConformer(conformer)
        Chem.AssignAtomChiralTagsFromStructure(molecule, replaceExistingTags=True)
        tag = molecule.GetAtomWithIdx(0).GetChiralTag()
        expected = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1, Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1}[tag]
        assert signed_tetrahedral_volume(points)[1] == expected


def test_inventory_alignment_retains_both_metal_and_nitrogen_frames_under_one_map():
    path = ROOT / "Experiment/Stereo/data/paper/core/inputs/coordination_inputs.json"
    records = json.loads(path.read_text())["records"]
    first, second = (read_coordination(record, ROOT) for record in records[:2])
    mapping, _count = align_inventory(second.graph, first.graph, second.states, first.states)
    assert len(mapping) == len(second.graph)
    mapped = tuple(transport(state, mapping) for state in second.states)
    assert len(mapped) == 2
    assert {state.family for state in mapped} == {"tetrahedral", "octahedral"}
    # A graph-only renumbering changes both physical and owner-scoped ports.
    renumbering = {node: node + 100 for node in second.graph}
    renamed = nx.relabel_nodes(second.graph, renumbering)
    moved = tuple(transport(state, renumbering) for state in second.states)
    back, _count = align_inventory(renamed, second.graph, moved, second.states)
    assert assignment_key(tuple(transport(state, back) for state in moved)) == assignment_key(second.states)


def test_omitting_a_coupled_nitrogen_frame_prevents_inventory_alignment():
    path = ROOT / "Experiment/Stereo/data/paper/core/inputs/coordination_inputs.json"
    record = json.loads(path.read_text())["records"][0]
    source = read_coordination(record, ROOT)
    with pytest.raises(ValueError, match="not_alignable"):
        align_inventory(source.graph, source.graph, source.states[:1], source.states)


@pytest.fixture(scope="module")
def small_annotation():
    path = ROOT / "Experiment/Stereo/data/paper/core/inputs/coordination_inputs.json"
    selected = json.loads(path.read_text())["records"][:2]
    report = json.loads(json.dumps({"coordination": annotate_coordination(selected)}))
    parsed = {record["id"]: read_coordination(record, ROOT) for record in selected}
    return report, parsed
