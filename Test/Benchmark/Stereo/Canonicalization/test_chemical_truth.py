"""Test independently frozen chemical witnesses and coupled configurations."""

from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from Experiment.Stereo.Canonicalization.independent_model import assignment_key, transport
from Experiment.Stereo.ChemicalValidation.annotation import align_inventory, annotate_coordination
from Experiment.Stereo.ChemicalValidation.coordination import read_coordination, signed_tetrahedral_volume
from Experiment.Stereo.ChemicalValidation.evaluate import _partition_checks, _replay_group_references
from Experiment.Stereo.ChemicalValidation.negative import replay_negative

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def negative():
    manifest = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_negative_selection.json"
    return deepcopy(json.loads(manifest.read_text())["records"][0])


def test_all_frozen_negative_witnesses_are_full_typed_branch_exchanges():
    records = json.loads((ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_negative_selection.json").read_text())["records"]
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
    path = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_coordination_selection.json"
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
    path = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_coordination_selection.json"
    record = json.loads(path.read_text())["records"][0]
    source = read_coordination(record, ROOT)
    with pytest.raises(ValueError, match="not_alignable"):
        align_inventory(source.graph, source.graph, source.states[:1], source.states)


def test_partition_checks_reject_both_splitting_and_merging():
    base = {"id": "a", "status": "complete", "group_id": 0, "reference_class": 1,
            "reference_mirror_class": 2, "code_interning_ids": {"source": 10, "coordinate_mirror": 20}}
    assert _partition_checks([base])["passed"]
    split = deepcopy(base)
    split["code_interning_ids"]["source"] = 30
    assert not _partition_checks([base, split])["passed"]
    merge = deepcopy(base)
    merge["reference_class"] = 3
    assert not _partition_checks([base, merge])["passed"]


@pytest.fixture(scope="module")
def small_annotation():
    path = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_coordination_selection.json"
    selected = json.loads(path.read_text())["records"][:2]
    report = json.loads(json.dumps({"coordination": annotate_coordination(selected)}))
    parsed = {record["id"]: read_coordination(record, ROOT) for record in selected}
    return report, parsed


def test_full_coupled_reference_annotation_replays_before_production(small_annotation):
    report, parsed = small_annotation
    checks = _replay_group_references(report, parsed)
    assert len(checks) == 1
    assert checks[0]["assignment_count"] == 60
    assert checks[0]["complete_partition_and_mirror_replayed"]


@pytest.mark.parametrize("mutation", ["class", "assignment", "partition", "group", "mirror_relation"])
def test_altered_frozen_truth_is_rejected_before_production(small_annotation, mutation):
    original, parsed = small_annotation
    report = deepcopy(original)
    record = report["coordination"]["accepted"][0]
    if mutation == "class":
        record["reference_class"] += 1000
    elif mutation == "assignment":
        record["assignment"][0] = (record["assignment"][0] + 1) % 30
    elif mutation == "partition":
        reference = report["coordination"]["constitutional_groups"][0]["reference"]
        reference["partition"] = [[assignment] for assignment in reference["assignments"]]
    elif mutation == "group":
        report["coordination"]["constitutional_groups"].append(report["coordination"]["constitutional_groups"][0])
    else:
        record["expected_mirror_relation"] = "chiral"
    with pytest.raises(ValueError, match="frozen_"):
        _replay_group_references(report, parsed)
