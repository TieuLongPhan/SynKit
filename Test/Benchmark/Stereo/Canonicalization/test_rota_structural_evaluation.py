"""Frozen RotA truth must replay before production can claim agreement."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from Experiment.Stereo.ChemicalValidation.rota_evaluate import (
    compare_perception, replay_frozen_annotation,
)
from Experiment.Stereo.ChemicalValidation.rota_annotation import source_inventory

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def frozen():
    selection = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/rota_adjudication_selection.json"
    annotation = ROOT / "Experiment/Stereo/Data/Revision/Chemical/rota_structural_annotation.json"
    return json.loads(selection.read_text())["records"], json.loads(annotation.read_text())


def test_all_sixty_independent_annotations_replay(frozen):
    selected, annotation = frozen
    assert replay_frozen_annotation(annotation, selected)["all_sixty_references_replayed"]


@pytest.mark.parametrize("mutation", ["scope", "carrier", "partition", "class_count", "production_used"])
def test_false_frozen_truth_is_rejected_before_production(frozen, mutation):
    selected, original = frozen
    annotation = deepcopy(original)
    if mutation == "scope":
        annotation["records"].pop()
    elif mutation == "carrier":
        annotation["records"][0]["constitutional_carrier"]["carrier_status"] = "invented_status"
    elif mutation == "partition":
        annotation["records"][0]["reference"]["partition"] = []
    elif mutation == "class_count":
        annotation["records"][0]["configured_class_count"] += 1
    else:
        annotation["production_evaluated"] = True
    with pytest.raises(ValueError, match="rota_"):
        replay_frozen_annotation(annotation, selected)


@pytest.mark.parametrize("family", ["axis", "tetrahedral_pair", "tetrahedral_spiro"])
def test_matching_perception_contract_survives_renumbering(frozen, family):
    selected, annotation = frozen
    index = next(index for index, truth in enumerate(annotation["records"]) if truth["independent_family"] == family)
    record = selected[index]
    _graph, seeds, _family, _facts = source_inventory(record)
    assert compare_perception(record, annotation["records"][index], seeds)["passed"]


def test_geometry_only_spiro_projection_cannot_be_promoted_to_confirmed():
    from rdkit import Chem
    from Experiment.Stereo.Perception.full_detection import _rota_predicted_loci, _rota_projection_issues

    molecule = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
    category = "Spiral atom and chain"
    broad = _rota_predicted_loci(molecule, (), category)
    confirmed = _rota_predicted_loci(molecule, (), category, require_confirmed_support=True)
    assert broad == {"3"}
    assert confirmed == set()
    assert _rota_projection_issues(molecule, (), category, broad, confirmed) == []
