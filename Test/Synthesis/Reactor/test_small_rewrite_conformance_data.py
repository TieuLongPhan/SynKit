import json
from pathlib import Path

import pytest
from rdkit import Chem

from synkit.Chem.Reaction import audit_explicit_h_reaction
from synkit.Graph.Stereo import StereoOutcome
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ROOT = Path(__file__).parents[3]
DATA_PATH = ROOT / "Data/MechanismBench/small_rewrite_conformance.json"
PAYLOAD = json.loads(DATA_PATH.read_text(encoding="utf-8"))
CASES = PAYLOAD["cases"]
HYDROGEN_ABSENCE_ERRORS = {
    "NO_EXPLICIT_MAPPED_HYDROGEN",
    "NO_CHANGED_EXPLICIT_HYDROGEN",
}


def _canonical_constitution(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def _rule(case):
    metadata = case.get("rule_metadata", {})
    serialized_outcomes = metadata.get("stereo_outcomes")
    if not serialized_outcomes:
        return case["mapped_reaction"]
    outcomes = {
        key: StereoOutcome(value["kind"], tuple(value["weights"]))
        for key, value in serialized_outcomes.items()
    }
    return SynRule.from_smart(
        case["mapped_reaction"],
        format="tuple",
        stereo_outcomes=outcomes,
    )


def test_small_dataset_is_deliberately_bounded_and_covers_core_semantics():
    assert PAYLOAD["schema"] == "SynKit-small-rewrite-conformance-v1"
    assert len(CASES) == 9
    assert len({case["case_id"] for case in CASES}) == len(CASES)
    assert {focus for case in CASES for focus in case["focus"]} == {
        "hydrogen",
        "polar",
        "radical",
        "stereo",
    }
    assert {
        change
        for case in CASES
        for change in case["expected"]["stereo_changes"].values()
    } == {"BROKEN", "FORMED", "INVERTED", "RETAINED"}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_case_is_balanced_and_stores_declared_rule_semantics(case):
    report = audit_explicit_h_reaction(case["mapped_reaction"])
    structural_errors = set(report.errors) - HYDROGEN_ABSENCE_ERRORS

    assert not structural_errors
    assert (
        list(report.changed_hydrogen_maps) == case["expected"]["changed_hydrogen_maps"]
    )
    if "hydrogen" in case["focus"]:
        assert report.accepted

    its = rsmi_to_its(
        case["mapped_reaction"],
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    assert {
        key: change.change
        for key, change in its.graph.get("stereo_changes", {}).items()
    } == case["expected"]["stereo_changes"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_case_applies_and_rejects_declared_wrong_stereoisomers(case):
    application = case["application"]
    reactor = SynReactor(
        application["substrate"],
        _rule(case),
        template_format="tuple",
        explicit_h=False,
        radical_policy="strict",
        stereo_mode="strict",
    )
    generated = {
        _canonical_constitution(candidate.split(">>", 1)[1])
        for candidate in reactor.smarts
    }

    assert len(reactor.smarts) == application["expected_product_count"]
    assert generated == {
        _canonical_constitution(application["expected_product_constitution"])
    }

    for rejected_substrate in application["rejected_substrates"]:
        rejected = SynReactor(
            rejected_substrate,
            _rule(case),
            template_format="tuple",
            explicit_h=False,
            radical_policy="strict",
            stereo_mode="strict",
        )
        assert rejected.mapping_count == 0
        assert rejected.smarts == []


def test_racemic_case_uses_one_rule_to_generate_both_enantiomers():
    case = next(case for case in CASES if case["case_id"] == "sn1-racemic-capture")
    rule = _rule(case)
    reactor = SynReactor(
        case["application"]["substrate"],
        rule,
        explicit_h=False,
        stereo_mode="strict",
    )
    cip_codes = set()
    for candidate in reactor.smarts:
        molecule = Chem.MolFromSmiles(candidate.split(">>", 1)[1])
        assert molecule is not None
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        cip_codes.update(
            atom.GetProp("_CIPCode")
            for atom in molecule.GetAtoms()
            if atom.HasProp("_CIPCode")
        )

    assert rule.stereo_outcomes == {"atom:2": StereoOutcome.racemic()}
    assert cip_codes == set(case["application"]["expected_cip_codes"])
