import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Stereo import StereoCoupling
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ROOT = Path(__file__).parents[3]
DATA_PATH = (
    ROOT / "Experiment/Lewis/mech_path/Data/MechanismBench/stereo_couplings.json"
)
PAYLOAD = json.loads(DATA_PATH.read_text(encoding="utf-8"))
POSITIVE = [case for case in PAYLOAD["cases"] if case["kind"] == "positive"]
NEGATIVE = [case for case in PAYLOAD["cases"] if case["kind"] == "negative"]


def _rule(case):
    return SynRule.from_smart(
        case["mapped_reaction"],
        format="tuple",
        implicit_h=False,
        stereo_couplings={
            case["coupling_target"]: case["relation"],
        },
    )


def _cip_signature(reaction):
    molecule = Chem.MolFromSmiles(reaction.split(">>", 1)[1])
    assert molecule is not None
    atom_maps = {atom.GetIdx(): atom.GetAtomMapNum() for atom in molecule.GetAtoms()}
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(molecule)
    return tuple(
        sorted(
            atom.GetProp("_CIPCode")
            for atom in molecule.GetAtoms()
            if atom_maps[atom.GetIdx()] in {2, 3} and atom.HasProp("_CIPCode")
        )
    )


def test_coupling_corpus_review_boundary_and_counts():
    assert PAYLOAD["schema"] == "MechanismBench-stereo-coupling-v1"
    assert PAYLOAD["status"] == "reviewed"
    assert len(POSITIVE) == PAYLOAD["positive_case_count"] == 4
    assert len(NEGATIVE) == PAYLOAD["negative_fixture_count"] == 4
    assert {case["relation"] for case in POSITIVE} == {"SYN", "ANTI"}
    assert {case["reverse_expected_geometry"] for case in POSITIVE} == {"E", "Z"}
    assert "not experimental selectivity" in PAYLOAD["description"]


@pytest.mark.parametrize("case", POSITIVE, ids=lambda case: case["case_id"])
def test_reviewed_coupling_forward_reverse_and_joint_branching(case):
    rule = _rule(case)
    raw = SynReactor(
        case["substrate"],
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        dedup_its=False,
    )
    consolidated = SynReactor(
        case["substrate"],
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )

    assert len(raw.its_list) == case["expected_raw_branch_count"] == 2
    assert len(consolidated.its_list) == case["expected_unique_product_count"]
    assert {_cip_signature(reaction) for reaction in raw.smarts_list} == {
        tuple(values) for values in case["expected_product_cip_sets"]
    }
    assert all(
        len(
            result.graph["stereo_coupling_branch"][
                next(iter(result.graph["stereo_coupling_branch"]))
            ]["joint_targets"]
        )
        == 2
        for result in raw.its_list
    )
    assert len(raw.its_list) < 4

    reverse_rule = rule.reversed()
    for result in raw.its_list:
        product = ITSReverter(result).to_product_graph()
        reverse = SynReactor(
            product,
            reverse_rule,
            template_format="tuple",
            explicit_h=False,
            stereo_mode="strict",
        )

        assert reverse.mapping_count == len(reverse.its_list) == 1
        reverse_result = reverse.its_list[0]
        planar = reverse_result.graph["stereo_descriptors"]["product"]["bond:2-3"]
        reactant_planar = result.graph["stereo_descriptors"]["reactant"]["bond:2-3"]
        assert planar == reactant_planar
        metadata = reverse_result.graph["stereo_coupling_branch"]["bond:2-3"]
        assert metadata["kind"] == "VICINAL_ELIMINATION"
        assert metadata["paired_input"] is True

    assert reverse_rule.reversed() == rule


def test_coupling_value_normalizes_pair_order_and_round_trips():
    forward = StereoCoupling(
        "vicinal_addition",
        "syn",
        (2, 3),
        (5, 6),
    )
    reordered = StereoCoupling(
        "VICINAL_ADDITION",
        "SYN",
        (3, 2),
        (6, 5),
    )

    assert forward == reordered
    assert forward.target == "bond:2-3"
    assert StereoCoupling.from_value(forward.to_dict()) == forward
    assert forward.reverse().reverse() == forward
    assert forward.relabel({2: 20, 3: 30, 5: 50, 6: 60}).centers == (
        20,
        30,
    )
    assert StereoCoupling.supported_kinds() == {
        "VICINAL_ADDITION",
        "VICINAL_ELIMINATION",
    }


@pytest.mark.parametrize(
    "case",
    NEGATIVE[:2],
    ids=lambda case: case["case_id"],
)
def test_malformed_coupling_records_are_rejected(case):
    with pytest.raises(ValueError):
        StereoCoupling.from_value(case["coupling"])


@pytest.mark.parametrize(
    "case",
    NEGATIVE[2:],
    ids=lambda case: case["case_id"],
)
def test_context_insufficient_coupling_records_have_no_application(case):
    rule = _rule(case)
    if case["error"] == "INSUFFICIENT_ELIMINATION_CONTEXT":
        rule = rule.reversed()
    reactor = SynReactor(
        case["substrate"],
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )

    assert reactor.mapping_count == 0
    assert reactor.its_list == []
