"""Regression contracts for the paired RBL reconstruction benchmark."""

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import RBLEngine, RBL_RESULT_SCHEMA

from Experiment.RBL.benchmark import (
    RBL_BENCHMARK_RECORD_SCHEMA,
    RBL_BENCHMARK_SCHEMA,
    RBL_EVALUATION_SCHEMA,
    RULE_ADAPTER,
    RULE_EXTRACTION_DESCRIPTION,
    canonical_reaction,
    extract_normalized_rule,
)
from Experiment.RBL.classify_complete_retry import balance_status
from Experiment.RBL.retry_complete import reaction_balance

R_34872 = {
    "R_id": "R_34872",
    "raw": "CI.CO>>COC",
    "complete": "CI.CO>>COC.I",
    "aam": ("[CH3:1][I:2].[CH3:3][O:4][H:5]>>" "[CH3:1][O:4][CH3:3].[I:2][H:5]"),
}

R_34427 = {
    "R_id": "R_34427",
    "raw": "CCCCC(=O)OC(=O)CCCC>>CCCCC(=O)O",
    "complete": "CCCCC(=O)OC(=O)CCCC.O>>CCCCC(=O)O.CCCCC(=O)O",
    "aam": (
        "[CH3:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])[O:7]"
        "[C:8](=[O:9])[CH2:10][CH2:11][CH2:12][CH3:13].[OH:14][H:15]>>"
        "[C:8](=[O:9])([CH2:10][CH2:11][CH2:12][CH3:13])[OH:14]."
        "[CH3:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])[O:7][H:15]"
    ),
}

R_4223_AAM = (
    "[Na+:1].[O-:2][C:3](=[O:4])[c:5]1[cH:6][c:8]2[n:9][cH:11]"
    "[cH:12][c:13]([Cl:14])[c:10]2[s:7]1.[OH:15][CH:16]1[CH2:17]"
    "[CH2:19][N:20]([H:21])[CH2:18]1>>[C:3](=[O:4])([c:5]1[cH:6]"
    "[c:8]2[n:9][cH:11][cH:12][c:13]([Cl:14])[c:10]2[s:7]1)[N:20]1"
    "[CH2:18][CH:16]([OH:15])[CH2:17][CH2:19]1.[Na:1][O:2][H:21]"
)


def test_rbl_public_api_and_serialization_schemas_are_versioned() -> None:
    assert RBL_RESULT_SCHEMA == "synkit.rbl-result/1"
    assert RBL_BENCHMARK_SCHEMA == "synkit.rbl-benchmark/1"
    assert RBL_BENCHMARK_RECORD_SCHEMA == "synkit.rbl-benchmark-record/1"
    assert RBL_EVALUATION_SCHEMA == "synkit.rbl-evaluation/1"

    engine = RBLEngine(mode="fast_track")
    assert engine.result["schema"] == RBL_RESULT_SCHEMA


def test_extracted_rule_is_explicitly_adapted_through_synrule() -> None:
    source = rsmi_to_its(R_4223_AAM, core=True, format="tuple")
    rule, audit = extract_normalized_rule({"R_id": "R_4223", "aam": R_4223_AAM})

    assert isinstance(rule, SynRule)
    assert audit["adapter"] == RULE_ADAPTER
    assert RULE_ADAPTER in RULE_EXTRACTION_DESCRIPTION
    assert source.nodes[2]["lone_pairs"] == (3, 2)
    assert rule.rc.raw.nodes[2]["lone_pairs"] == (1, 0)
    assert audit["normalization_violations"] == 0
    assert {
        (tuple(change["before"]), tuple(change["after"]))
        for change in audit["resource_changes"]
        if change["attribute"] == "lone_pairs"
    } >= {((3, 2), (1, 0))}


def test_rbl_replays_the_normalized_synrule_against_incomplete_input() -> None:
    rule, _audit = extract_normalized_rule(R_34872)
    engine = RBLEngine(mode="fast_track").process(R_34872["raw"], rule)
    standardizer = Standardize()
    candidates = {
        standardizer.fit(
            candidate,
            remove_aam=True,
            ignore_stereo=True,
            remove_invalid=False,
        )
        for candidate in engine.fused_rsmis
    }

    assert R_34872["complete"] in candidates


def test_hcomplete_rule_replays_unambiguous_h2_transfer_exactly() -> None:
    aam = "[CH2:1]=[O:2].[H:3][H:4]>>[CH2:1]([H:3])[O:2][H:4]"
    standardizer = Standardize()
    complete = canonical_reaction(aam, standardizer)
    record = {"R_id": "H2", "aam": aam, "complete": complete}
    rule, audit = extract_normalized_rule(record)

    engine = RBLEngine(mode="fast_track").process(complete, rule)
    candidates = {
        canonical_reaction(candidate, standardizer) for candidate in engine.fused_rsmis
    }

    assert audit["hydrogen_completion"]["status"] == "unambiguous"
    assert audit["hydrogen_completion"]["candidates"] == 1
    assert audit["hydrogen_completion"]["exhaustive"] is True
    assert audit["hydrogen_completion"]["signature"]
    assert (
        sum(
            attributes.get("element") == ("H", "H")
            for _, attributes in rule.rc.raw.nodes(data=True)
        )
        == 2
    )
    assert engine.result["reason"] == "quick_check_match"
    assert complete in candidates


def test_explicit_h_rule_replays_anhydride_hydrolysis_on_complete_input() -> None:
    rule, audit = extract_normalized_rule(R_34427)
    engine = RBLEngine(mode="fast_track").process(R_34427["complete"], rule)
    standardizer = Standardize()
    candidates = {
        canonical_reaction(candidate, standardizer) for candidate in engine.fused_rsmis
    }

    assert audit["adapter"] == "SynRule(implicit_h=False, format='tuple')"
    assert R_34427["complete"] in candidates


def test_reaction_balance_counts_implicit_hydrogen_and_formal_charge() -> None:
    report = reaction_balance("[NH4+]>>N")

    assert report["parsed"] is True
    assert report["element_delta_products_minus_reactants"] == {"H": -1}
    assert report["charge_delta_products_minus_reactants"] == -1
    assert report["element_balanced"] is False
    assert report["charge_balanced"] is False


def test_charge_conservation_is_distinct_from_neutrality() -> None:
    report = reaction_balance("[Na+]>>[Na+]")

    assert report["element_and_charge_balanced"] is True
    assert report["charge_balanced"] is True
    assert report["both_sides_neutral"] is False


def test_reaction_balance_supports_disconnected_ionic_sides() -> None:
    report = reaction_balance("[Na+].[Cl-]>>[Cl-].[Na+]")

    assert report["element_and_charge_balanced"] is True
    assert report["both_sides_neutral"] is True


def test_balance_status_separates_element_and_charge_failures() -> None:
    assert balance_status(reaction_balance("CC>>CC")) == "balanced"
    assert balance_status(reaction_balance("CC>>C")) == "element_only_unbalanced"
    assert balance_status(reaction_balance("[Na+]>>[Na]")) == ("charge_only_unbalanced")
    assert balance_status(reaction_balance("[NH4+]>>N")) == (
        "element_and_charge_unbalanced"
    )
