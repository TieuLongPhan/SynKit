import json
from pathlib import Path

import pytest

from synkit.Graph.Stereo import (
    StereoDeterminacy,
    StereoEvidenceKind,
    StereoLifecycle,
    StereoPopulation,
    StereoReactionDecision,
    StereoReactionSemantics,
    StereoRefusal,
    StereoRefusalCode,
    StereoRelationKind,
)

ROOT = Path(__file__).parents[3]


def _population(value):
    return {
        "single_stereoisomer": StereoPopulation.SINGLE,
        "racemic": StereoPopulation.RACEMIC,
        "enantiomeric_mixture": StereoPopulation.ENANTIOMERIC_MIXTURE,
        "diastereomer_set": StereoPopulation.DIASTEREOMER_SET,
        "meso": StereoPopulation.MESO,
        "achiral": StereoPopulation.ACHIRAL,
        "unknown": StereoPopulation.UNKNOWN,
    }[value]


def _determinacy(value):
    return {
        "stereospecific": StereoDeterminacy.STEREOSPECIFIC,
        "stereoselective": StereoDeterminacy.STEREOSELECTIVE,
        "non_stereospecific": StereoDeterminacy.NON_STEREOSPECIFIC,
        "underdetermined": StereoDeterminacy.UNDERDETERMINED,
    }[value]


def _lifecycle_and_relation(value):
    return {
        "created": (StereoLifecycle.FORMED, None),
        "destroyed": (StereoLifecycle.BROKEN, None),
        "erased_then_created": (StereoLifecycle.FLEETING, None),
        "inverted": (StereoLifecycle.INVERTED, StereoRelationKind.OPPOSITE),
        "retained": (StereoLifecycle.RETAINED, StereoRelationKind.EQUIVALENT),
        "unknown": (
            StereoLifecycle.UNSPECIFIED,
            StereoRelationKind.UNSPECIFIED,
        ),
        "not_applicable": (
            StereoLifecycle.UNSPECIFIED,
            StereoRelationKind.UNRELATED,
        ),
    }[value]


def _representation_specific_lifecycle(case):
    tokens = {
        value
        for step in case.get("steps", ())
        for value in step.get("expected_stereo_changes", {}).values()
    }
    if not tokens:
        tokens = {
            value["effect"]
            for step in case.get("record", {}).get("steps", ())
            for value in step.get("stereo_effects", ())
        }
    token = sorted(tokens)[0] if len(tokens) == 1 else "UNSPECIFIED"
    lifecycle = {
        "PRESERVE": StereoLifecycle.RETAINED,
        "INVERT": StereoLifecycle.INVERTED,
        "FORM": StereoLifecycle.FORMED,
        "FORMED": StereoLifecycle.FORMED,
        "BREAK": StereoLifecycle.BROKEN,
        "BROKEN": StereoLifecycle.BROKEN,
        "FLEETING": StereoLifecycle.FLEETING,
        "UNSPECIFIED": StereoLifecycle.UNSPECIFIED,
    }[token]
    relation = {
        StereoLifecycle.RETAINED: StereoRelationKind.EQUIVALENT,
        StereoLifecycle.INVERTED: StereoRelationKind.OPPOSITE,
        StereoLifecycle.UNSPECIFIED: StereoRelationKind.UNSPECIFIED,
    }.get(lifecycle)
    return lifecycle, relation


def _decision_for(case):
    if case["case_kind"] == "negative_assertion":
        code = {
            "ST-41": StereoRefusalCode.MISSING_CONTEXT,
            "ST-42": StereoRefusalCode.MISSING_CONTEXT,
            "ST-43": StereoRefusalCode.UNSUPPORTED_GEOMETRY,
            "ST-44": StereoRefusalCode.UNSUPPORTED_GEOMETRY,
            "ST-45": StereoRefusalCode.MISSING_CONTEXT,
            "ST-46": StereoRefusalCode.MISSING_CONTEXT,
            "ST-47": StereoRefusalCode.UNSUPPORTED_GEOMETRY,
            "ST-48": StereoRefusalCode.CONTRADICTORY_ASSERTION,
        }[case["case_id"]]
        return StereoReactionDecision(
            refusal=StereoRefusal(
                code,
                case["name"],
                targets=(case["case_id"],),
            )
        )

    oracle = case.get("oracle")
    if oracle is not None:
        lifecycle, relation = _lifecycle_and_relation(oracle["local_geometry"])
        population = _population(oracle["product_population"])
        determinacy = _determinacy(oracle["stereo_determinacy"])
    else:
        lifecycle, relation = _representation_specific_lifecycle(case)
        population = StereoPopulation.SINGLE
        determinacy = (
            StereoDeterminacy.UNDERDETERMINED
            if lifecycle is StereoLifecycle.UNSPECIFIED
            else StereoDeterminacy.STEREOSPECIFIC
        )
    return StereoReactionDecision(
        assertion=StereoReactionSemantics(
            descriptor_class="mixed",
            lifecycle=lifecycle,
            relation=relation,
            population=population,
            determinacy=determinacy,
            evidence=(
                StereoEvidenceKind.MECHANISM_CONSTRAINED
                if case["representation"] == "mechanism_replay"
                else StereoEvidenceKind.SOURCE_DECLARED
            ),
            context=tuple(case.get("conditions", ())),
            provenance=case["case_id"],
        )
    )


def test_unknown_population_is_not_racemic():
    assert StereoPopulation.UNKNOWN is not StereoPopulation.RACEMIC
    assert StereoPopulation.UNKNOWN.value != StereoPopulation.RACEMIC.value
    with pytest.raises(ValueError, match="must be underdetermined"):
        StereoReactionSemantics(
            "tetrahedral",
            StereoLifecycle.UNSPECIFIED,
            StereoRelationKind.UNSPECIFIED,
            StereoPopulation.UNKNOWN,
            StereoDeterminacy.STEREOSPECIFIC,
        )


def test_endpoint_relations_are_required_only_when_two_configurations_exist():
    with pytest.raises(ValueError, match="requires a transported"):
        StereoReactionSemantics(
            "tetrahedral",
            StereoLifecycle.INVERTED,
            None,
        )
    with pytest.raises(ValueError, match="does not compare"):
        StereoReactionSemantics(
            "tetrahedral",
            StereoLifecycle.FORMED,
            StereoRelationKind.OPPOSITE,
        )


def test_decision_is_exactly_one_assertion_or_refusal():
    assertion = StereoReactionSemantics(
        "tetrahedral",
        StereoLifecycle.RETAINED,
        StereoRelationKind.EQUIVALENT,
    )
    refusal = StereoRefusal(
        StereoRefusalCode.AMBIGUOUS_ALIGNMENT,
        "Two replacement maps are admissible.",
    )
    with pytest.raises(ValueError, match="exactly one"):
        StereoReactionDecision()
    with pytest.raises(ValueError, match="exactly one"):
        StereoReactionDecision(assertion, refusal)


def test_assertion_and_refusal_round_trips_are_deterministic():
    decisions = (
        StereoReactionDecision(
            assertion=StereoReactionSemantics(
                "planar_bond",
                StereoLifecycle.RETAINED,
                StereoRelationKind.EQUIVALENT,
                StereoPopulation.ENANTIOMERIC_MIXTURE,
                StereoDeterminacy.STEREOSELECTIVE,
                StereoEvidenceKind.RULE_DECLARED,
                ("light", "catalyst", "light"),
                "reviewed-case",
            )
        ),
        StereoReactionDecision(
            refusal=StereoRefusal(
                StereoRefusalCode.MISSING_CONTEXT,
                "Thermal or photochemical conditions are required.",
                ("bond:1-2", "bond:1-2"),
                ("temperature", "irradiation"),
            )
        ),
    )
    for decision in decisions:
        payload = decision.to_dict()
        restored = StereoReactionDecision.from_dict(payload)
        assert restored == decision
        assert json.dumps(payload, sort_keys=True) == json.dumps(
            restored.to_dict(),
            sort_keys=True,
        )


def test_all_80_positive_and_8_negative_baseline_cases_have_one_decision():
    payload = json.loads(
        (
            ROOT / "Experiment/Lewis/mech_path/Data/MechanismBench/stereo.json"
        ).read_text()
    )
    decisions = {case["case_id"]: _decision_for(case) for case in payload["cases"]}

    assert len(decisions) == 88
    assert sum(decision.accepted for decision in decisions.values()) == 80
    assert sum(not decision.accepted for decision in decisions.values()) == 8
    assert all(
        StereoReactionDecision.from_dict(decision.to_dict()) == decision
        for decision in decisions.values()
    )


def test_historical_descriptor_change_does_not_override_chemical_relation():
    payload = json.loads(
        (
            ROOT / "Experiment/Lewis/mech_path/Data/MechanismBench/stereo.json"
        ).read_text()
    )
    retained_but_flipped = [
        case
        for case in payload["cases"]
        if case.get("oracle", {}).get("local_geometry") == "retained"
        and case.get("oracle", {}).get("descriptor_change") == "flipped"
    ]

    assert retained_but_flipped
    for case in retained_but_flipped:
        decision = _decision_for(case)
        assert decision.assertion.lifecycle is StereoLifecycle.RETAINED
        assert decision.assertion.relation is StereoRelationKind.EQUIVALENT
