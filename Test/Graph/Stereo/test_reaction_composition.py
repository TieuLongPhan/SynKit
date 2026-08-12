import copy

import pytest

from synkit.Graph.Stereo import (
    StereoChange,
    StereoCompositionError,
    StereoCompositionIssueCode,
    StereoCoupling,
    StereoOutcome,
    StereoReactionComposition,
    StereoReactionValue,
    TetrahedralStereo,
    compose_reaction_stereo,
    reverse_reaction_stereo,
)
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import SynReactor

STATE = TetrahedralStereo((1, 2, 3, 4, 5), 1, "composition")
INVERSE = STATE.invert()
FIRST_SN2 = (
    "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"
)
SECOND_SN2 = (
    "[CH3:1][C@@H:2]([F:3])[OH:5].[NH2-:6]>>" "[CH3:1][C@H:2]([F:3])[NH2:6].[OH-:5]"
)
FUSED_SN2 = (
    "[CH3:1][C@H:2]([F:3])[Cl:4].[NH2-:6]>>" "[CH3:1][C@H:2]([F:3])[NH2:6].[Cl-:4]"
)


def _value(change, *, outcome=None, coupling=None):
    target = "atom:1"
    return StereoReactionValue(
        guards={target: change.before} if change.before is not None else {},
        effects={target: change},
        outcomes={target: outcome} if outcome is not None else {},
        couplings=({"bond:2-3": coupling} if coupling is not None else {}),
    )


def test_invert_then_invert_composes_to_retention_and_reverses():
    first = _value(StereoChange.from_endpoints(STATE, INVERSE))
    second = _value(StereoChange.from_endpoints(INVERSE, STATE))

    composition = compose_reaction_stereo(first, second)
    effect = composition.result.effects["atom:1"]

    assert effect.change == "RETAINED"
    assert effect.before == effect.after == STATE
    assert composition.total_weight == pytest.approx(1.0)
    assert (
        reverse_reaction_stereo(reverse_reaction_stereo(composition.result))
        == composition.result
    )


def test_sequential_and_fused_sn2_rules_have_the_same_stereo_endpoint():
    first_rule = SynRule.from_smart(
        FIRST_SN2,
        format="tuple",
        implicit_h=False,
    )
    second_rule = SynRule.from_smart(
        SECOND_SN2,
        format="tuple",
        implicit_h=False,
    )
    fused_rule = SynRule.from_smart(
        FUSED_SN2,
        format="tuple",
        implicit_h=False,
    )
    composition = compose_reaction_stereo(
        StereoReactionValue(
            guards=first_rule.stereo_guards,
            effects=first_rule.stereo_effects,
        ),
        StereoReactionValue(
            guards=second_rule.stereo_guards,
            effects=second_rule.stereo_effects,
        ),
    )

    assert composition.result.effects == fused_rule.stereo_effects

    first = SynReactor(
        "C[C@H](F)Cl.[OH-]",
        first_rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )
    second = SynReactor(
        "C[C@@H](F)O.[NH2-]",
        second_rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )
    fused = SynReactor(
        "C[C@H](F)Cl.[NH2-]",
        fused_rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )

    assert first.mapping_count == second.mapping_count == fused.mapping_count == 1
    assert (
        second.its_list[0].graph["stereo_descriptors"]["product"]["atom:2"]
        == fused.its_list[0].graph["stereo_descriptors"]["product"]["atom:2"]
    )


def test_weighted_formation_then_break_converges_with_full_measure():
    formed = _value(
        StereoChange.from_endpoints(None, STATE),
        outcome=StereoOutcome.racemic(),
    )
    broken = _value(StereoChange.from_endpoints(STATE, None))

    composition = compose_reaction_stereo(formed, broken)

    assert composition.result.effects["atom:1"].change == "FLEETING"
    assert composition.result.outcomes == {}
    assert len(composition.branches) == 1
    branch = composition.branches[0]
    assert branch.final_choices == ()
    assert branch.weight == pytest.approx(1.0)
    assert branch.multiplicity == 2
    assert [item.weight for item in branch.contributions] == [0.5, 0.5]


def test_independent_outcomes_multiply_without_losing_total_measure():
    second_state = TetrahedralStereo((6, 7, 8, 9, 10), 1, "second")
    first = _value(
        StereoChange.from_endpoints(None, STATE),
        outcome=StereoOutcome.enantiomeric_mixture(0.7, 0.3),
    )
    second = StereoReactionValue(
        effects={
            "atom:6": StereoChange.from_endpoints(None, second_state),
        },
        outcomes={
            "atom:6": StereoOutcome.enantiomeric_mixture(0.6, 0.4),
        },
    )

    composition = compose_reaction_stereo(first, second)

    assert len(composition.branches) == 4
    assert sorted(branch.weight for branch in composition.branches) == (
        pytest.approx([0.12, 0.18, 0.28, 0.42])
    )
    assert composition.total_weight == pytest.approx(1.0)


def test_broken_then_formed_refuses_information_loss():
    broken = _value(StereoChange.from_endpoints(STATE, None))
    formed = _value(StereoChange.from_endpoints(None, STATE))

    with pytest.raises(StereoCompositionError) as captured:
        compose_reaction_stereo(broken, formed)

    assert captured.value.code is StereoCompositionIssueCode.INFORMATION_LOSS


def test_intermediate_guard_and_coupling_conflicts_refuse():
    first = _value(StereoChange.from_endpoints(STATE, INVERSE))
    wrong_guard = StereoReactionValue(guards={"atom:1": STATE})

    with pytest.raises(StereoCompositionError) as guard_error:
        compose_reaction_stereo(first, wrong_guard)
    assert guard_error.value.code is (StereoCompositionIssueCode.INTERMEDIATE_MISMATCH)

    coupling = StereoCoupling.vicinal_addition(
        "SYN",
        centers=(2, 3),
        ligands=(4, 5),
    )
    coupled = StereoReactionValue(couplings={"bond:2-3": coupling})
    with pytest.raises(StereoCompositionError) as coupling_error:
        compose_reaction_stereo(coupled, coupled)
    assert coupling_error.value.code is (StereoCompositionIssueCode.COUPLING_CONFLICT)


def test_population_and_unspecified_changes_are_not_reversed():
    population = _value(
        StereoChange.from_endpoints(None, STATE),
        outcome=StereoOutcome.racemic(),
    )
    unspecified = _value(
        StereoChange.from_endpoints(
            STATE,
            TetrahedralStereo(STATE.atoms, None),
        )
    )

    for value in (population, unspecified):
        with pytest.raises(StereoCompositionError) as captured:
            reverse_reaction_stereo(value)
        assert captured.value.code is StereoCompositionIssueCode.NON_INVERTIBLE


def test_composition_proof_round_trip_and_every_field_tamper():
    first = _value(StereoChange.from_endpoints(STATE, INVERSE))
    second = _value(StereoChange.from_endpoints(INVERSE, STATE))
    payload = compose_reaction_stereo(first, second).to_dict()

    assert StereoReactionComposition.from_dict(payload).to_dict() == payload

    mutations = (
        ("schema", "invalid"),
        ("result", {}),
        ("branches", []),
        ("correlation_policy", "forged"),
        ("total_weight", 0.5),
        ("proof_digest", "0" * 64),
    )
    for field, replacement in mutations:
        corrupted = copy.deepcopy(payload)
        corrupted[field] = replacement
        with pytest.raises(StereoCompositionError) as captured:
            StereoReactionComposition.from_dict(corrupted)
        assert captured.value.code is StereoCompositionIssueCode.PROOF_TAMPERED
