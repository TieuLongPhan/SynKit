import json

import pytest

from synkit.Graph.Stereo import (
    LEGACY_REACTION_STEREO_SCHEMA,
    REACTION_STEREO_SCHEMA,
    PlanarBondStereo,
    ReactionStereoSchemaError,
    StereoChange,
    StereoCoupling,
    StereoDeterminacy,
    StereoLifecycle,
    StereoOutcome,
    StereoPopulation,
    StereoReactionSemantics,
    StereoReactionValue,
    StereoRefusal,
    StereoRefusalCode,
    StereoRelationKind,
    TetrahedralStereo,
    reaction_stereo_schema,
)


def _complete_value():
    before = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    after = TetrahedralStereo((2, 1, 3, 4, 6), -1)
    effect = StereoChange.from_endpoints(
        before,
        after,
        reference_mapping={5: 6},
    )
    return StereoReactionValue(
        guards={"atom:2": before},
        effects={"atom:2": effect},
        outcomes={"atom:2": StereoOutcome.enantiomeric_mixture(0.8, 0.2)},
        assertions={
            "atom:2": StereoReactionSemantics(
                "tetrahedral",
                StereoLifecycle.INVERTED,
                StereoRelationKind.OPPOSITE,
                StereoPopulation.ENANTIOMERIC_MIXTURE,
                StereoDeterminacy.STEREOSELECTIVE,
            )
        },
    )


def test_schema_v2_round_trip_and_normalized_bytes_are_stable():
    value = _complete_value()
    restored = StereoReactionValue.from_json(value.normalized_json())

    assert restored == value
    assert restored.normalized_json() == value.normalized_json()
    assert json.loads(value.normalized_json())["schema"] == REACTION_STEREO_SCHEMA
    assert reaction_stereo_schema()["properties"]["schema"] == {
        "const": REACTION_STEREO_SCHEMA
    }


def test_input_order_does_not_change_equality_hash_or_normalized_json():
    first = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    second = TetrahedralStereo((9, 6, 7, 8, 10), -1)
    left = StereoReactionValue(guards=(("atom:9", second), ("atom:1", first)))
    right = StereoReactionValue(guards=(("atom:1", first), ("atom:9", second)))

    assert left == right
    assert hash(left) == hash(right)
    assert left.normalized_json() == right.normalized_json()
    with pytest.raises(TypeError):
        left.guards["atom:1"] = second


def test_duplicate_targets_and_wrong_ownership_are_refused():
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    with pytest.raises(ReactionStereoSchemaError) as duplicate:
        StereoReactionValue(guards=(("atom:1", descriptor), ("atom:1", descriptor)))
    assert duplicate.value.refusal.code is StereoRefusalCode.CONTRADICTORY_ASSERTION

    with pytest.raises(ReactionStereoSchemaError) as wrong_owner:
        StereoReactionValue(guards={"atom:2": descriptor})
    assert wrong_owner.value.refusal.code is StereoRefusalCode.INVALID_REFERENCE


def test_outcome_requires_effect_and_coupling_key_must_match():
    with pytest.raises(ReactionStereoSchemaError) as missing_effect:
        StereoReactionValue(outcomes={"atom:1": StereoOutcome.racemic()})
    assert (
        missing_effect.value.refusal.code is StereoRefusalCode.CONTRADICTORY_ASSERTION
    )

    coupling = StereoCoupling.vicinal_addition(
        "SYN",
        centers=(1, 2),
        ligands=(3, 4),
    )
    with pytest.raises(ReactionStereoSchemaError) as wrong_target:
        StereoReactionValue(couplings={"bond:8-9": coupling})
    assert wrong_target.value.refusal.code is StereoRefusalCode.INVALID_COUPLING


def test_change_round_trip_preserves_explicit_reference_transport():
    before = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    after = TetrahedralStereo((2, 1, 3, 4, 6), -1)
    change = StereoChange.from_endpoints(
        before,
        after,
        reference_mapping={5: 6},
    )

    assert StereoChange.from_dict(change.to_dict()) == change
    assert StereoChange.from_dict(change.to_dict()).signature() == change.signature()


def test_v1_remains_readable_and_lossy_downgrade_is_refused():
    descriptor = PlanarBondStereo((1, 3, 2, 4, 5, 6), 0)
    legacy = {
        "schema": LEGACY_REACTION_STEREO_SCHEMA,
        "guards": {"bond:2-4": descriptor.to_dict()},
        "effects": {},
        "outcomes": {},
        "couplings": {},
    }
    restored = StereoReactionValue.from_dict(legacy)

    assert restored.schema == REACTION_STEREO_SCHEMA
    assert restored.guards == {"bond:2-4": descriptor}
    assert StereoReactionValue.from_dict(restored.to_legacy_dict()) == restored

    with pytest.raises(ReactionStereoSchemaError) as loss:
        _complete_value().to_legacy_dict()
    assert loss.value.refusal.code is StereoRefusalCode.LOSSY_PROJECTION


def test_refusals_are_sorted_and_round_trip_without_becoming_assertions():
    value = StereoReactionValue(
        refusals=(
            StereoRefusal(
                StereoRefusalCode.MISSING_CONTEXT,
                "Irradiation mode is absent.",
            ),
            StereoRefusal(
                StereoRefusalCode.AMBIGUOUS_ALIGNMENT,
                "Two endpoint transports remain.",
            ),
        )
    )
    restored = StereoReactionValue.from_dict(value.to_dict())

    assert restored == value
    assert not restored.assertions
    assert [item.code for item in restored.refusals] == [
        StereoRefusalCode.AMBIGUOUS_ALIGNMENT,
        StereoRefusalCode.MISSING_CONTEXT,
    ]
