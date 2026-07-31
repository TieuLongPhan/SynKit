import pytest

from synkit.Graph.Stereo import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    FrameworkFrame,
    FrameworkStereo,
    HelicalStereo,
    NonInvertibleStereoEffectError,
    OctahedralStereo,
    PlanarBondStereo,
    PlanarChiralityStereo,
    SquarePlanarStereo,
    StereoAlignmentError,
    StereoChange,
    StereoReferenceAlignment,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
)


def _configured_families():
    return (
        TetrahedralStereo((1, 2, 3, 4, 5), 1),
        SquarePlanarStereo((1, 2, 3, 4, 5), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
        PlanarBondStereo((3, 4, 1, 2, 5, 6), 0),
        AtropBondStereo((3, 4, 1, 2, 5, 6), 1),
        CumuleneAxisStereo(
            (1, 2, 3),
            ((4, 5), (6, 7)),
            1,
        ),
        ExtendedCisTransStereo(
            (1, 2, 3, 4),
            ((5, 6), (7, 8)),
            0,
        ),
        HelicalStereo((1, 2, 3, 4), 1),
        PlanarChiralityStereo((1, 2, 3, 4), 5, 1),
        FrameworkStereo(
            frozenset({1, 2, 3, 4, 5}),
            (FrameworkFrame(1, (2, 3, 4, 5), 1),),
            1,
        ),
    )


@pytest.mark.parametrize(
    "before",
    _configured_families(),
    ids=lambda value: value.descriptor_class,
)
def test_all_configured_families_transport_inverse_covariantly(before):
    after = before.invert()
    change = StereoChange.from_endpoints(before, after)

    expected_relation = before.relation_to(after).kind
    assert change.relation.kind is expected_relation
    assert change.apply_to(before) == after
    assert change.apply_to(before.invert()) == after.invert()
    assert change.reverse().reverse() == change

    relabeling = {atom: atom + 100 for atom in before.dependencies}
    relabeled = change.relabel(relabeling)
    assert relabeled.relation.kind is expected_relation
    assert relabeled.apply_to(before.relabel(relabeling)) == after.relabel(relabeling)


def test_inferred_alignment_is_stable_under_double_reversal():
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), 1)
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), -1)
    change = StereoChange.from_endpoints(before, after)

    assert change.alignment.status == "inferred"
    assert change.reverse().alignment.status == "inferred"
    assert change.reverse().reverse() == change


def test_reference_transport_composes_multiple_step_replacements():
    first = StereoReferenceAlignment(
        ((4, 6),),
        "inferred",
        (4,),
        (6,),
    )
    second = StereoReferenceAlignment(
        ((6, 8), (5, 7)),
        "explicit",
        (6, 5),
        (8, 7),
    )

    composed = first.then(second)

    assert composed.status == "explicit"
    assert composed.mapping == ((4, 8), (5, 7))
    assert composed.reverse().then(composed).mapping == ()


def test_reference_transport_relabels_owner_scoped_virtual_resources():
    alignment = StereoReferenceAlignment(
        (("@H:2", 5),),
        "explicit",
        ("@H:2",),
        (5,),
    )

    assert alignment.relabel({2: 20, 5: 50}).mapping == (("@H:20", 50),)


def test_change_composition_agrees_with_direct_multiple_transport():
    first_state = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    second_state = TetrahedralStereo((2, 1, 3, 6, 5), 1)
    final_state = TetrahedralStereo((2, 1, 3, 6, 7), -1)
    first = StereoChange.from_endpoints(first_state, second_state)
    second = StereoChange.from_endpoints(second_state, final_state)

    composed = first.then(second)
    direct = StereoChange.from_endpoints(
        first_state,
        final_state,
        reference_mapping={4: 6, 5: 7},
    )

    assert composed == direct
    assert composed.apply_to(first_state) == final_state


def test_formed_then_broken_is_fleeting():
    descriptor = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    formed = StereoChange.from_endpoints(None, descriptor)
    broken = StereoChange.from_endpoints(descriptor, None)

    composed = formed.then(broken)

    assert composed.change == "FLEETING"
    assert composed.transition == descriptor
    assert composed.reverse() == composed


def test_broken_then_formed_does_not_invent_continuity():
    before = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    after = before.invert()
    broken = StereoChange.from_endpoints(before, None)
    formed = StereoChange.from_endpoints(None, after)

    composed = broken.then(formed)

    assert composed.change == "UNSPECIFIED"
    assert composed.relation is None
    assert not composed.alignment.accepted
    assert composed.alignment.issue_code == "STEREO_COMPOSITION_INFORMATION_LOSS"
    with pytest.raises(NonInvertibleStereoEffectError):
        composed.reverse()


def test_noncomposable_intermediate_is_refused():
    left = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    other = TetrahedralStereo((9, 6, 7, 8, 10), 1)

    with pytest.raises(StereoAlignmentError) as error:
        StereoChange.from_endpoints(left, left).then(
            StereoChange.from_endpoints(other, other)
        )

    assert error.value.issue_code == "STEREO_COMPOSITION_NONCOMPOSABLE"
