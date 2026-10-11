"""Reaction witnesses must follow their frame basis, including nonnormal groups."""

import pytest

from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoAlignmentError,
    StereoChange,
    StereoRelationKind,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
)


CASES = (
    (
        TetrahedralStereo((0, 1, 2, 3, 4), 1),
        TetrahedralStereo((0, 2, 1, 3, 4), 1),
    ),
    (
        SquarePlanarStereo((0, 1, 2, 3, 4), 0),
        SquarePlanarStereo((0, 2, 1, 3, 4), 0),
    ),
    (
        TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1),
        TrigonalBipyramidalStereo((0, 3, 2, 1, 4, 5), 1),
    ),
    (
        OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1),
        OctahedralStereo((0, 3, 2, 1, 4, 5, 6), 1),
    ),
    (
        PlanarBondStereo((1, 2, 0, 3, 4, 5), 0),
        PlanarBondStereo((2, 1, 0, 3, 4, 5), 0),
    ),
    (
        AtropBondStereo((1, 2, 0, 3, 4, 5), 1),
        AtropBondStereo((2, 1, 0, 3, 4, 5), 1),
    ),
)
COORDINATION_CASES = CASES[1:4]


def _from_frame(prototype, frame, parity=None):
    return type(prototype)(
        frame, prototype.parity if parity is None else parity, prototype.provenance
    )


@pytest.mark.parametrize("before,after", CASES, ids=lambda value: value.descriptor_class)
@pytest.mark.parametrize("replace_ligand", (False, True))
def test_every_guard_representative_replays_the_same_product_and_reverse(
    before, after, replace_ligand
):
    if replace_ligand:
        leaving = before.atoms[-1]
        after = after.replace_reference(leaving, 90)
    change = StereoChange.from_endpoints(before, after)
    renumbering = {atom: 2000 - 11 * atom for atom in change.dependencies}
    renamed_change = change.relabel(renumbering)

    for basis in before.configuration.definition.preserving_group.elements:
        substrate = _from_frame(before, basis.apply(before.configuration.frame))
        assert substrate.same_configuration(before)
        product = change.apply_to(substrate)
        assert product.same_configuration(after)
        # A product follows the supplied representative rather than copying
        # the product template's raw frame.
        assert product.configuration.frame == basis.apply(after.configuration.frame)
        assert change.reverse().apply_to(product).same_configuration(substrate)
        renamed_product = renamed_change.apply_to(substrate.relabel(renumbering))
        assert renamed_product.same_configuration(product.relabel(renumbering))


def test_square_planar_fixed_witness_counterexample_is_not_an_orbit_operation():
    before, after = CASES[1]
    change = StereoChange.from_endpoints(before, after)
    raw_orbits = set()
    corrected_orbits = set()
    for basis in before.configuration.definition.preserving_group.elements:
        substrate = _from_frame(before, basis.apply(before.configuration.frame))
        raw_frame = change.relation.witness.apply(substrate.configuration.frame)
        raw_orbits.add(_from_frame(after, raw_frame).configuration.canonical_frame)
        corrected_orbits.add(change.apply_to(substrate).configuration.canonical_frame)
    assert len(raw_orbits) == 2
    assert corrected_orbits == {after.configuration.canonical_frame}


@pytest.mark.parametrize(
    "before,after", COORDINATION_CASES, ids=lambda value: value.descriptor_class
)
def test_nonnormalizing_reconfiguration_refuses_a_different_guard_orbit(before, after):
    change = StereoChange.from_endpoints(before, after)
    assert change.relation.kind is StereoRelationKind.RECONFIGURED
    for basis in after.configuration.definition.preserving_group.elements:
        substrate = _from_frame(after, basis.apply(after.configuration.frame))
        assert not substrate.same_configuration(before)
        with pytest.raises(StereoAlignmentError) as error:
            change.apply_to(substrate)
        assert error.value.issue_code == "STEREO_RELATION_AMBIGUOUS_BASIS"


@pytest.mark.parametrize("before,unused", CASES, ids=lambda value: value.descriptor_class)
def test_normalizing_effect_is_intrinsic_on_other_configuration_orbits(before, unused):
    after = before.invert()
    change = StereoChange.from_endpoints(before, after)
    other = next(
        candidate
        for permutation in before.configuration.definition.unspecified_group.elements
        if not (
            candidate := _from_frame(before, permutation.apply(before.configuration.frame))
        ).same_configuration(before)
    )
    for basis in before.configuration.definition.preserving_group.elements:
        substrate = _from_frame(other, basis.apply(other.configuration.frame))
        product = change.apply_to(substrate)
        assert product.same_configuration(other.invert())
        assert change.reverse().apply_to(product).same_configuration(substrate)


@pytest.mark.parametrize(
    "before,after", COORDINATION_CASES, ids=lambda value: value.descriptor_class
)
def test_nonnormalizing_effect_does_not_orient_unspecified_inputs_or_outputs(before, after):
    change = StereoChange.from_endpoints(before, after)
    unknown = type(before)(before.atoms, None)
    assert change.apply_to(unknown).parity is None
    erase = StereoChange.from_endpoints(before, type(after)(after.atoms, None))
    assert erase.apply_to(after).parity is None


def test_known_substrate_with_different_material_references_is_refused():
    before, after = CASES[1]
    wrong_references = before.replace_reference(1, 99)
    with pytest.raises(StereoAlignmentError) as error:
        StereoChange.from_endpoints(before, after).apply_to(wrong_references)
    assert error.value.issue_code == "STEREO_ALIGNMENT_INCOMPATIBLE_REFERENCES"
