"""Differential laws for the Sprint 17 descriptor migration."""

from itertools import permutations

import pytest

from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoRelationKind,
    StereoSpecification,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    stereo_from_dict,
)
from synkit.Graph.Stereo.legacy import (
    legacy_canonical_form,
    legacy_descriptor_query_matches,
)
from synkit.Graph.Stereo.matching import descriptor_query_matches

ATOM_CASES = (
    (TetrahedralStereo, 4, (-1, 1, None)),
    (SquarePlanarStereo, 4, (0, None)),
    (TrigonalBipyramidalStereo, 5, (-1, 1, None)),
    (OctahedralStereo, 6, (-1, 1, None)),
)
BOND_CASES = (
    (PlanarBondStereo, (0, None)),
    (AtropBondStereo, (-1, 1, None)),
)


def _assert_partition_bijection(descriptors: list[object]) -> None:
    legacy_to_orbit: dict[tuple[object, ...], tuple[object, ...]] = {}
    orbit_to_legacy: dict[tuple[object, ...], tuple[object, ...]] = {}
    legacy_to_hash: dict[tuple[object, ...], int] = {}
    baseline = descriptors[0]

    for descriptor in descriptors:
        legacy = legacy_canonical_form(descriptor)
        orbit = descriptor.configuration.canonical_form()
        assert descriptor.canonical_form() == legacy
        assert legacy_to_orbit.setdefault(legacy, orbit) == orbit
        assert orbit_to_legacy.setdefault(orbit, legacy) == legacy
        assert legacy_to_hash.setdefault(legacy, hash(descriptor)) == hash(descriptor)

        diagnostics = []
        result = descriptor.same_configuration(
            baseline,
            semantics="compare",
            diagnostics=diagnostics,
        )
        assert result == (legacy == legacy_canonical_form(baseline))
        assert diagnostics[0].agreement
        assert diagnostics[0].stage == "descriptor_identity"

        restored = stereo_from_dict(descriptor.to_dict())
        assert restored.to_dict() == descriptor.to_dict()
        assert restored == descriptor
        assert hash(restored) == hash(descriptor)

    assert len(legacy_to_orbit) == len(orbit_to_legacy)


@pytest.mark.parametrize("descriptor_type,arity,parities", ATOM_CASES)
def test_all_atom_encodings_have_exactly_the_legacy_identity_partition(
    descriptor_type: type,
    arity: int,
    parities: tuple[int | None, ...],
) -> None:
    descriptors = [
        descriptor_type((0, *references), parity)
        for references in permutations(range(1, arity + 1))
        for parity in parities
    ]

    _assert_partition_bijection(descriptors)


@pytest.mark.parametrize("descriptor_type,parities", BOND_CASES)
def test_all_bond_encodings_have_exactly_the_legacy_identity_partition(
    descriptor_type: type,
    parities: tuple[int | None, ...],
) -> None:
    descriptors = [
        descriptor_type(frame, parity)
        for frame in permutations(range(1, 7))
        for parity in parities
    ]

    _assert_partition_bijection(descriptors)


@pytest.mark.parametrize("descriptor_type,arity,parities", ATOM_CASES)
def test_atom_query_compare_mode_has_no_unregistered_divergence(
    descriptor_type: type,
    arity: int,
    parities: tuple[int | None, ...],
) -> None:
    query = descriptor_type((0, *range(1, arity + 1)), parities[0])
    for references in permutations(range(1, arity + 1)):
        for parity in parities:
            candidate = descriptor_type((0, *references), parity)
            for policy in ("exact", "wildcard", "either"):
                diagnostics = []
                result = descriptor_query_matches(
                    query,
                    candidate,
                    unknown_policy=policy,
                    semantics="compare",
                    diagnostics=diagnostics,
                )
                assert result == legacy_descriptor_query_matches(
                    query,
                    candidate,
                    unknown_policy=policy,
                )
                assert len(diagnostics) == 1
                assert diagnostics[0].agreement
                assert diagnostics[0].registered


@pytest.mark.parametrize("descriptor_type,parities", BOND_CASES)
def test_bond_query_compare_mode_has_no_unregistered_divergence(
    descriptor_type: type,
    parities: tuple[int | None, ...],
) -> None:
    query = descriptor_type((1, 2, 3, 4, 5, 6), parities[0])
    for frame in permutations((1, 2, 5, 6)):
        candidate_atoms = (*frame[:2], 3, 4, *frame[2:])
        for parity in parities:
            candidate = descriptor_type(candidate_atoms, parity)
            for policy in ("exact", "wildcard", "either"):
                diagnostics = []
                result = descriptor_query_matches(
                    query,
                    candidate,
                    unknown_policy=policy,
                    semantics="compare",
                    diagnostics=diagnostics,
                )
                assert result == legacy_descriptor_query_matches(
                    query,
                    candidate,
                    unknown_policy=policy,
                )
                assert len(diagnostics) == 1
                assert diagnostics[0].agreement


@pytest.mark.parametrize(
    "source,target,expected",
    (
        (
            TetrahedralStereo((0, 1, 2, 3, 4), 1),
            TetrahedralStereo((0, 1, 2, 3, 4), -1),
            StereoRelationKind.OPPOSITE,
        ),
        (
            SquarePlanarStereo((0, 1, 2, 3, 4), 0),
            SquarePlanarStereo((0, 2, 1, 3, 4), 0),
            StereoRelationKind.RECONFIGURED,
        ),
        (
            TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1),
            TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), -1),
            StereoRelationKind.RECONFIGURED,
        ),
        (
            OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1),
            OctahedralStereo((0, 1, 2, 3, 4, 5, 6), -1),
            StereoRelationKind.RECONFIGURED,
        ),
        (
            PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
            PlanarBondStereo((2, 1, 3, 4, 5, 6), 0),
            StereoRelationKind.OPPOSITE,
        ),
        (
            AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
            AtropBondStereo((1, 2, 3, 4, 5, 6), -1),
            StereoRelationKind.OPPOSITE,
        ),
    ),
)
def test_descriptor_relations_are_geometry_specific(
    source: object,
    target: object,
    expected: StereoRelationKind,
) -> None:
    relation = source.relation_to(target)

    assert relation.kind is expected
    assert relation.replayable


@pytest.mark.parametrize(
    "descriptor",
    (
        TetrahedralStereo((1, 2, 3, 4, 5), None),
        SquarePlanarStereo((1, 2, 3, 4, 5), None),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), None),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), None),
        PlanarBondStereo((1, 2, 3, 4, 5, 6), None),
        AtropBondStereo((1, 2, 3, 4, 5, 6), None),
    ),
)
def test_unspecified_is_stored_information_not_absence(descriptor: object) -> None:
    fixed_parity = (
        0
        if descriptor.descriptor_class
        in {
            "square_planar",
            "planar_bond",
        }
        else 1
    )
    fixed = type(descriptor)(descriptor.atoms, fixed_parity)

    assert descriptor.specification is StereoSpecification.UNSPECIFIED
    assert fixed.specification is StereoSpecification.FIXED
    assert descriptor != fixed
    assert descriptor is not None
    assert descriptor.relation_to(fixed).kind is StereoRelationKind.UNSPECIFIED


def test_reference_replacement_preserves_encoding_and_protects_loci() -> None:
    atom = TetrahedralStereo((1, 2, 3, 4, 5), -1, "manual")
    bond = PlanarBondStereo((1, 2, 3, 4, 5, 6), 0, "manual")

    replaced_atom = atom.replace_reference(2, "@H:1")
    replaced_bond = bond.replace_references({1: "@H:3", 6: "@LP:4"})
    assert replaced_atom.atoms == (1, "@H:1", 3, 4, 5)
    assert replaced_atom.parity == -1
    assert replaced_atom.provenance == "manual"
    assert replaced_bond.atoms == ("@H:3", 2, 3, 4, 5, "@LP:4")

    with pytest.raises(ValueError, match="loci"):
        atom.replace_reference(1, 99)
    with pytest.raises(ValueError, match="loci"):
        bond.replace_reference(3, 99)
    with pytest.raises(ValueError, match="absent"):
        atom.replace_reference(99, 100)
    with pytest.raises(ValueError, match="distinct"):
        atom.replace_reference(2, 3)


def test_legacy_dictionary_payload_is_exactly_unchanged() -> None:
    descriptor = AtropBondStereo((1, 2, 3, 4, 5, 6), -1, "rdkit")

    assert descriptor.to_dict() == {
        "descriptor_class": "atrop_bond",
        "atoms": [1, 2, 3, 4, 5, 6],
        "parity": -1,
        "provenance": "rdkit",
    }


@pytest.mark.parametrize(
    "descriptor",
    (
        TetrahedralStereo((1, 2, 3, 4, 5), 1),
        PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
        AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
    ),
)
def test_binary_opposite_is_derived_from_the_orbit(descriptor: object) -> None:
    opposite = descriptor.opposite()

    assert descriptor.relation_to(opposite).kind is StereoRelationKind.OPPOSITE
    assert opposite.opposite() == descriptor


@pytest.mark.parametrize(
    "descriptor",
    (
        SquarePlanarStereo((1, 2, 3, 4, 5), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
    ),
)
def test_nonbinary_geometries_refuse_opposite_convenience(descriptor: object) -> None:
    with pytest.raises(ValueError, match="no binary opposite"):
        descriptor.opposite()
