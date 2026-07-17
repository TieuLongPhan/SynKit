"""Prove that the independent Sprint 16 oracle reproduces Beta-2 semantics."""

from itertools import permutations

import pytest

from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoSemanticComparison,
    StereoSemanticsMode,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
)
from synkit.Graph.Stereo.changes import classify_stereo_change
from synkit.Graph.Stereo.legacy import (
    legacy_canonical_form,
    legacy_classify_stereo_change,
    legacy_descriptor_query_matches,
    legacy_inverted_form,
)
from synkit.Graph.Stereo.matching import descriptor_query_matches


ATOM_CASES = (
    (TetrahedralStereo, 4, (-1, 1, None)),
    (SquarePlanarStereo, 4, (0, None)),
    (TrigonalBipyramidalStereo, 5, (-1, 1, None)),
    (OctahedralStereo, 6, (-1, 1, None)),
)


@pytest.mark.parametrize("descriptor_type,arity,parities", ATOM_CASES)
def test_legacy_oracle_matches_every_atom_frame_permutation(
    descriptor_type: type,
    arity: int,
    parities: tuple[int | None, ...],
) -> None:
    for references in permutations(range(2, arity + 2)):
        for parity in parities:
            descriptor = descriptor_type((1, *references), parity)
            assert legacy_canonical_form(descriptor) == descriptor.canonical_form()
            assert legacy_inverted_form(descriptor) == (
                descriptor.invert().canonical_form()
            )


@pytest.mark.parametrize(
    "descriptor_type,parities",
    (
        (PlanarBondStereo, (0, None)),
        (AtropBondStereo, (-1, 1, None)),
    ),
)
def test_legacy_oracle_matches_every_bond_frame_permutation(
    descriptor_type: type,
    parities: tuple[int | None, ...],
) -> None:
    for references in permutations((1, 2, 5, 6)):
        frame = (*references[:2], 3, 4, *references[2:])
        for parity in parities:
            descriptor = descriptor_type(frame, parity)
            assert legacy_canonical_form(descriptor) == descriptor.canonical_form()
            assert legacy_inverted_form(descriptor) == (
                descriptor.invert().canonical_form()
            )


@pytest.mark.parametrize("unknown_policy", ("exact", "wildcard", "either"))
@pytest.mark.parametrize(
    "query,candidate",
    (
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 3, 2, 4, 5), 1),
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), None),
            TetrahedralStereo((1, 3, 2, 4, 5), 1),
        ),
        (
            PlanarBondStereo((1, 2, 3, 4, 5, 6), None),
            PlanarBondStereo((2, 1, 3, 4, 5, 6), 0),
        ),
    ),
)
def test_legacy_query_oracle_matches_beta2_policy(
    query: object,
    candidate: object,
    unknown_policy: str,
) -> None:
    assert legacy_descriptor_query_matches(
        query,
        candidate,
        unknown_policy=unknown_policy,
    ) == descriptor_query_matches(
        query,
        candidate,
        unknown_policy=unknown_policy,
    )


@pytest.mark.parametrize(
    "before,after,transition",
    (
        (None, None, None),
        (None, None, SquarePlanarStereo((1, 2, 3, 4, 5), 0)),
        (None, TetrahedralStereo((1, 2, 3, 4, 5), 1), None),
        (TetrahedralStereo((1, 2, 3, 4, 5), 1), None, None),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 2, 3, 4, 5), -1),
            None,
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 2, 3, 4, 99), -1),
            None,
        ),
        (
            OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
            OctahedralStereo((1, 2, 3, 4, 5, 6, 7), -1),
            None,
        ),
    ),
)
def test_legacy_change_oracle_matches_beta2_classifier(
    before: object | None,
    after: object | None,
    transition: object | None,
) -> None:
    assert legacy_classify_stereo_change(before, after, transition) == (
        classify_stereo_change(before, after, transition)
    )


def test_comparison_contract_never_hides_unregistered_divergence() -> None:
    agreement = StereoSemanticComparison.create("identity", "same", "same")
    divergence = StereoSemanticComparison.create("classification", "new", "old")
    registered = StereoSemanticComparison.create(
        "classification",
        "RECONFIGURED:class-1",
        "INVERTED",
        expected_divergence="NON_TETRAHEDRAL_RELATION_REFINEMENT",
    )

    assert StereoSemanticsMode("compare") is StereoSemanticsMode.COMPARE
    assert agreement.registered
    assert not divergence.registered
    assert registered.registered
    assert registered.to_dict()["orbit_result"] == "RECONFIGURED:class-1"
