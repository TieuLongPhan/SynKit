"""RDKit-independent proofs for the coupled-framework value."""

from __future__ import annotations

import itertools

import pytest

from synkit.Graph.Stereo import (
    FrameworkFrame,
    FrameworkStereo,
    GlobalStereoCertificate,
    GlobalStereoInformationState,
)
from synkit.Graph.Stereo._descriptor_core import permutation_sign


def _value(orientation: int | None = 1) -> FrameworkStereo:
    return FrameworkStereo(
        frozenset({1, 2, 3, 4, 5}),
        (FrameworkFrame(1, (2, 3, 4, "@H:1"), 1),),
        orientation,
        "proof",
    )


def test_positive_negative_and_unspecified_are_distinct() -> None:
    positive = _value(1)
    negative = _value(-1)
    unspecified = _value(None)

    assert positive != negative
    assert positive != unspecified
    assert positive.invert() == negative
    assert positive.invert().invert() == positive
    assert unspecified.invert() == unspecified


def test_all_local_frame_representations_have_one_identity() -> None:
    reference = _value(1)
    frame = reference.frames[0]

    for ordering in itertools.permutations(frame.references):
        equivalent = FrameworkStereo(
            reference.support_atoms,
            (
                FrameworkFrame(
                    frame.center,
                    ordering,
                    frame.relation
                    * permutation_sign(frame.references, ordering),
                ),
            ),
            1,
            "different provenance",
        )
        assert equivalent == reference
        assert hash(equivalent) == hash(reference)


def test_relabel_and_inverse_relabel_restore_the_value() -> None:
    value = _value(1)
    mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    inverse = {target: source for source, target in mapping.items()}

    assert value.relabel(mapping).relabel(inverse) == value


def test_certificate_promotions_are_explicit() -> None:
    unspecified = _value(None)
    certificate = GlobalStereoCertificate(
        unspecified,
        GlobalStereoInformationState.NECESSARILY_CHIRAL,
        True,
        "proof",
    )

    assert certificate.necessarily_chiral
    with pytest.raises(ValueError, match="inconsistent"):
        GlobalStereoCertificate(
            _value(1),
            GlobalStereoInformationState.CONFIGURED_NEGATIVE,
            True,
            "proof",
        )


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: FrameworkFrame(1, (2, 3, 4), 1),
        lambda: FrameworkFrame(1, (2, 3, 4, "@H:2"), 1),
        lambda: FrameworkFrame(1, (2, 3, 4, "@H:1"), 0),
        lambda: FrameworkStereo(
            frozenset({1, 2, 3, 4}),
            (FrameworkFrame(1, (2, 3, 5, "@H:1"), 1),),
            1,
        ),
    ),
)
def test_malformed_values_fail_closed(constructor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor()  # type: ignore[operator]
