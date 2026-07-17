"""Frozen Beta-2 stereo semantics and dual-run comparison records.

This module intentionally does not call descriptor ``canonical_form()``,
``invert()``, equality, or the production change classifier.  It is an
independent snapshot used to detect migration divergences in later sprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Hashable, Sequence

_SQUARE_PERMUTATIONS = (
    (0, 1, 2, 3, 4),
    (0, 2, 3, 4, 1),
    (0, 3, 4, 1, 2),
    (0, 4, 1, 2, 3),
    (0, 4, 3, 2, 1),
    (0, 3, 2, 1, 4),
    (0, 2, 1, 4, 3),
    (0, 1, 4, 3, 2),
)
_TBP_INVERSION = (0, 1, 2, 3, 5, 4)
_TBP_PERMUTATIONS = (
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 5, 3, 4),
    (0, 1, 2, 4, 5, 3),
    (0, 2, 1, 3, 5, 4),
    (0, 2, 1, 5, 4, 3),
    (0, 2, 1, 4, 3, 5),
)
_OCT_INVERSION = (0, 2, 1, 3, 4, 5, 6)
_OCT_PERMUTATIONS = (
    (0, 1, 2, 3, 4, 5, 6),
    (0, 1, 2, 6, 3, 4, 5),
    (0, 1, 2, 5, 6, 3, 4),
    (0, 1, 2, 4, 5, 6, 3),
    (0, 2, 1, 4, 3, 6, 5),
    (0, 2, 1, 5, 4, 3, 6),
    (0, 2, 1, 6, 5, 4, 3),
    (0, 2, 1, 3, 6, 5, 4),
    (0, 3, 5, 2, 4, 1, 6),
    (0, 3, 5, 6, 2, 4, 1),
    (0, 3, 5, 1, 6, 2, 4),
    (0, 3, 5, 4, 1, 6, 2),
    (0, 5, 3, 1, 4, 2, 6),
    (0, 5, 3, 6, 1, 4, 2),
    (0, 5, 3, 2, 6, 1, 4),
    (0, 5, 3, 4, 2, 6, 1),
    (0, 4, 6, 3, 2, 5, 1),
    (0, 4, 6, 1, 3, 2, 5),
    (0, 4, 6, 5, 1, 3, 2),
    (0, 4, 6, 2, 5, 1, 3),
    (0, 6, 4, 3, 1, 5, 2),
    (0, 6, 4, 2, 3, 1, 5),
    (0, 6, 4, 5, 2, 3, 1),
    (0, 6, 4, 1, 5, 2, 3),
)
_BOND_PERMUTATIONS = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 2, 3, 5, 4),
    (4, 5, 3, 2, 0, 1),
    (5, 4, 3, 2, 1, 0),
)
_ATROP_INVERSION = (1, 0, 2, 3, 4, 5)
_ATROP_PERMUTATIONS = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 2, 3, 5, 4),
    (4, 5, 3, 2, 1, 0),
    (5, 4, 3, 2, 0, 1),
)
_ATOM_CLASSES = frozenset(
    {"tetrahedral", "square_planar", "trigonal_bipyramidal", "octahedral"}
)


def _reference_sort_key(value: Hashable) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _tuple_sort_key(
    values: Sequence[Hashable],
) -> tuple[tuple[str, str], ...]:
    return tuple(_reference_sort_key(value) for value in values)


def _permutation_sign(
    values: Sequence[Hashable],
    ordered: Sequence[Hashable],
) -> int:
    positions = {value: index for index, value in enumerate(ordered)}
    permutation = [positions[value] for value in values]
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _permuted_form(
    descriptor_class: str,
    atoms: Sequence[Hashable],
    parity: int,
    permutation_group: Sequence[Sequence[int]],
    inversion: Sequence[int] | None = None,
) -> tuple[Any, ...]:
    working = tuple(atoms)
    canonical_parity = parity
    if parity == -1:
        if inversion is None:
            raise ValueError(f"{descriptor_class} has no legacy inversion frame.")
        working = tuple(working[index] for index in inversion)
        canonical_parity = 1
    forms = tuple(
        tuple(working[index] for index in permutation)
        for permutation in permutation_group
    )
    return descriptor_class, canonical_parity, min(forms, key=_tuple_sort_key)


def _unknown_atom_form(
    descriptor_class: str,
    atoms: Sequence[Hashable],
) -> tuple[Any, ...]:
    return (
        descriptor_class,
        atoms[0],
        tuple(sorted(atoms[1:], key=_reference_sort_key)),
        None,
    )


def _unknown_bond_form(
    descriptor_class: str,
    atoms: Sequence[Hashable],
) -> tuple[Any, ...]:
    left = atoms[2], tuple(sorted(atoms[:2], key=_reference_sort_key))
    right = atoms[3], tuple(sorted(atoms[4:], key=_reference_sort_key))
    return descriptor_class, None, tuple(sorted((left, right), key=repr))


def legacy_canonical_state(
    descriptor_class: str,
    atoms: Sequence[Hashable],
    parity: int | None,
) -> tuple[Any, ...]:
    """Return the exact Beta-2 canonical form from raw descriptor state."""
    atoms = tuple(atoms)
    if descriptor_class == "tetrahedral":
        references = atoms[1:]
        ordered = tuple(sorted(references, key=_reference_sort_key))
        orientation = (
            None if parity is None else parity * _permutation_sign(references, ordered)
        )
        return descriptor_class, atoms[0], *ordered, orientation
    if parity is None:
        return (
            _unknown_atom_form(descriptor_class, atoms)
            if descriptor_class in _ATOM_CLASSES
            else _unknown_bond_form(descriptor_class, atoms)
        )
    if descriptor_class == "square_planar":
        return _permuted_form(
            descriptor_class,
            atoms,
            parity,
            _SQUARE_PERMUTATIONS,
        )
    if descriptor_class == "trigonal_bipyramidal":
        return _permuted_form(
            descriptor_class,
            atoms,
            parity,
            _TBP_PERMUTATIONS,
            _TBP_INVERSION,
        )
    if descriptor_class == "octahedral":
        return _permuted_form(
            descriptor_class,
            atoms,
            parity,
            _OCT_PERMUTATIONS,
            _OCT_INVERSION,
        )
    if descriptor_class == "planar_bond":
        forms = tuple(
            tuple(atoms[index] for index in permutation)
            for permutation in _BOND_PERMUTATIONS
        )
        return (
            descriptor_class,
            0,
            min(
                forms,
                key=lambda values: tuple(map(str, values)),
            ),
        )
    if descriptor_class == "atrop_bond":
        return _permuted_form(
            descriptor_class,
            atoms,
            parity,
            _ATROP_PERMUTATIONS,
            _ATROP_INVERSION,
        )
    raise ValueError(f"Unsupported legacy descriptor class: {descriptor_class!r}.")


def legacy_canonical_form(descriptor: Any) -> tuple[Any, ...]:
    return legacy_canonical_state(
        descriptor.descriptor_class,
        descriptor.atoms,
        descriptor.parity,
    )


def legacy_inverted_form(descriptor: Any) -> tuple[Any, ...]:
    descriptor_class = descriptor.descriptor_class
    atoms = tuple(descriptor.atoms)
    parity = descriptor.parity
    if parity is None or descriptor_class == "square_planar":
        return legacy_canonical_state(descriptor_class, atoms, parity)
    if descriptor_class == "planar_bond":
        atoms = (atoms[1], atoms[0], *atoms[2:])
        return legacy_canonical_state(descriptor_class, atoms, 0)
    return legacy_canonical_state(descriptor_class, atoms, -parity)


def legacy_same_configuration(left: Any, right: Any) -> bool:
    return type(left) is type(right) and legacy_canonical_form(left) == (
        legacy_canonical_form(right)
    )


def _legacy_descriptor_id(descriptor: Any) -> str:
    if descriptor.descriptor_class in _ATOM_CLASSES:
        return f"atom:{descriptor.atoms[0]}"
    centers = sorted(descriptor.atoms[2:4], key=str)
    return f"bond:{centers[0]}-{centers[1]}"


def _peripheral_references(descriptor: Any) -> tuple[Hashable, ...]:
    return (
        tuple(descriptor.atoms[1:])
        if descriptor.descriptor_class in _ATOM_CLASSES
        else (*descriptor.atoms[:2], *descriptor.atoms[4:])
    )


def _legacy_aligned_state(
    old: Any,
    new: Any,
) -> tuple[tuple[Hashable, ...], int | None] | None:
    if type(old) is not type(new) or _legacy_descriptor_id(old) != (
        _legacy_descriptor_id(new)
    ):
        return None
    old_refs = _peripheral_references(old)
    new_refs = _peripheral_references(new)
    removed = list(set(old_refs) - set(new_refs))
    added = list(set(new_refs) - set(old_refs))
    if len(removed) != 1 or len(added) != 1:
        return None
    atoms = list(old.atoms)
    atoms[atoms.index(removed[0])] = added[0]
    return tuple(atoms), old.parity


def legacy_classify_stereo_change(
    old: Any | None,
    new: Any | None,
    transition: Any | None = None,
) -> str:
    """Reproduce the exact Beta-2 reaction stereo label."""
    if old is None and new is None:
        return "FLEETING" if transition is not None else "UNSPECIFIED"
    if old is None:
        return "FORMED"
    if new is None:
        return "BROKEN"
    if legacy_same_configuration(old, new):
        return "RETAINED"
    if type(old) is type(new) and legacy_inverted_form(old) == (
        legacy_canonical_form(new)
    ):
        return "INVERTED"
    aligned = _legacy_aligned_state(old, new)
    if aligned is not None:
        atoms, parity = aligned
        aligned_form = legacy_canonical_state(old.descriptor_class, atoms, parity)
        if aligned_form == legacy_canonical_form(new):
            return "RETAINED"
        aligned_state = SimpleNamespace(
            descriptor_class=old.descriptor_class,
            atoms=atoms,
            parity=parity,
            provenance=None,
        )
        if legacy_inverted_form(aligned_state) == legacy_canonical_form(new):
            return "INVERTED"
    return "UNSPECIFIED"


def legacy_apply_stereo_change(change: Any, descriptor: Any) -> Any | None:
    """Reproduce Beta-2 same-locus reaction stereo propagation.

    This deliberately retains the former universal descriptor-inversion
    behavior as a comparison oracle. It does not call the orbit relation or
    reaction application methods.
    """
    before, after = change.before, change.after
    if before is None:
        return after
    if after is None:
        return None
    if descriptor.parity is None:
        return type(after)(after.atoms, None, after.provenance)
    descriptor_form = legacy_canonical_form(descriptor)
    if descriptor_form == legacy_canonical_form(before):
        return after
    if descriptor_form == legacy_inverted_form(before):
        if after.parity is None:
            return after
        if after.descriptor_class == "square_planar":
            return after
        if after.descriptor_class == "planar_bond":
            atoms = (after.atoms[1], after.atoms[0], *after.atoms[2:])
            return type(after)(atoms, 0, after.provenance)
        return type(after)(after.atoms, -after.parity, after.provenance)
    return type(after)(after.atoms, None, after.provenance)


def legacy_descriptor_query_matches(
    query: Any,
    candidate: Any,
    *,
    unknown_policy: str = "exact",
) -> bool:
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    if type(query) is not type(candidate):
        return False
    candidate_form = legacy_canonical_form(candidate)
    if unknown_policy == "either" and query.parity is not None:
        return legacy_canonical_form(query) == candidate_form or (
            legacy_inverted_form(query) == candidate_form
        )
    if query.parity is not None or unknown_policy == "exact":
        return legacy_canonical_form(query) == candidate_form
    unknown_candidate = legacy_canonical_state(
        candidate.descriptor_class,
        candidate.atoms,
        None,
    )
    return legacy_canonical_form(query) == unknown_candidate


class StereoSemanticsMode(str, Enum):
    ORBIT = "orbit"
    LEGACY = "legacy"
    COMPARE = "compare"


@dataclass(frozen=True)
class StereoSemanticComparison:
    """One stage-local orbit/legacy comparison with no fallback semantics."""

    stage: str
    orbit_result: Any
    legacy_result: Any
    agreement: bool
    expected_divergence: str | None = None

    @classmethod
    def create(
        cls,
        stage: str,
        orbit_result: Any,
        legacy_result: Any,
        *,
        expected_divergence: str | None = None,
    ) -> "StereoSemanticComparison":
        return cls(
            stage,
            orbit_result,
            legacy_result,
            orbit_result == legacy_result,
            expected_divergence,
        )

    @property
    def registered(self) -> bool:
        return self.agreement or self.expected_divergence is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "orbit_result": self.orbit_result,
            "legacy_result": self.legacy_result,
            "agreement": self.agreement,
            "expected_divergence": self.expected_divergence,
            "registered": self.registered,
        }


__all__ = [
    "StereoSemanticComparison",
    "StereoSemanticsMode",
    "legacy_canonical_form",
    "legacy_canonical_state",
    "legacy_classify_stereo_change",
    "legacy_apply_stereo_change",
    "legacy_descriptor_query_matches",
    "legacy_inverted_form",
    "legacy_same_configuration",
]
