"""Finite permutation-orbit semantics for local stereochemical frames.

The kernel is deliberately independent of descriptor, RDKit, rule, and graph
objects.  A local configuration is an ordered frame modulo the preserving
permutation group of its shape.  Concrete relation witnesses are retained;
double cosets are invariant classifications, not composable operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations
from typing import Any, Hashable, Mapping, Sequence


def _stable_key(values: Sequence[Hashable]) -> tuple[tuple[str, str], ...]:
    return tuple((type(value).__name__, repr(value)) for value in values)


@dataclass(frozen=True, order=True)
class Permutation:
    """A finite position permutation using ``result[i] = source[image[i]]``."""

    image: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", tuple(self.image))
        if tuple(sorted(self.image)) != tuple(range(len(self.image))):
            raise ValueError("A permutation image must be a bijection on 0..n-1.")

    @property
    def degree(self) -> int:
        return len(self.image)

    @classmethod
    def identity(cls, degree: int) -> "Permutation":
        if degree < 0:
            raise ValueError("Permutation degree must be non-negative.")
        return cls(tuple(range(degree)))

    def apply(self, values: Sequence[Any]) -> tuple[Any, ...]:
        if len(values) != self.degree:
            raise ValueError("Permutation and frame arities differ.")
        return tuple(values[index] for index in self.image)

    def then(self, after: "Permutation") -> "Permutation":
        """Return ``after ∘ self`` under :meth:`apply`."""
        if self.degree != after.degree:
            raise ValueError("Only permutations of equal degree compose.")
        return Permutation(tuple(self.image[index] for index in after.image))

    def inverse(self) -> "Permutation":
        inverse = [0] * self.degree
        for output, source in enumerate(self.image):
            inverse[source] = output
        return Permutation(tuple(inverse))


@dataclass(frozen=True)
class PermutationGroup:
    """A validated finite permutation group of one fixed degree."""

    name: str
    degree: int
    elements: tuple[Permutation, ...]
    _closed_by_construction: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        elements = tuple(sorted(set(self.elements)))
        object.__setattr__(self, "elements", elements)
        if self.degree < 0 or not elements:
            raise ValueError("A finite permutation group must be non-empty.")
        if any(element.degree != self.degree for element in elements):
            raise ValueError("Every group element must have the declared degree.")
        images = {element.image for element in elements}
        identity = Permutation.identity(self.degree)
        if identity.image not in images:
            raise ValueError("A permutation group must contain its identity.")
        if any(element.inverse().image not in images for element in elements):
            raise ValueError("A permutation group must contain every inverse.")
        if not self._closed_by_construction:
            for left in elements:
                for right in elements:
                    image = tuple(left.image[index] for index in right.image)
                    if image not in images:
                        raise ValueError("A permutation group must be closed.")

    def __contains__(self, value: object) -> bool:
        return isinstance(value, Permutation) and value in self.elements

    def orbit(self, frame: Sequence[Hashable]) -> tuple[tuple[Hashable, ...], ...]:
        if len(frame) != self.degree:
            raise ValueError("Group and frame arities differ.")
        forms = {element.apply(frame) for element in self.elements}
        return tuple(sorted(forms, key=_stable_key))

    def canonical(self, frame: Sequence[Hashable]) -> tuple[Hashable, ...]:
        return self.orbit(frame)[0]


def _lift_permutation(
    degree: int,
    positions: Sequence[int],
    local_image: Sequence[int],
) -> Permutation:
    image = list(range(degree))
    for output, local_source in enumerate(local_image):
        image[positions[output]] = positions[local_source]
    return Permutation(tuple(image))


def _symmetric_group(
    name: str,
    degree: int,
    positions: Sequence[int],
) -> PermutationGroup:
    elements = tuple(
        _lift_permutation(degree, positions, local_image)
        for local_image in permutations(range(len(positions)))
    )
    # All bijections of a finite position set are closed under composition.
    return PermutationGroup(name, degree, elements, _closed_by_construction=True)


def _permutation_sign(image: Sequence[int]) -> int:
    inversions = sum(
        image[left] > image[right]
        for left in range(len(image))
        for right in range(left + 1, len(image))
    )
    return -1 if inversions % 2 else 1


def _alternating_group(
    name: str,
    degree: int,
    positions: Sequence[int],
) -> PermutationGroup:
    elements = tuple(
        _lift_permutation(degree, positions, local_image)
        for local_image in permutations(range(len(positions)))
        if _permutation_sign(local_image) == 1
    )
    # Even permutations are the kernel of the sign homomorphism and are closed.
    return PermutationGroup(name, degree, elements, _closed_by_construction=True)


def _group_from_images(
    name: str,
    images: Sequence[Sequence[int]],
) -> PermutationGroup:
    elements = tuple(Permutation(tuple(image)) for image in images)
    return PermutationGroup(name, len(elements[0].image), elements)


def _generated_group(
    name: str,
    degree: int,
    generators: Sequence[Permutation],
) -> PermutationGroup:
    identity = Permutation.identity(degree)
    discovered = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            for candidate in (current.then(generator), generator.then(current)):
                if candidate not in discovered:
                    discovered.add(candidate)
                    frontier.append(candidate)
    # The fixed-point iteration stops only after the generator closure is complete.
    return PermutationGroup(
        name,
        degree,
        tuple(discovered),
        _closed_by_construction=True,
    )


class StereoSpecification(str, Enum):
    FIXED = "fixed"
    UNSPECIFIED = "unspecified"


class StereoRelationKind(str, Enum):
    EQUIVALENT = "equivalent"
    OPPOSITE = "opposite"
    RECONFIGURED = "reconfigured"
    UNRELATED = "unrelated"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class ShapeDefinition:
    """Geometry-specific frame positions and admissible permutation groups."""

    name: str
    frame_arity: int
    locus_positions: frozenset[int]
    preserving_group: PermutationGroup
    unspecified_group: PermutationGroup
    opposite_permutation: Permutation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "locus_positions", frozenset(self.locus_positions))
        if self.preserving_group.degree != self.frame_arity:
            raise ValueError("The preserving group has the wrong frame arity.")
        if self.unspecified_group.degree != self.frame_arity:
            raise ValueError("The unspecified group has the wrong frame arity.")
        if not self.locus_positions <= frozenset(range(self.frame_arity)):
            raise ValueError("Locus positions must lie inside the local frame.")
        preserving = set(self.preserving_group.elements)
        unspecified = set(self.unspecified_group.elements)
        if not preserving <= unspecified:
            raise ValueError("The unspecified group must contain the preserving group.")
        if self.opposite_permutation is not None:
            if self.opposite_permutation not in unspecified:
                raise ValueError("An opposite witness must be admissible for the shape.")
            if self.opposite_permutation in preserving:
                raise ValueError("An opposite witness cannot preserve configuration.")

    @property
    def reference_positions(self) -> tuple[int, ...]:
        return tuple(
            position
            for position in range(self.frame_arity)
            if position not in self.locus_positions
        )

    def group_for(self, specification: StereoSpecification) -> PermutationGroup:
        return (
            self.preserving_group
            if specification is StereoSpecification.FIXED
            else self.unspecified_group
        )


@dataclass(frozen=True)
class PermutationWitness:
    """A replayable relative permutation between two concrete frames."""

    shape: str
    permutation: Permutation

    def apply(self, frame: Sequence[Hashable]) -> tuple[Hashable, ...]:
        return self.permutation.apply(frame)

    def then(self, after: "PermutationWitness") -> "PermutationWitness":
        if self.shape != after.shape:
            raise ValueError("Stereo witnesses compose only within one shape.")
        return PermutationWitness(
            self.shape,
            self.permutation.then(after.permutation),
        )


@dataclass(frozen=True)
class StereoRelation:
    kind: StereoRelationKind
    shape: str | None
    class_id: tuple[int, ...] | None = None
    witness: PermutationWitness | None = None
    source_canonical: tuple[Hashable, ...] | None = None
    target_canonical: tuple[Hashable, ...] | None = None

    @property
    def replayable(self) -> bool:
        return self.witness is not None


@dataclass(frozen=True, eq=False)
class StereoConfiguration:
    """One local frame modulo the fixed or unspecified group of its shape."""

    shape: str
    frame: tuple[Hashable, ...]
    specification: StereoSpecification = StereoSpecification.FIXED

    def __post_init__(self) -> None:
        if not isinstance(self.specification, StereoSpecification):
            object.__setattr__(
                self,
                "specification",
                StereoSpecification(self.specification),
            )
        object.__setattr__(self, "frame", tuple(self.frame))
        definition = self.definition
        if len(self.frame) != definition.frame_arity:
            raise ValueError(
                f"Shape {self.shape!r} requires {definition.frame_arity} frame values."
            )
        try:
            unique = len(set(self.frame))
        except TypeError as exc:
            raise TypeError("Stereo frame references must be hashable.") from exc
        if unique != len(self.frame):
            raise ValueError("Stereo frame references must be distinct.")

    @property
    def definition(self) -> ShapeDefinition:
        try:
            return SHAPE_DEFINITIONS[self.shape]
        except KeyError as exc:
            raise ValueError(f"Unknown stereo shape {self.shape!r}.") from exc

    @property
    def canonical_frame(self) -> tuple[Hashable, ...]:
        return self.definition.group_for(self.specification).canonical(self.frame)

    def canonical_form(self) -> tuple[Any, ...]:
        return self.shape, self.specification.value, self.canonical_frame

    def same_configuration(self, other: object) -> bool:
        return (
            isinstance(other, StereoConfiguration)
            and self.shape == other.shape
            and self.specification is other.specification
            and self.canonical_frame == other.canonical_frame
        )

    def __eq__(self, other: object) -> bool:
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def relabel(self, mapping: Mapping[Hashable, Hashable]) -> "StereoConfiguration":
        return StereoConfiguration(
            self.shape,
            tuple(mapping.get(reference, reference) for reference in self.frame),
            self.specification,
        )

    def replace_reference(
        self,
        old: Hashable,
        new: Hashable,
    ) -> "StereoConfiguration":
        return self.replace_references({old: new})

    def replace_references(
        self,
        replacements: Mapping[Hashable, Hashable],
    ) -> "StereoConfiguration":
        unknown = set(replacements) - set(self.frame)
        if unknown:
            raise ValueError(f"Replacement sources are absent: {sorted(map(repr, unknown))}.")
        protected = {
            self.frame[position]
            for position in self.definition.locus_positions
        }
        replaced_loci = protected & set(replacements)
        if replaced_loci:
            raise ValueError("Reference replacement cannot replace descriptor loci.")
        return StereoConfiguration(
            self.shape,
            tuple(replacements.get(reference, reference) for reference in self.frame),
            self.specification,
        )

    def _witness_to(self, other: "StereoConfiguration") -> Permutation | None:
        for permutation in self.definition.unspecified_group.elements:
            if permutation.apply(self.frame) == other.frame:
                return permutation
        return None

    def _double_coset_id(self, witness: Permutation) -> tuple[int, ...]:
        group = self.definition.preserving_group.elements
        images = {
            left.then(witness).then(right).image
            for left in group
            for right in group
        }
        return min(images)

    def _opposite_class_id(self) -> tuple[int, ...] | None:
        opposite = self.definition.opposite_permutation
        return None if opposite is None else self._double_coset_id(opposite)

    def relation_to(self, other: object) -> StereoRelation:
        if not isinstance(other, StereoConfiguration) or self.shape != other.shape:
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        witness = self._witness_to(other)
        if witness is None:
            return StereoRelation(
                StereoRelationKind.UNRELATED,
                self.shape,
                source_canonical=self.canonical_frame,
                target_canonical=other.canonical_frame,
            )
        wrapped = PermutationWitness(self.shape, witness)
        if (
            self.specification is StereoSpecification.UNSPECIFIED
            or other.specification is StereoSpecification.UNSPECIFIED
        ):
            return StereoRelation(
                StereoRelationKind.UNSPECIFIED,
                self.shape,
                witness=wrapped,
                source_canonical=self.canonical_frame,
                target_canonical=other.canonical_frame,
            )
        class_id = self._double_coset_id(witness)
        if witness in self.definition.preserving_group:
            kind = StereoRelationKind.EQUIVALENT
        elif class_id == self._opposite_class_id():
            kind = StereoRelationKind.OPPOSITE
        else:
            kind = StereoRelationKind.RECONFIGURED
        return StereoRelation(
            kind,
            self.shape,
            class_id,
            wrapped,
            self.canonical_frame,
            other.canonical_frame,
        )

    def opposite(self) -> "StereoConfiguration":
        opposite = self.definition.opposite_permutation
        if opposite is None:
            raise ValueError(f"Shape {self.shape!r} has no binary opposite operation.")
        if self.specification is StereoSpecification.UNSPECIFIED:
            return self
        return StereoConfiguration(
            self.shape,
            opposite.apply(self.frame),
            self.specification,
        )


_TETRA_FIXED = _alternating_group("tetrahedral:A4", 5, (1, 2, 3, 4))
_TETRA_ALL = _symmetric_group("tetrahedral:S4", 5, (1, 2, 3, 4))
_TETRA_OPPOSITE = Permutation((0, 2, 1, 3, 4))

_SQUARE_FIXED = _group_from_images(
    "square_planar:D4",
    (
        (0, 1, 2, 3, 4),
        (0, 2, 3, 4, 1),
        (0, 3, 4, 1, 2),
        (0, 4, 1, 2, 3),
        (0, 4, 3, 2, 1),
        (0, 3, 2, 1, 4),
        (0, 2, 1, 4, 3),
        (0, 1, 4, 3, 2),
    ),
)
_SQUARE_ALL = _symmetric_group("square_planar:S4", 5, (1, 2, 3, 4))

_TBP_FIXED = _group_from_images(
    "trigonal_bipyramidal:D3",
    (
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 5, 3, 4),
        (0, 1, 2, 4, 5, 3),
        (0, 2, 1, 3, 5, 4),
        (0, 2, 1, 5, 4, 3),
        (0, 2, 1, 4, 3, 5),
    ),
)
_TBP_ALL = _symmetric_group("trigonal_bipyramidal:S5", 6, (1, 2, 3, 4, 5))

_OCT_FIXED = _group_from_images(
    "octahedral:O",
    (
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
    ),
)
_OCT_ALL = _symmetric_group("octahedral:S6", 7, (1, 2, 3, 4, 5, 6))

_PLANAR_BOND_FIXED = _group_from_images(
    "planar_bond:V4",
    (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 0, 1),
        (5, 4, 3, 2, 1, 0),
    ),
)
_PLANAR_BOND_ALL = _generated_group(
    "planar_bond:unknown",
    6,
    (
        Permutation((1, 0, 2, 3, 5, 4)),
        Permutation((4, 5, 3, 2, 0, 1)),
        Permutation((1, 0, 2, 3, 4, 5)),
    ),
)
_BOND_OPPOSITE = Permutation((1, 0, 2, 3, 4, 5))

_ATROP_BOND_FIXED = _group_from_images(
    "atrop_bond:V4",
    (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 1, 0),
        (5, 4, 3, 2, 0, 1),
    ),
)
_ATROP_BOND_ALL = _generated_group(
    "atrop_bond:unknown",
    6,
    (
        Permutation((1, 0, 2, 3, 5, 4)),
        Permutation((4, 5, 3, 2, 1, 0)),
        Permutation((1, 0, 2, 3, 4, 5)),
    ),
)


SHAPE_DEFINITIONS: Mapping[str, ShapeDefinition] = {
    "tetrahedral": ShapeDefinition(
        "tetrahedral", 5, frozenset({0}), _TETRA_FIXED, _TETRA_ALL, _TETRA_OPPOSITE
    ),
    "square_planar": ShapeDefinition(
        "square_planar", 5, frozenset({0}), _SQUARE_FIXED, _SQUARE_ALL
    ),
    "trigonal_bipyramidal": ShapeDefinition(
        "trigonal_bipyramidal", 6, frozenset({0}), _TBP_FIXED, _TBP_ALL
    ),
    "octahedral": ShapeDefinition(
        "octahedral", 7, frozenset({0}), _OCT_FIXED, _OCT_ALL
    ),
    "planar_bond": ShapeDefinition(
        "planar_bond",
        6,
        frozenset({2, 3}),
        _PLANAR_BOND_FIXED,
        _PLANAR_BOND_ALL,
        _BOND_OPPOSITE,
    ),
    "atrop_bond": ShapeDefinition(
        "atrop_bond",
        6,
        frozenset({2, 3}),
        _ATROP_BOND_FIXED,
        _ATROP_BOND_ALL,
        _BOND_OPPOSITE,
    ),
}


__all__ = [
    "Permutation",
    "PermutationGroup",
    "PermutationWitness",
    "SHAPE_DEFINITIONS",
    "ShapeDefinition",
    "StereoConfiguration",
    "StereoRelation",
    "StereoRelationKind",
    "StereoSpecification",
]
