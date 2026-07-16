"""Relative stereodescriptors adapted for mapped SynKit Lewis-state graphs.

Stereo identity is relative and permutation-aware; CIP labels are
intentionally outside descriptor identity.  The non-tetrahedral permutation
groups are adapted from StereoMolGraph commit
``2189f610f23eaaf992e2e01a12ea4d0532496601`` (MIT, copyright (c) 2025
Maxim Papusha); the corresponding notice is shipped in
``LICENSES/StereoMolGraph-MIT.txt``.

SynKit keeps its own descriptor values because they participate in executable
reaction rules and retain Lewis/electron state in the surrounding graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

VirtualReferenceKind = Literal["H", "LP"]
Reference = int | str

_VIRTUAL_REFERENCE_PATTERN = re.compile(r"^@(H|LP):(-?\d+)$")


@dataclass(frozen=True)
class VirtualStereoReference:
    """Parsed identity of a ligand not represented by a graph atom.

    The serialized form remains a compact string so descriptors retain their
    existing JSON/GML representation.  ``kind`` is deliberately part of the
    identity: a hydrogen and a lone pair at the same center are not
    interchangeable stereochemical references.
    """

    kind: VirtualReferenceKind
    center: int

    def __post_init__(self) -> None:
        if self.kind not in {"H", "LP"}:
            raise ValueError("Virtual stereo reference kind must be 'H' or 'LP'.")
        if type(self.center) is not int:
            raise TypeError("Virtual stereo reference centers must be integers.")

    def __str__(self) -> str:
        return f"@{self.kind}:{self.center}"


def virtual_reference(kind: VirtualReferenceKind, center: int) -> str:
    """Return the canonical serialized identity for a virtual ligand."""
    if kind not in {"H", "LP"}:
        raise ValueError("Virtual stereo reference kind must be 'H' or 'LP'.")
    if type(center) is not int:
        raise TypeError("Virtual stereo reference centers must be integers.")
    return str(VirtualStereoReference(kind, center))


def parse_virtual_reference(value: object) -> VirtualStereoReference | None:
    """Parse a canonical ``@H:<center>`` or ``@LP:<center>`` reference."""
    if not isinstance(value, str):
        return None
    match = _VIRTUAL_REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        return None
    kind, center = match.groups()
    parsed_kind: VirtualReferenceKind = "H" if kind == "H" else "LP"
    return VirtualStereoReference(parsed_kind, int(center))


# Public capability boundaries. ``SUPPORTED`` means graph storage, relative
# identity, relabeling, rule matching/rewriting, and JSON/GML serialization.
# RDKit and coordinate inference are deliberately narrower adapters.
SUPPORTED_STEREO_DESCRIPTOR_CLASSES = frozenset(
    {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "planar_bond",
        "atrop_bond",
    }
)
RDKIT_STEREO_DESCRIPTOR_CLASSES = frozenset(
    {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "planar_bond",
    }
)
DEFERRED_STEREO_DESCRIPTOR_CLASSES = frozenset(
    {"rigid_bond_33", "rigid_bond_23", "rigid_bond_13", "rigid_bond_12"}
)


def _relabel_reference(value: Reference, mapping: Mapping[int, int]) -> Reference:
    if type(value) is int:
        return mapping.get(value, value)
    virtual = parse_virtual_reference(value)
    if virtual is not None:
        return virtual_reference(
            virtual.kind,
            mapping.get(virtual.center, virtual.center),
        )
    return value


def _validate_reference(value: object, *, owner: int, position: int) -> None:
    if type(value) is int:
        return
    virtual = parse_virtual_reference(value)
    if virtual is None:
        raise ValueError(
            "Stereo references must be integer atom IDs or canonical "
            "'@H:<center>'/'@LP:<center>' virtual references; "
            f"invalid value at position {position}: {value!r}."
        )
    if virtual.center != owner:
        raise ValueError(
            f"Virtual stereo reference {value!r} belongs to center "
            f"{virtual.center}, not ligand owner {owner}."
        )


def _validate_atom_references(atoms: Sequence[Reference]) -> None:
    center = atoms[0]
    if type(center) is not int:
        raise ValueError("Atom-centered stereo requires an integer center ID.")
    for position, value in enumerate(atoms[1:], start=1):
        _validate_reference(value, owner=center, position=position)


def _validate_bond_references(atoms: Sequence[Reference]) -> None:
    left, right = atoms[2:4]
    if type(left) is not int or type(right) is not int:
        raise ValueError("Bond-centered stereo requires integer central atom IDs.")
    for position, value in enumerate(atoms[:2]):
        _validate_reference(value, owner=left, position=position)
    for position, value in enumerate(atoms[4:], start=4):
        _validate_reference(value, owner=right, position=position)


def _reference_sort_key(value: Reference) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _tuple_sort_key(values: Sequence[Reference]) -> tuple[tuple[str, str], ...]:
    return tuple(_reference_sort_key(value) for value in values)


def _permuted_canonical_form(
    descriptor_class: str,
    atoms: Sequence[Reference],
    parity: int,
    permutations: Sequence[Sequence[int]],
    inversion: Sequence[int] | None = None,
) -> tuple[Any, ...]:
    """Return a canonical relative form under a descriptor symmetry group."""
    working = tuple(atoms)
    canonical_parity = parity
    if parity == -1:
        if inversion is None:
            raise ValueError(f"{descriptor_class} does not define an inverse.")
        working = tuple(working[index] for index in inversion)
        canonical_parity = 1
    forms = tuple(
        tuple(working[index] for index in permutation) for permutation in permutations
    )
    return (
        descriptor_class,
        canonical_parity,
        min(forms, key=_tuple_sort_key),
    )


def _unknown_atom_form(
    descriptor_class: str,
    atoms: Sequence[Reference],
) -> tuple[Any, ...]:
    """Keep an unknown orientation attached to its atom locus and ligands."""
    return (
        descriptor_class,
        atoms[0],
        tuple(sorted(atoms[1:], key=_reference_sort_key)),
        None,
    )


def _unknown_bond_form(
    descriptor_class: str,
    atoms: Sequence[Reference],
) -> tuple[Any, ...]:
    """Keep an unknown orientation attached to its bond and endpoint groups."""
    left = (atoms[2], tuple(sorted(atoms[:2], key=_reference_sort_key)))
    right = (atoms[3], tuple(sorted(atoms[4:], key=_reference_sort_key)))
    return (
        descriptor_class,
        None,
        tuple(sorted((left, right), key=repr)),
    )


def _descriptor_dict(descriptor: Any) -> dict[str, Any]:
    return {
        "descriptor_class": descriptor.descriptor_class,
        "atoms": list(descriptor.atoms),
        "parity": descriptor.parity,
        "provenance": descriptor.provenance,
    }


def _permutation_sign(values: Sequence[Reference], ordered: Sequence[Reference]) -> int:
    positions = {value: index for index, value in enumerate(ordered)}
    permutation = [positions[value] for value in values]
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True, eq=False)
class TetrahedralStereo:
    atoms: tuple[Reference, Reference, Reference, Reference, Reference]
    parity: int | None
    provenance: str | None = None

    descriptor_class = "tetrahedral"

    def __post_init__(self) -> None:
        if len(self.atoms) != 5:
            raise ValueError("Tetrahedral stereo requires center plus four references.")
        _validate_atom_references(self.atoms)
        if self.parity not in (-1, 1, None):
            raise ValueError("Tetrahedral parity must be -1, 1, or None.")
        if len(set(self.atoms[1:])) != 4:
            raise ValueError("Tetrahedral references must be distinct.")

    @property
    def center(self) -> Reference:
        return self.atoms[0]

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if isinstance(value, int))

    def canonical_form(self) -> tuple[Any, ...]:
        refs = self.atoms[1:]
        ordered = tuple(sorted(refs, key=_reference_sort_key))
        parity = (
            None
            if self.parity is None
            else self.parity * _permutation_sign(refs, ordered)
        )
        return (self.descriptor_class, self.center, *ordered, parity)

    def invert(self) -> "TetrahedralStereo":
        return (
            self
            if self.parity is None
            else TetrahedralStereo(self.atoms, -self.parity, self.provenance)
        )

    def relabel(self, mapping: Mapping[int, int]) -> "TetrahedralStereo":
        return TetrahedralStereo(
            tuple(_relabel_reference(value, mapping) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TetrahedralStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class SquarePlanarStereo:
    """Relative square-planar atom stereo (center plus four cyclic ligands)."""

    atoms: tuple[Reference, Reference, Reference, Reference, Reference]
    parity: int | None = 0
    provenance: str | None = None

    descriptor_class = "square_planar"
    _PERMUTATIONS = (
        (0, 1, 2, 3, 4),
        (0, 2, 3, 4, 1),
        (0, 3, 4, 1, 2),
        (0, 4, 1, 2, 3),
        (0, 4, 3, 2, 1),
        (0, 3, 2, 1, 4),
        (0, 2, 1, 4, 3),
        (0, 1, 4, 3, 2),
    )

    def __post_init__(self) -> None:
        if len(self.atoms) != 5:
            raise ValueError(
                "Square-planar stereo requires center plus four references."
            )
        _validate_atom_references(self.atoms)
        if self.parity not in (0, None):
            raise ValueError("Square-planar parity must be 0 or None.")

    @property
    def center(self) -> Reference:
        return self.atoms[0]

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if isinstance(value, int))

    def canonical_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            return _unknown_atom_form(self.descriptor_class, self.atoms)
        return _permuted_canonical_form(
            self.descriptor_class,
            self.atoms,
            self.parity,
            self._PERMUTATIONS,
        )

    def invert(self) -> "SquarePlanarStereo":
        return self

    def relabel(self, mapping: Mapping[int, int]) -> "SquarePlanarStereo":
        return SquarePlanarStereo(
            tuple(_relabel_reference(value, mapping) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SquarePlanarStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class TrigonalBipyramidalStereo:
    """Relative trigonal-bipyramidal atom stereo."""

    atoms: tuple[
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
    ]
    parity: int | None
    provenance: str | None = None

    descriptor_class = "trigonal_bipyramidal"
    _INVERSION = (0, 1, 2, 3, 5, 4)
    _PERMUTATIONS = (
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 5, 3, 4),
        (0, 1, 2, 4, 5, 3),
        (0, 2, 1, 3, 5, 4),
        (0, 2, 1, 5, 4, 3),
        (0, 2, 1, 4, 3, 5),
    )

    def __post_init__(self) -> None:
        if len(self.atoms) != 6:
            raise ValueError(
                "Trigonal-bipyramidal stereo requires center plus five references."
            )
        _validate_atom_references(self.atoms)
        if self.parity not in (-1, 1, None):
            raise ValueError("Trigonal-bipyramidal parity must be -1, 1, or None.")

    @property
    def center(self) -> Reference:
        return self.atoms[0]

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if isinstance(value, int))

    def canonical_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            return _unknown_atom_form(self.descriptor_class, self.atoms)
        return _permuted_canonical_form(
            self.descriptor_class,
            self.atoms,
            self.parity,
            self._PERMUTATIONS,
            self._INVERSION,
        )

    def invert(self) -> "TrigonalBipyramidalStereo":
        return (
            self
            if self.parity is None
            else TrigonalBipyramidalStereo(self.atoms, -self.parity, self.provenance)
        )

    def relabel(self, mapping: Mapping[int, int]) -> "TrigonalBipyramidalStereo":
        return TrigonalBipyramidalStereo(
            tuple(_relabel_reference(value, mapping) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TrigonalBipyramidalStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class OctahedralStereo:
    """Relative octahedral atom stereo."""

    atoms: tuple[
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
    ]
    parity: int | None
    provenance: str | None = None

    descriptor_class = "octahedral"
    _INVERSION = (0, 2, 1, 3, 4, 5, 6)
    _PERMUTATIONS = (
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

    def __post_init__(self) -> None:
        if len(self.atoms) != 7:
            raise ValueError("Octahedral stereo requires center plus six references.")
        _validate_atom_references(self.atoms)
        if self.parity not in (-1, 1, None):
            raise ValueError("Octahedral parity must be -1, 1, or None.")

    @property
    def center(self) -> Reference:
        return self.atoms[0]

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if isinstance(value, int))

    def canonical_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            return _unknown_atom_form(self.descriptor_class, self.atoms)
        return _permuted_canonical_form(
            self.descriptor_class,
            self.atoms,
            self.parity,
            self._PERMUTATIONS,
            self._INVERSION,
        )

    def invert(self) -> "OctahedralStereo":
        return (
            self
            if self.parity is None
            else OctahedralStereo(self.atoms, -self.parity, self.provenance)
        )

    def relabel(self, mapping: Mapping[int, int]) -> "OctahedralStereo":
        return OctahedralStereo(
            tuple(_relabel_reference(value, mapping) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, OctahedralStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class PlanarBondStereo:
    atoms: tuple[Reference, Reference, Reference, Reference, Reference, Reference]
    parity: int | None = 0
    provenance: str | None = None

    descriptor_class = "planar_bond"
    _PERMUTATIONS = (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 0, 1),
        (5, 4, 3, 2, 1, 0),
    )

    def __post_init__(self) -> None:
        if len(self.atoms) != 6:
            raise ValueError("Planar-bond stereo requires six references.")
        _validate_bond_references(self.atoms)
        if self.parity not in (0, None):
            raise ValueError("Planar-bond parity must be 0 or None.")
        if self.atoms[2] == self.atoms[3]:
            raise ValueError("Planar-bond central atoms must be distinct.")

    @property
    def bond(self) -> frozenset[Reference]:
        return frozenset(self.atoms[2:4])

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if isinstance(value, int))

    def canonical_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            left = tuple(sorted(self.atoms[:2], key=str))
            right = tuple(sorted(self.atoms[4:], key=str))
            ends = sorted(((self.atoms[2], left), (self.atoms[3], right)), key=str)
            return (self.descriptor_class, None, tuple(ends))
        forms = tuple(
            tuple(self.atoms[index] for index in permutation)
            for permutation in self._PERMUTATIONS
        )
        return (
            self.descriptor_class,
            0,
            min(forms, key=lambda values: tuple(map(str, values))),
        )

    def invert(self) -> "PlanarBondStereo":
        if self.parity is None:
            return self
        atoms = (self.atoms[1], self.atoms[0], *self.atoms[2:])
        return PlanarBondStereo(atoms, 0, self.provenance)

    def relabel(self, mapping: Mapping[int, int]) -> "PlanarBondStereo":
        return PlanarBondStereo(
            tuple(_relabel_reference(value, mapping) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PlanarBondStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class AtropBondStereo:
    """Relative axial orientation around an atropisomeric bond."""

    atoms: tuple[
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
        Reference,
    ]
    parity: int | None
    provenance: str | None = None

    descriptor_class = "atrop_bond"
    _INVERSION = (1, 0, 2, 3, 4, 5)
    _PERMUTATIONS = (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 1, 0),
        (5, 4, 3, 2, 0, 1),
    )

    def __post_init__(self) -> None:
        if len(self.atoms) != 6:
            raise ValueError("Atrop-bond stereo requires six references.")
        _validate_bond_references(self.atoms)
        if self.parity not in (-1, 1, None):
            raise ValueError("Atrop-bond parity must be -1, 1, or None.")
        if self.atoms[2] == self.atoms[3]:
            raise ValueError("Atrop-bond central atoms must be distinct.")

    @property
    def bond(self) -> frozenset[Reference]:
        return frozenset(self.atoms[2:4])

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(value for value in self.atoms if isinstance(value, int))

    def canonical_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            return _unknown_bond_form(self.descriptor_class, self.atoms)
        return _permuted_canonical_form(
            self.descriptor_class,
            self.atoms,
            self.parity,
            self._PERMUTATIONS,
            self._INVERSION,
        )

    def invert(self) -> "AtropBondStereo":
        return (
            self
            if self.parity is None
            else AtropBondStereo(self.atoms, -self.parity, self.provenance)
        )

    def relabel(self, mapping: Mapping[int, int]) -> "AtropBondStereo":
        return AtropBondStereo(
            tuple(_relabel_reference(value, mapping) for value in self.atoms),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AtropBondStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


StereoValue = (
    TetrahedralStereo
    | SquarePlanarStereo
    | TrigonalBipyramidalStereo
    | OctahedralStereo
    | PlanarBondStereo
    | AtropBondStereo
)


def stereo_from_dict(value: Mapping[str, Any]) -> StereoValue:
    descriptor_class = value["descriptor_class"]
    descriptor_types = {
        "tetrahedral": TetrahedralStereo,
        "square_planar": SquarePlanarStereo,
        "trigonal_bipyramidal": TrigonalBipyramidalStereo,
        "octahedral": OctahedralStereo,
        "planar_bond": PlanarBondStereo,
        "atrop_bond": AtropBondStereo,
    }
    descriptor_type = descriptor_types.get(descriptor_class)
    if descriptor_type is not None:
        return descriptor_type(  # type: ignore[call-arg,return-value]
            tuple(value["atoms"]),
            value.get("parity"),
            value.get("provenance"),
        )
    deferred = descriptor_class in DEFERRED_STEREO_DESCRIPTOR_CLASSES
    suffix = " (known but not implemented)" if deferred else ""
    raise ValueError(
        f"Unsupported stereo descriptor class: {descriptor_class!r}{suffix}"
    )


def descriptor_id(descriptor: StereoValue) -> str:
    if isinstance(
        descriptor,
        (
            TetrahedralStereo,
            SquarePlanarStereo,
            TrigonalBipyramidalStereo,
            OctahedralStereo,
        ),
    ):
        return f"atom:{descriptor.center}"
    center = sorted(descriptor.bond, key=str)
    return f"bond:{center[0]}-{center[1]}"
