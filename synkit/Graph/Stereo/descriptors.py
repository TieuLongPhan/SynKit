"""Relative stereodescriptors adapted for mapped SynKit Lewis-labelled graphs.

Stereo identity is relative and permutation-aware; CIP labels are
intentionally outside descriptor identity.  The non-tetrahedral permutation
groups are adapted from StereoMolGraph commit
``2189f610f23eaaf992e2e01a12ea4d0532496601`` (MIT, copyright (c) 2025
Maxim Papusha); the corresponding notice is shipped in
``synkit/Graph/Stereo/LICENSES/StereoMolGraph-MIT.txt``.

SynKit keeps its own descriptor values because they participate in executable
reaction rules and retain Lewis/electron state in the surrounding graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ._descriptor_core import (
    OrbitDescriptorMixin as _OrbitDescriptorMixin,
    descriptor_dict as _descriptor_dict,
    permutation_sign as _permutation_sign,
    permuted_canonical_form as _permuted_canonical_form,
    reference_sort_key as _reference_sort_key,
    unknown_atom_form as _unknown_atom_form,
    unknown_bond_form as _unknown_bond_form,
)
from .orbits import (
    StereoRelation,
    StereoRelationKind,
)
from .supports import (
    AtomCenteredStereo,
    AxisStereo,
    AxisStereoSupport,
    BondCenteredStereo,
    PathStereo,
    Reference,
)

VirtualReferenceKind = Literal["H", "LP"]

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


# ``SUPPORTED`` includes graph identity, rules, and JSON/GML; adapters may be narrower.
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
CONFIGURED_STEREO_DESCRIPTOR_CLASSES = SUPPORTED_STEREO_DESCRIPTOR_CLASSES | {
    "cumulene_axis",
    "extended_cis_trans",
    "helical",
    "planar_chirality",
    "framework",
}
RDKIT_STEREO_DESCRIPTOR_CLASSES = frozenset(
    {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "planar_bond",
        "atrop_bond",
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


@dataclass(frozen=True, eq=False)
class TetrahedralStereo(_OrbitDescriptorMixin, AtomCenteredStereo):
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
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class SquarePlanarStereo(_OrbitDescriptorMixin, AtomCenteredStereo):
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
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class TrigonalBipyramidalStereo(_OrbitDescriptorMixin, AtomCenteredStereo):
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
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class OctahedralStereo(_OrbitDescriptorMixin, AtomCenteredStereo):
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
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class PlanarBondStereo(_OrbitDescriptorMixin, BondCenteredStereo):
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
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


@dataclass(frozen=True, eq=False)
class AtropBondStereo(_OrbitDescriptorMixin, AxisStereo):
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
    def support(self) -> AxisStereoSupport:
        return AxisStereoSupport(
            (self.atoms[2], self.atoms[3]),  # type: ignore[arg-type]
            (self.atoms[:2], self.atoms[4:]),  # type: ignore[arg-type]
        )

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
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        return _descriptor_dict(self)


AtropAxisStereo = AtropBondStereo


@dataclass(frozen=True, eq=False)
class CumuleneAxisStereo(_OrbitDescriptorMixin, AxisStereo):
    """Configured axial stereo over an even-cumulene path.

    ``axis_path`` retains every cumulene atom.  ``terminal_frames`` contains
    two ordered two-reference frames owned by the first and last path atoms.
    Stability evidence and CIP labels are intentionally not descriptor fields.
    """

    axis_path: tuple[int, ...]
    terminal_frames: tuple[tuple[Reference, Reference], tuple[Reference, Reference]]
    parity: int | None
    provenance: str | None = None

    descriptor_class = "cumulene_axis"
    _INVERSION = (1, 0, 2, 3, 4, 5)
    _PERMUTATIONS = AtropBondStereo._PERMUTATIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_path", tuple(self.axis_path))
        frames = tuple(tuple(frame) for frame in self.terminal_frames)
        object.__setattr__(self, "terminal_frames", frames)
        if len(self.axis_path) < 3:
            raise ValueError("Cumulene-axis stereo requires at least three atoms.")
        if (len(self.axis_path) - 1) % 2:
            raise ValueError("Cumulene-axis stereo requires an even number of bonds.")
        if any(type(atom) is not int for atom in self.axis_path):
            raise TypeError("Cumulene-axis paths require integer atom IDs.")
        if len(set(self.axis_path)) != len(self.axis_path):
            raise ValueError("Cumulene-axis paths must not repeat atoms.")
        if len(self.terminal_frames) != 2 or any(
            len(frame) != 2 for frame in self.terminal_frames
        ):
            raise ValueError("Cumulene-axis stereo requires two two-reference frames.")
        owners = self.axis_path[0], self.axis_path[-1]
        for frame, owner in zip(self.terminal_frames, owners):
            for position, reference in enumerate(frame):
                _validate_reference(reference, owner=owner, position=position)
            if len(set(frame)) != 2:
                raise ValueError("Cumulene terminal references must be distinct.")
            on_axis = any(
                type(reference) is int and reference in self.axis_path
                for reference in frame
            )
            if on_axis:
                raise ValueError("Cumulene terminal references cannot lie on the axis.")
        if self.parity not in (-1, 1, None):
            raise ValueError("Cumulene-axis parity must be -1, 1, or None.")

    @property
    def support(self) -> AxisStereoSupport:
        return AxisStereoSupport(self.axis_path, self.terminal_frames)

    @property
    def atoms(self) -> tuple[Reference, Reference, int, int, Reference, Reference]:
        """Return the compatible six-position local orbit frame."""
        left, right = self.terminal_frames
        return (*left, self.axis_path[0], self.axis_path[-1], *right)

    @property
    def dependencies(self) -> frozenset[int]:
        return self.support.dependencies

    def _canonical_axis_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            left = tuple(sorted(self.terminal_frames[0], key=_reference_sort_key))
            right = tuple(sorted(self.terminal_frames[1], key=_reference_sort_key))
            reversed_path = tuple(reversed(self.axis_path))
            candidates = ((self.axis_path, left, right), (reversed_path, right, left))
            return self.descriptor_class, None, min(candidates, key=repr)

        atoms: tuple[Reference, ...] = self.atoms
        if self.parity == -1:
            atoms = tuple(atoms[index] for index in self._INVERSION)
        candidates = []
        for permutation in self._PERMUTATIONS:
            frame = tuple(atoms[index] for index in permutation)
            path = (
                tuple(reversed(self.axis_path))
                if permutation[2] == 3
                else self.axis_path
            )
            candidates.append((path, frame))
        return self.descriptor_class, 1, min(candidates, key=repr)

    def canonical_form(self) -> tuple[Any, ...]:
        return self._canonical_axis_form()

    def same_configuration(
        self,
        other: object,
        *,
        semantics: str = "orbit",
        diagnostics: list[Any] | None = None,
    ) -> bool:
        del semantics, diagnostics
        return (
            isinstance(other, CumuleneAxisStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def relation_to(self, other: object) -> StereoRelation:
        if not isinstance(other, CumuleneAxisStereo):
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        paths_match = self.axis_path in (
            other.axis_path,
            tuple(reversed(other.axis_path)),
        )
        if not paths_match:
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        return self.configuration.relation_to(other.configuration)

    def invert(self) -> "CumuleneAxisStereo":
        return (
            self
            if self.parity is None
            else CumuleneAxisStereo(
                self.axis_path,
                self.terminal_frames,
                -self.parity,
                self.provenance,
            )
        )

    def opposite(self) -> "CumuleneAxisStereo":
        return self.invert()

    def reversed(self) -> "CumuleneAxisStereo":
        """Reverse the axis using the preserving endpoint-frame convention."""
        left, right = self.terminal_frames
        return CumuleneAxisStereo(
            tuple(reversed(self.axis_path)),
            (right, tuple(reversed(left))),
            self.parity,
            self.provenance,
        )

    def relabel(self, mapping: Mapping[int, int]) -> "CumuleneAxisStereo":
        return CumuleneAxisStereo(
            tuple(mapping.get(atom, atom) for atom in self.axis_path),
            tuple(
                tuple(_relabel_reference(reference, mapping) for reference in frame)
                for frame in self.terminal_frames
            ),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def replace_references(
        self,
        replacements: Mapping[Reference, Reference],
    ) -> "CumuleneAxisStereo":
        known = set(self.axis_path) | {
            reference for frame in self.terminal_frames for reference in frame
        }
        unknown = set(replacements) - known
        if unknown:
            raise ValueError(
                f"Replacement sources are absent: {sorted(map(repr, unknown))}."
            )
        protected = set(self.axis_path) & set(replacements)
        if protected:
            raise ValueError("Reference replacement cannot replace descriptor loci.")
        frames = tuple(
            tuple(replacements.get(reference, reference) for reference in frame)
            for frame in self.terminal_frames
        )
        return CumuleneAxisStereo(
            self.axis_path,
            frames,  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "axis_path": list(self.axis_path),
            "terminal_frames": [list(frame) for frame in self.terminal_frames],
            "parity": self.parity,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, eq=False)
class ExtendedCisTransStereo(_OrbitDescriptorMixin, PathStereo):
    """Extended E/Z stereo over an odd-bond cumulene path.

    Unlike :class:`CumuleneAxisStereo`, this geometry is mirror-fixed.  The
    complete cumulene path is the carrier; no individual double bond owns the
    configuration.
    """

    path: tuple[int, ...]
    terminal_frames: tuple[tuple[Reference, Reference], tuple[Reference, Reference]]
    parity: int | None = 0
    provenance: str | None = None

    descriptor_class = "extended_cis_trans"
    _PERMUTATIONS = PlanarBondStereo._PERMUTATIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        frames = tuple(tuple(frame) for frame in self.terminal_frames)
        object.__setattr__(self, "terminal_frames", frames)
        if len(self.path) < 4:
            raise ValueError(
                "Extended cis/trans stereo requires at least four cumulene atoms."
            )
        if (len(self.path) - 1) % 2 != 1:
            raise ValueError(
                "Extended cis/trans stereo requires an odd number of bonds."
            )
        if any(type(atom) is not int for atom in self.path):
            raise TypeError("Extended cis/trans paths require integer atom IDs.")
        if len(set(self.path)) != len(self.path):
            raise ValueError("Extended cis/trans paths must not repeat atoms.")
        if len(self.terminal_frames) != 2 or any(
            len(frame) != 2 for frame in self.terminal_frames
        ):
            raise ValueError(
                "Extended cis/trans stereo requires two two-reference frames."
            )
        owners = self.path[0], self.path[-1]
        for frame, owner in zip(self.terminal_frames, owners):
            for position, reference in enumerate(frame):
                _validate_reference(reference, owner=owner, position=position)
            if len(set(frame)) != 2:
                raise ValueError(
                    "Extended cis/trans terminal references must be distinct."
                )
            if any(
                type(reference) is int and reference in self.path for reference in frame
            ):
                raise ValueError(
                    "Extended cis/trans terminal references cannot lie on the path."
                )
        if self.parity not in (0, None):
            raise ValueError("Extended cis/trans parity must be 0 or None.")

    @property
    def support(self) -> AxisStereoSupport:
        """Return the path plus its two terminal reference frames."""
        return AxisStereoSupport(self.path, self.terminal_frames)

    @property
    def atoms(self) -> tuple[Reference, Reference, int, int, Reference, Reference]:
        """Return the compatible six-position terminal-frame orbit."""
        left, right = self.terminal_frames
        return (*left, self.path[0], self.path[-1], *right)

    @property
    def dependencies(self) -> frozenset[int]:
        return self.support.dependencies

    def canonical_form(self) -> tuple[Any, ...]:
        if self.parity is None:
            left = tuple(sorted(self.terminal_frames[0], key=_reference_sort_key))
            right = tuple(sorted(self.terminal_frames[1], key=_reference_sort_key))
            candidates = (
                (self.path, left, right),
                (tuple(reversed(self.path)), right, left),
            )
            return self.descriptor_class, None, min(candidates, key=repr)

        candidates = []
        for permutation in self._PERMUTATIONS:
            frame = tuple(self.atoms[index] for index in permutation)
            path = tuple(reversed(self.path)) if permutation[2] == 3 else self.path
            candidates.append((path, frame))
        return self.descriptor_class, 0, min(candidates, key=repr)

    def same_configuration(
        self,
        other: object,
        *,
        semantics: str = "orbit",
        diagnostics: list[Any] | None = None,
    ) -> bool:
        del semantics, diagnostics
        return (
            isinstance(other, ExtendedCisTransStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def relation_to(self, other: object) -> StereoRelation:
        if not isinstance(other, ExtendedCisTransStereo):
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        paths_match = self.path in (
            other.path,
            tuple(reversed(other.path)),
        )
        if not paths_match:
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        return self.configuration.relation_to(other.configuration)

    def invert(self) -> "ExtendedCisTransStereo":
        if self.parity is None:
            return self
        left, right = self.terminal_frames
        return ExtendedCisTransStereo(
            self.path,
            ((left[1], left[0]), right),
            0,
            self.provenance,
        )

    def opposite(self) -> "ExtendedCisTransStereo":
        return self.invert()

    def reversed(self) -> "ExtendedCisTransStereo":
        left, right = self.terminal_frames
        return ExtendedCisTransStereo(
            tuple(reversed(self.path)),
            (right, left),
            self.parity,
            self.provenance,
        )

    def relabel(self, mapping: Mapping[int, int]) -> "ExtendedCisTransStereo":
        return ExtendedCisTransStereo(
            tuple(mapping.get(atom, atom) for atom in self.path),
            tuple(
                tuple(_relabel_reference(reference, mapping) for reference in frame)
                for frame in self.terminal_frames
            ),  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def replace_references(
        self,
        replacements: Mapping[Reference, Reference],
    ) -> "ExtendedCisTransStereo":
        known = set(self.path) | {
            reference for frame in self.terminal_frames for reference in frame
        }
        unknown = set(replacements) - known
        if unknown:
            raise ValueError(
                "Replacement sources are absent: " f"{sorted(map(repr, unknown))}."
            )
        if set(self.path) & set(replacements):
            raise ValueError("Reference replacement cannot replace descriptor loci.")
        frames = tuple(
            tuple(replacements.get(reference, reference) for reference in frame)
            for frame in self.terminal_frames
        )
        return ExtendedCisTransStereo(
            self.path,
            frames,  # type: ignore[arg-type]
            self.parity,
            self.provenance,
        )

    def __eq__(self, other: object) -> bool:
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "path": list(self.path),
            "terminal_frames": [list(frame) for frame in self.terminal_frames],
            "parity": self.parity,
            "provenance": self.provenance,
        }


from .extended_descriptors import HelicalStereo, PlanarChiralityStereo  # noqa: E402
from .global_stereo import FrameworkStereo  # noqa: E402

StereoValue = (
    TetrahedralStereo
    | SquarePlanarStereo
    | TrigonalBipyramidalStereo
    | OctahedralStereo
    | PlanarBondStereo
    | AtropBondStereo
    | CumuleneAxisStereo
    | ExtendedCisTransStereo
    | HelicalStereo
    | PlanarChiralityStereo
    | FrameworkStereo
)


def stereo_from_dict(value: Mapping[str, Any]) -> StereoValue:
    from ._descriptor_serialization import stereo_from_dict as restore

    return restore(value)


def descriptor_id(descriptor: StereoValue) -> str:
    from ._descriptor_serialization import descriptor_id as identify

    return identify(descriptor)
