"""Shared permutation-orbit mechanics for relative stereodescriptors."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Mapping, Sequence

from .orbits import (
    Permutation,
    StereoConfiguration,
    StereoRelation,
    StereoSpecification,
)
from .supports import Reference


def reference_sort_key(value: Reference) -> tuple[str, str]:
    """Return a total ordering key across material and virtual references."""
    return type(value).__name__, repr(value)


def _tuple_sort_key(
    values: Sequence[Reference],
) -> tuple[tuple[str, str], ...]:
    return tuple(reference_sort_key(value) for value in values)


def permuted_canonical_form(
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


def unknown_atom_form(
    descriptor_class: str,
    atoms: Sequence[Reference],
) -> tuple[Any, ...]:
    """Keep an unknown orientation attached to its atom locus and ligands."""
    return (
        descriptor_class,
        atoms[0],
        tuple(sorted(atoms[1:], key=reference_sort_key)),
        None,
    )


def unknown_bond_form(
    descriptor_class: str,
    atoms: Sequence[Reference],
) -> tuple[Any, ...]:
    """Keep an unknown orientation attached to its bond and endpoint groups."""
    left = (atoms[2], tuple(sorted(atoms[:2], key=reference_sort_key)))
    right = (atoms[3], tuple(sorted(atoms[4:], key=reference_sort_key)))
    return (
        descriptor_class,
        None,
        tuple(sorted((left, right), key=repr)),
    )


def descriptor_dict(descriptor: Any) -> dict[str, Any]:
    """Serialize the common descriptor wire fields."""
    return {
        "descriptor_class": descriptor.descriptor_class,
        "atoms": list(descriptor.atoms),
        "parity": descriptor.parity,
        "provenance": descriptor.provenance,
    }


def permutation_sign(
    values: Sequence[Reference],
    ordered: Sequence[Reference],
) -> int:
    """Return the parity of ``values`` relative to ``ordered``."""
    positions = {value: index for index, value in enumerate(ordered)}
    permutation = [positions[value] for value in values]
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


_NEGATIVE_ORBIT_IMAGES = {
    "tetrahedral": (0, 2, 1, 3, 4),
    "trigonal_bipyramidal": (0, 1, 2, 3, 5, 4),
    "octahedral": (0, 2, 1, 3, 4, 5, 6),
    "atrop_bond": (1, 0, 2, 3, 4, 5),
    "cumulene_axis": (1, 0, 2, 3, 4, 5),
}


class OrbitDescriptorMixin:
    """Adapt the stable descriptor wire model to finite-orbit semantics."""

    descriptor_class: str
    atoms: tuple[Reference, ...]
    parity: int | None
    provenance: str | None

    @property
    def specification(self) -> StereoSpecification:
        """Return whether orientation is fixed or intentionally unspecified."""
        return (
            StereoSpecification.UNSPECIFIED
            if self.parity is None
            else StereoSpecification.FIXED
        )

    @cached_property
    def configuration(self) -> StereoConfiguration:
        """Return the authoritative permutation-orbit configuration."""
        frame = tuple(self.atoms)
        if self.parity == -1:
            image = _NEGATIVE_ORBIT_IMAGES[self.descriptor_class]
            frame = Permutation(image).apply(frame)
        return StereoConfiguration(
            self.descriptor_class,
            frame,
            self.specification,
        )

    def same_configuration(
        self,
        other: object,
        *,
        semantics: str = "orbit",
        diagnostics: list[Any] | None = None,
    ) -> bool:
        """Test relative stereo identity, optionally auditing Beta-2."""
        from .legacy import (  # Independent oracle; kept lazy at this boundary.
            StereoSemanticComparison,
            StereoSemanticsMode,
            legacy_same_configuration,
        )

        mode = StereoSemanticsMode(semantics)
        if mode is StereoSemanticsMode.LEGACY:
            return legacy_same_configuration(self, other)
        orbit_result = (
            type(self) is type(other)
            and isinstance(other, OrbitDescriptorMixin)
            and self.configuration.same_configuration(other.configuration)
        )
        if mode is StereoSemanticsMode.ORBIT:
            return orbit_result
        legacy_result = legacy_same_configuration(self, other)
        if diagnostics is not None:
            diagnostics.append(
                StereoSemanticComparison.create(
                    "descriptor_identity",
                    orbit_result,
                    legacy_result,
                )
            )
        return orbit_result

    def relation_to(self, other: object) -> StereoRelation:
        """Classify and witness the geometric relation to another descriptor."""
        configuration = (
            other.configuration
            if type(self) is type(other) and isinstance(other, OrbitDescriptorMixin)
            else other
        )
        return self.configuration.relation_to(configuration)

    def replace_reference(self, old: Reference, new: Reference) -> Any:
        """Replace one peripheral reference without changing orientation."""
        return self.replace_references({old: new})

    def replace_references(
        self,
        replacements: Mapping[Reference, Reference],
    ) -> Any:
        """Replace peripheral references after formal locus validation."""
        self.configuration.replace_references(replacements)
        atoms = tuple(replacements.get(value, value) for value in self.atoms)
        return type(self)(atoms, self.parity, self.provenance)

    def opposite(self) -> Any:
        """Return the binary opposite when the geometry defines one."""
        opposite = self.configuration.opposite()
        if self.specification is StereoSpecification.UNSPECIFIED:
            return self
        parity = 0 if self.descriptor_class == "planar_bond" else 1
        return type(self)(opposite.frame, parity, self.provenance)
