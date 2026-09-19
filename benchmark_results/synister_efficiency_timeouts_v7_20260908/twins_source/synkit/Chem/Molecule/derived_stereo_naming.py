"""Derived local names bound to one exact configured-stereograph assignment.

Naming is a one-way reporting layer.  It consumes the exact assignment
certificate produced by :mod:`synkit.Graph.Stereo.enumeration`, validates that
the certificate and fixed descriptor population agree, and delegates local
CIP projection to :mod:`synkit.Chem.Molecule.cip_assignment`.  No name is
written back to a descriptor, molecule, canonical code, or identity key.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from rdkit import Chem

from synkit.Chem.Molecule.cip_assignment import (
    CIPAssignment,
    assign_cip_labels,
)
from synkit.Graph.Stereo import (
    STEREOGRAPH_SCHEMA,
    StereoAssignment,
    StereoSpecification,
)

DERIVED_STEREO_NAMING_SCHEMA = "synkit.derived-stereo-naming/1"


@dataclass(frozen=True)
class DerivedStereoNamingResult:
    """Immutable local-name report for one exact assignment certificate."""

    schema: str
    source_schema: str
    source_canonical_digest: str
    source_mirror_status: str
    registry_keys: tuple[str, ...]
    assignments: tuple[CIPAssignment, ...]

    @property
    def complete(self) -> bool:
        """Whether every configured locus received a derived local name."""
        return all(assignment.assigned for assignment in self.assignments)

    @property
    def labels(self) -> tuple[tuple[str, str], ...]:
        """Return assigned labels keyed by their source registry handles."""
        return tuple(
            (key, assignment.label)
            for key, assignment in zip(
                self.registry_keys,
                self.assignments,
            )
            if assignment.assigned and assignment.label is not None
        )

    @property
    def status_counts(self) -> tuple[tuple[str, int], ...]:
        """Return deterministic status counts for reporting."""
        counts = Counter(assignment.status.value for assignment in self.assignments)
        return tuple(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        """Return a report payload that keeps names outside identity."""
        return {
            "schema": self.schema,
            "source_schema": self.source_schema,
            "source_canonical_digest": self.source_canonical_digest,
            "source_mirror_status": self.source_mirror_status,
            "complete": self.complete,
            "labels": [list(value) for value in self.labels],
            "status_counts": dict(self.status_counts),
            "assignments": [
                {
                    "registry_key": key,
                    **assignment.to_dict(),
                }
                for key, assignment in zip(
                    self.registry_keys,
                    self.assignments,
                )
            ],
        }


def _validate_exact_assignment(assignment: StereoAssignment) -> str:
    if not isinstance(assignment, StereoAssignment):
        raise TypeError("Derived naming requires an exact StereoAssignment.")
    source_schema, separator, _body = assignment.canonical_code.partition("\n")
    if not separator or source_schema != STEREOGRAPH_SCHEMA:
        raise ValueError(
            "Derived naming requires a complete configured-stereograph certificate."
        )
    observed_digest = sha256(assignment.canonical_code.encode("utf-8")).hexdigest()
    if observed_digest != assignment.canonical_digest:
        raise ValueError("Stereo-assignment canonical digest is inconsistent.")
    if any(
        descriptor.specification is StereoSpecification.UNSPECIFIED
        for _key, descriptor in assignment.registry_items
    ):
        raise ValueError("Derived naming requires a complete fixed stereo assignment.")
    return source_schema


def derive_stereo_names(
    molecule: Chem.Mol,
    assignment: StereoAssignment,
    *,
    reference_to_index: Mapping[int, int] | None = None,
) -> DerivedStereoNamingResult:
    """Derive witnessed local labels from one exact stereo assignment."""
    if molecule is None:
        raise ValueError("Derived stereo naming requires a molecule.")
    source_schema = _validate_exact_assignment(assignment)
    keys = tuple(key for key, _descriptor in assignment.registry_items)
    descriptors = tuple(descriptor for _key, descriptor in assignment.registry_items)
    names = assign_cip_labels(
        molecule,
        descriptors,
        reference_to_index=reference_to_index,
    )
    return DerivedStereoNamingResult(
        DERIVED_STEREO_NAMING_SCHEMA,
        source_schema,
        assignment.canonical_digest,
        assignment.mirror_status.value,
        keys,
        names,
    )


def _rdkit_reference_to_index(molecule: Chem.Mol) -> dict[int, int]:
    maps = tuple(int(atom.GetAtomMapNum()) for atom in molecule.GetAtoms())
    fully_mapped = (
        bool(maps) and all(value > 0 for value in maps) and len(set(maps)) == len(maps)
    )
    return {
        (
            int(atom.GetAtomMapNum()) if fully_mapped else atom.GetIdx() + 1
        ): atom.GetIdx()
        for atom in molecule.GetAtoms()
    }


def derive_rdkit_stereo_names(
    molecule: Chem.Mol,
    assignment: StereoAssignment,
) -> DerivedStereoNamingResult:
    """Derive names using the namespace chosen by RDKit enumeration."""
    return derive_stereo_names(
        molecule,
        assignment,
        reference_to_index=_rdkit_reference_to_index(molecule),
    )


__all__ = [
    "DERIVED_STEREO_NAMING_SCHEMA",
    "DerivedStereoNamingResult",
    "derive_rdkit_stereo_names",
    "derive_stereo_names",
]
