"""Authorized molecular evidence for configured extended stereochemistry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Any, Mapping

from synkit.Graph.Stereo import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    stereo_from_dict,
)

ExtendedStereoDescriptor = (
    CumuleneAxisStereo | ExtendedCisTransStereo | AtropBondStereo | HelicalStereo
)
_EXTENDED_TYPES = (
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    AtropBondStereo,
    HelicalStereo,
)


class StereoEvidenceSource(str, Enum):
    """Authorized origins of a fixed extended configuration."""

    EXPLICIT_STEREO = "explicit_stereo"
    VALIDATED_GEOMETRY = "validated_geometry"
    DECLARED_SIDECAR = "declared_sidecar"


class ExtendedStereoStability(str, Enum):
    """Configurational-stability evidence, separate from orientation."""

    UNASSESSED = "unassessed"
    VALIDATED = "validated"


class StereoPopulationStatus(str, Enum):
    """Population claim attached to a complete configuration set."""

    PURE = "pure"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MolecularStereoConfiguration:
    """One complete set of fixed extended descriptors for a molecule.

    Descriptor references are zero-based RDKit atom indices.  Conversion to
    the one-based graph identity used by molecular mirror matching happens at
    the classifier boundary.
    """

    descriptors: tuple[ExtendedStereoDescriptor, ...]
    evidence_source: StereoEvidenceSource
    stability: ExtendedStereoStability = ExtendedStereoStability.UNASSESSED
    population_fraction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(
            self, "evidence_source", StereoEvidenceSource(self.evidence_source)
        )
        object.__setattr__(self, "stability", ExtendedStereoStability(self.stability))
        if not self.descriptors:
            raise ValueError("Molecular stereo configurations cannot be empty.")
        if not all(isinstance(value, _EXTENDED_TYPES) for value in self.descriptors):
            raise TypeError("Only configured axis or helical descriptors are accepted.")
        if any(descriptor.parity is None for descriptor in self.descriptors):
            raise ValueError(
                "Molecular configuration evidence requires fixed orientation."
            )
        if self.population_fraction is not None and not (
            0.0 < float(self.population_fraction) <= 1.0
        ):
            raise ValueError("Population fractions must lie in (0, 1].")

    def graph_descriptors(self) -> tuple[ExtendedStereoDescriptor, ...]:
        """Project zero-based RDKit references to one-based graph identities."""
        mapping = {
            atom: atom + 1
            for descriptor in self.descriptors
            for atom in descriptor.dependencies
        }
        return tuple(descriptor.relabel(mapping) for descriptor in self.descriptors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptors": [descriptor.to_dict() for descriptor in self.descriptors],
            "evidence_source": self.evidence_source.value,
            "stability": self.stability.value,
            "population_fraction": self.population_fraction,
            "reference_space": "rdkit_index_zero_based",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MolecularStereoConfiguration":
        if value.get("reference_space") != "rdkit_index_zero_based":
            raise ValueError(
                "Molecular stereo evidence requires zero-based references."
            )
        descriptors = tuple(
            stereo_from_dict(descriptor) for descriptor in value["descriptors"]
        )
        return cls(
            descriptors,  # type: ignore[arg-type]
            StereoEvidenceSource(value["evidence_source"]),
            ExtendedStereoStability(value.get("stability", "unassessed")),
            value.get("population_fraction"),
        )


@dataclass(frozen=True)
class MolecularStereoConfigurationSet:
    """Finite complete alternatives with an explicit population claim."""

    configurations: tuple[MolecularStereoConfiguration, ...]
    population_status: StereoPopulationStatus = StereoPopulationStatus.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "configurations", tuple(self.configurations))
        object.__setattr__(
            self, "population_status", StereoPopulationStatus(self.population_status)
        )
        if not self.configurations:
            raise ValueError("Molecular configuration sets cannot be empty.")
        if (
            self.population_status is StereoPopulationStatus.PURE
            and len(self.configurations) != 1
        ):
            raise ValueError("Pure populations require exactly one configuration.")
        if (
            self.population_status is StereoPopulationStatus.MIXED
            and len(self.configurations) < 2
        ):
            raise ValueError("Mixed populations require at least two configurations.")
        fractions = [item.population_fraction for item in self.configurations]
        if any(value is not None for value in fractions):
            if any(value is None for value in fractions) or not isclose(
                sum(float(value) for value in fractions if value is not None),
                1.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Declared population fractions must be complete and sum to one."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "configurations": [item.to_dict() for item in self.configurations],
            "population_status": self.population_status.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MolecularStereoConfigurationSet":
        return cls(
            tuple(
                MolecularStereoConfiguration.from_dict(item)
                for item in value["configurations"]
            ),
            StereoPopulationStatus(value.get("population_status", "unknown")),
        )


__all__ = [
    "ExtendedStereoDescriptor",
    "ExtendedStereoStability",
    "MolecularStereoConfiguration",
    "MolecularStereoConfigurationSet",
    "StereoEvidenceSource",
    "StereoPopulationStatus",
]
