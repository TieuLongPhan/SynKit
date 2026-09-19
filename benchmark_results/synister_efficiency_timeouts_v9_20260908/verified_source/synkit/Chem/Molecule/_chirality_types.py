"""Data models and topology-colour helpers for molecular chirality."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from synkit.Graph.Stereo.descriptors import Reference
from synkit.Graph.Stereo.supports import AxisStereoSupport


class MolecularChirality(str, Enum):
    """Global relationship between a molecule and its mirror image."""

    ACHIRAL = "Achiral"
    CHIRAL = "Chiral"


class MolecularChiralityOutcome(str, Enum):
    """Configuration-aware conclusion for possibly underspecified input."""

    NECESSARILY_ACHIRAL = "necessarily_achiral"
    NECESSARILY_CHIRAL = "necessarily_chiral"
    CONFIGURATION_DEPENDENT = "configuration_dependent"
    UNSUPPORTED_OR_INCOMPLETE = "unsupported_or_incomplete"


class PotentialStereoLocusType(str, Enum):
    """Topology-supported stereo locus whose configuration is not supplied."""

    CUMULENE_AXIS = "cumulene_axis"
    ATROP_AXIS = "atrop_axis"


class StereoOrientationState(str, Enum):
    """Information state of a potential stereo locus."""

    UNSPECIFIED = "unspecified"


class StereoStabilityStatus(str, Enum):
    """Configurational-stability evidence attached to a potential locus."""

    UNASSESSED = "unassessed"


class UnspecifiedMolecularStereoError(ValueError):
    """Raised when strict binary classification receives unresolved stereo."""

    def __init__(self, loci: tuple[str, ...]) -> None:
        self.loci = loci
        joined = ", ".join(loci)
        super().__init__(f"Molecular stereochemistry is underspecified at: {joined}")


@dataclass(frozen=True)
class MolecularChiralityResult:
    """Evidence returned by whole-molecule mirror classification."""

    classification: MolecularChirality
    mirror_isomorphism: tuple[tuple[int, int], ...] | None
    descriptor_count: int
    completed_tetrahedral_centers: tuple[int, ...]
    # Compatibility fields retained for readers of the exploratory report.
    # Sound classification never populates them from 2D connectivity alone.
    completed_extended_tetrahedral_axes: tuple[tuple[int, int], ...] = ()
    completed_biaryl_atrop_axes: tuple[tuple[int, int], ...] = ()
    identity_profile: str = "element-isotope-hydrogen-connectivity"
    decision_method: str = "exact_mirror_isomorphism"
    input_stereo_status: str = "specified"
    unspecified_stereo_loci: tuple[str, ...] = ()
    potential_stereo_loci: tuple["PotentialStereoLocus", ...] = ()
    configured_extended_descriptor_count: int = 0
    stereo_evidence_source: str | None = None
    extended_stability_status: str | None = None
    population_fraction: float | None = None

    @property
    def is_chiral(self) -> bool:
        """Return ``True`` when no orientation-preserving mirror map exists."""
        return self.classification is MolecularChirality.CHIRAL


@dataclass(frozen=True, init=False)
class PotentialStereoLocus:
    """A typed candidate locus without an invented configuration.

    Atom indices and material terminal references are zero-based RDKit atom
    indices. Virtual hydrogen references use ``@H:<owner-index>``. Detection
    from 2D connectivity proves only that the topology can support the locus;
    it supplies neither handedness nor configurational-stability evidence.
    """

    locus_type: PotentialStereoLocusType
    support: AxisStereoSupport
    orientation_state: StereoOrientationState = StereoOrientationState.UNSPECIFIED
    evidence_provenance: str = "two_dimensional_connectivity"
    stability_status: StereoStabilityStatus = StereoStabilityStatus.UNASSESSED

    def __init__(
        self,
        locus_type: PotentialStereoLocusType,
        atom_indices: tuple[int, ...] | None = None,
        terminal_references: tuple[tuple[Reference, ...], ...] | None = None,
        orientation_state: StereoOrientationState = StereoOrientationState.UNSPECIFIED,
        evidence_provenance: str = "two_dimensional_connectivity",
        stability_status: StereoStabilityStatus = StereoStabilityStatus.UNASSESSED,
        *,
        support: AxisStereoSupport | None = None,
    ) -> None:
        """Build from typed support or the compatible legacy field pair."""
        if support is None:
            if atom_indices is None or terminal_references is None:
                raise TypeError(
                    "Potential stereo loci require axis support or both legacy "
                    "atom_indices and terminal_references."
                )
            support = AxisStereoSupport(
                tuple(atom_indices),
                tuple(tuple(frame) for frame in terminal_references),  # type: ignore[arg-type]
            )
        elif atom_indices is not None or terminal_references is not None:
            raise TypeError("Supply typed support or legacy support fields, not both.")
        object.__setattr__(self, "locus_type", PotentialStereoLocusType(locus_type))
        object.__setattr__(self, "support", support)
        object.__setattr__(
            self, "orientation_state", StereoOrientationState(orientation_state)
        )
        object.__setattr__(self, "evidence_provenance", evidence_provenance)
        object.__setattr__(
            self, "stability_status", StereoStabilityStatus(stability_status)
        )

    @property
    def atom_indices(self) -> tuple[int, ...]:
        """Return the compatible axis-path view."""
        return self.support.path

    @property
    def terminal_references(self) -> tuple[tuple[Reference, Reference], ...]:
        """Return the compatible terminal-frame view."""
        return self.support.terminal_frames

    @property
    def identifier(self) -> str:
        """Return a deterministic diagnostic identifier."""
        support = "-".join(str(index) for index in self.atom_indices)
        return f"{self.locus_type.value}:{support}"


@dataclass(frozen=True)
class MolecularChiralityAssessment:
    """Configuration-aware result over every enumerated stereo completion."""

    outcome: MolecularChiralityOutcome
    observed_classifications: tuple[MolecularChirality, ...]
    input_stereo_status: str
    unspecified_stereo_loci: tuple[str, ...]
    unsupported_stereo_loci: tuple[str, ...]
    theoretical_isomer_upper_bound: int
    evaluated_isomer_count: int
    enumeration_complete: bool
    max_isomers: int
    representative_isomers: tuple[tuple[str, MolecularChirality], ...] = ()
    configured_alternative_count: int = 0
    configured_population_status: str | None = None
    decision_method: str = "stereo_completion_enumeration"

    @property
    def is_definitive(self) -> bool:
        """Return whether the outcome is proven despite possible truncation."""
        return self.outcome is not MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE


def molecular_node_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Match StereoMolGraph's explicit-H topology without expanding H atoms."""
    return (
        left.get("element") == right.get("element")
        and int(left.get("isotope", 0)) == int(right.get("isotope", 0))
        and int(left.get("hcount", 0)) == int(right.get("hcount", 0))
        and left.get("_molecular_colour") == right.get("_molecular_colour")
    )


def connectivity_edge_match(
    _left: Mapping[str, Any],
    _right: Mapping[str, Any],
) -> bool:
    """Use connectivity, not one selected Lewis/resonance bond assignment."""
    return True


def _intern_colours(signatures: Mapping[int, Any]) -> dict[int, int]:
    """Intern comparable signatures into deterministic compact integers."""
    palette = {
        signature: index
        for index, signature in enumerate(sorted(set(signatures.values()), key=repr))
    }
    return {node: palette[signature] for node, signature in signatures.items()}


def molecular_node_colours(graph: Any) -> dict[int, int]:
    """Return map-independent 1-WL molecular identity colours."""
    colours = _intern_colours(
        {
            node: (
                attributes.get("element"),
                int(attributes.get("isotope", 0)),
                int(attributes.get("hcount", 0)),
            )
            for node, attributes in graph.nodes(data=True)
        }
    )
    for _iteration in range(max(1, len(graph))):
        refined = _intern_colours(
            {
                node: (
                    colours[node],
                    tuple(sorted(colours[neighbor] for neighbor in graph[node])),
                )
                for node in graph
            }
        )
        if refined == colours:
            break
        colours = refined
    return colours
