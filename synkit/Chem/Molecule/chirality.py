"""Whole-molecule chirality classification by mirror automorphism.

This module is intentionally independent of reaction rules and reaction
stereo transport.  It asks one molecular question: is a molecule isomorphic
to its mirror image?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Mapping

import networkx as nx
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

from synkit.Graph.Stereo.descriptors import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    Reference,
    TetrahedralStereo,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)
from synkit.Graph.Stereo.extended_descriptors import HelicalStereo
from synkit.Graph.Stereo.identity import descriptor_relative_form
from synkit.Graph.Stereo.supports import AxisStereoSupport
from synkit.IO.mol_to_graph import MolToGraph

from ._chirality_loci import detect_potential_axis_supports
from .stereo_evidence import (
    ExtendedStereoStability,
    MolecularStereoConfiguration,
    MolecularStereoConfigurationSet,
    StereoEvidenceSource,
    StereoPopulationStatus,
)

__all__ = [
    "MolecularChirality",
    "MolecularChiralityAssessment",
    "MolecularChiralityOutcome",
    "MolecularChiralityResult",
    "MolecularStereoConfiguration",
    "MolecularStereoConfigurationSet",
    "ExtendedStereoStability",
    "PotentialStereoLocus",
    "PotentialStereoLocusType",
    "StereoOrientationState",
    "StereoEvidenceSource",
    "StereoPopulationStatus",
    "StereoStabilityStatus",
    "UnspecifiedMolecularStereoError",
    "assess_molecular_chirality",
    "classify_molecular_chirality",
    "clear_molecular_chirality_cache",
    "detect_potential_stereo_loci",
    "is_molecular_chiral",
]


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


def _molecular_node_match(
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


def _connectivity_edge_match(
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


def _molecular_node_colours(graph: Any) -> dict[int, int]:
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


def _resolved_cumulene_form(
    descriptor: CumuleneAxisStereo,
    resolve: Any,
) -> tuple[Any, ...]:
    """Resolve a complete cumulene path without entering rule semantics."""
    path = tuple(resolve(atom) for atom in descriptor.axis_path)
    frames = tuple(
        tuple(resolve(reference) for reference in frame)
        for frame in descriptor.terminal_frames
    )
    if descriptor.parity is None:
        left, right = (tuple(sorted(frame, key=repr)) for frame in frames)
        candidates = ((path, left, right), (tuple(reversed(path)), right, left))
        return descriptor.descriptor_class, None, min(candidates, key=repr)
    atoms = (*frames[0], path[0], path[-1], *frames[1])
    if descriptor.parity == -1:
        atoms = tuple(atoms[index] for index in descriptor._INVERSION)
    candidates = []
    for permutation in descriptor._PERMUTATIONS:
        frame = tuple(atoms[index] for index in permutation)
        oriented_path = tuple(reversed(path)) if permutation[2] == 3 else path
        candidates.append((oriented_path, frame))
    return descriptor.descriptor_class, 1, min(candidates, key=repr)


def _resolved_extended_cis_trans_form(
    descriptor: ExtendedCisTransStereo,
    resolve: Any,
) -> tuple[Any, ...]:
    """Resolve a complete odd-bond cumulene path and terminal frames."""
    path = tuple(resolve(atom) for atom in descriptor.path)
    frames = tuple(
        tuple(resolve(reference) for reference in frame)
        for frame in descriptor.terminal_frames
    )
    if descriptor.parity is None:
        left, right = (tuple(sorted(frame, key=repr)) for frame in frames)
        candidates = ((path, left, right), (tuple(reversed(path)), right, left))
        return descriptor.descriptor_class, None, min(candidates, key=repr)
    atoms = (*frames[0], path[0], path[-1], *frames[1])
    candidates = []
    for permutation in descriptor._PERMUTATIONS:
        frame = tuple(atoms[index] for index in permutation)
        oriented_path = tuple(reversed(path)) if permutation[2] == 3 else path
        candidates.append((oriented_path, frame))
    return descriptor.descriptor_class, 0, min(candidates, key=repr)


def _resolved_helical_form(
    descriptor: HelicalStereo,
    resolve: Any,
) -> tuple[Any, ...]:
    """Resolve open/cyclic path identity for molecular mirror matching."""
    path = tuple(resolve(atom) for atom in descriptor.path)
    if descriptor.cyclic:
        reverse = tuple(reversed(path))
        variants = tuple(
            sequence[offset:] + sequence[:offset]
            for sequence in (path, reverse)
            for offset in range(len(path))
        )
    else:
        variants = path, tuple(reversed(path))
    return (
        descriptor.descriptor_class,
        min(variants, key=repr),
        descriptor.cyclic,
        descriptor.parity,
        descriptor.coupling_id,
    )


def _molecular_stereo_form(graph: Any) -> tuple[Any, ...]:
    """Return a safe colour-refinement prefilter for mirror isomorphism."""
    registry = graph.graph.get("stereo_descriptors", {})
    if not registry:
        return ()
    colours = _molecular_node_colours(graph)

    def resolve(reference: int | str) -> tuple[str, Any]:
        if type(reference) is int:
            if reference not in colours:
                raise ValueError(f"Stereo reference {reference} is absent.")
            return "atom", colours[reference]
        virtual = parse_virtual_reference(reference)
        if virtual is None or virtual.center not in colours:
            raise ValueError(f"Invalid virtual stereo reference: {reference!r}.")
        return "virtual", (virtual.kind, colours[virtual.center])

    def descriptor_form(descriptor: Any) -> tuple[Any, ...]:
        if isinstance(descriptor, CumuleneAxisStereo):
            return _resolved_cumulene_form(descriptor, resolve)
        if isinstance(descriptor, ExtendedCisTransStereo):
            return _resolved_extended_cis_trans_form(descriptor, resolve)
        if isinstance(descriptor, HelicalStereo):
            return _resolved_helical_form(descriptor, resolve)
        return descriptor_relative_form(descriptor, resolve)

    return tuple(
        sorted(
            (descriptor_form(descriptor) for descriptor in registry.values()),
            key=repr,
        )
    )


def _resolved_registry_matches(
    source: tuple[Any, ...],
    target: frozenset[Any],
    mapping: Mapping[int, int],
) -> bool:
    """Reject any fully transported descriptor absent from the target."""
    mapped_nodes = mapping.keys()
    for descriptor in source:
        if not descriptor.dependencies.issubset(mapped_nodes):
            continue
        if descriptor.relabel(mapping) not in target:
            return False
    return True


class _MolecularStereoGraphMatcher(nx.isomorphism.GraphMatcher):
    """VF2 matcher that checks complete local frames during expansion."""

    def __init__(self, left: Any, right: Any) -> None:
        nx.set_node_attributes(
            left,
            _molecular_node_colours(left),
            "_molecular_colour",
        )
        nx.set_node_attributes(
            right,
            _molecular_node_colours(right),
            "_molecular_colour",
        )
        super().__init__(
            left,
            right,
            node_match=_molecular_node_match,
            edge_match=_connectivity_edge_match,
        )
        self._left_registry = tuple(left.graph.get("stereo_descriptors", {}).values())
        self._right_registry = tuple(right.graph.get("stereo_descriptors", {}).values())
        self._left_configurations = frozenset(self._left_registry)
        self._right_configurations = frozenset(self._right_registry)

    def semantic_feasibility(self, left_node: int, right_node: int) -> bool:
        if not super().semantic_feasibility(left_node, right_node):
            return False
        forward = dict(self.core_1)
        reverse = dict(self.core_2)
        forward[left_node] = right_node
        reverse[right_node] = left_node
        return _resolved_registry_matches(
            self._left_registry,
            self._right_configurations,
            forward,
        ) and _resolved_registry_matches(
            self._right_registry,
            self._left_configurations,
            reverse,
        )


def _indexed_copy(molecule: Chem.Mol) -> Chem.Mol:
    if molecule is None:
        raise ValueError("Molecular chirality classification requires a molecule.")
    working = Chem.Mol(molecule)
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    Chem.AssignStereochemistry(working, cleanIt=False, force=True)
    return working


def _configuration_covers_rdkit_locus(
    molecule: Chem.Mol,
    info: Any,
    configuration: MolecularStereoConfiguration,
) -> bool:
    if info.type != Chem.StereoType.Bond_Double:
        return False
    bond = molecule.GetBondWithIdx(int(info.centeredOn))
    locus = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return any(
        isinstance(
            descriptor,
            (CumuleneAxisStereo, ExtendedCisTransStereo),
        )
        and locus
        in {
            frozenset(pair)
            for pair in zip(
                (
                    descriptor.axis_path
                    if isinstance(descriptor, CumuleneAxisStereo)
                    else descriptor.path
                ),
                (
                    descriptor.axis_path[1:]
                    if isinstance(descriptor, CumuleneAxisStereo)
                    else descriptor.path[1:]
                ),
            )
        }
        for descriptor in configuration.descriptors
    )


def _unspecified_stereo_loci(
    molecule: Chem.Mol,
    configurations: tuple[MolecularStereoConfiguration, ...] = (),
) -> tuple[str, ...]:
    def unresolved(info: Any) -> bool:
        if info.specified == Chem.StereoSpecified.Specified:
            return False
        return not configurations or not all(
            _configuration_covers_rdkit_locus(molecule, info, configuration)
            for configuration in configurations
        )

    return tuple(
        sorted(
            f"{info.type}:{int(info.centeredOn)}"
            for info in Chem.FindPotentialStereo(molecule)
            if unresolved(info)
        )
    )


def _unsupported_stereo_loci(molecule: Chem.Mol) -> tuple[str, ...]:
    supported = {
        Chem.StereoType.Atom_Tetrahedral,
        Chem.StereoType.Bond_Double,
    }
    return tuple(
        sorted(
            f"{info.type}:{int(info.centeredOn)}"
            for info in Chem.FindPotentialStereo(molecule)
            if info.specified != Chem.StereoSpecified.Specified
            and info.type not in supported
        )
    )


def _complete_tetrahedral_topology(
    molecule: Chem.Mol,
    registry: dict[str, Any],
) -> tuple[int, ...]:
    """Add fixed probes at eligible unrepresented sp3 topologies.

    These probes are not claims that each atom is a stereocentre. Their
    arbitrary common orientation lets the whole-molecule automorphism test
    decide which local probes cancel by symmetry and which collectively make
    the molecular topology chiral. This is the role of ``stereo_complete`` in
    the published StereoMolGraph validation protocol.
    """
    represented = {
        descriptor.atoms[0]
        for descriptor in registry.values()
        if descriptor.descriptor_class == "tetrahedral"
    }
    completed = []
    for atom in molecule.GetAtoms():
        center = atom.GetIdx() + 1
        if center in represented:
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        references: list[int | str] = [
            neighbor.GetIdx() + 1 for neighbor in atom.GetNeighbors()
        ]
        hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
        if len(references) == 3 and hidden_hydrogens == 1:
            references.append(virtual_reference("H", center))
        elif len(references) != 4:
            continue
        descriptor = TetrahedralStereo(
            (center, *references),
            1,
            "molecular_chirality:stereo_complete",
        )
        registry[descriptor_id(descriptor)] = descriptor
        completed.append(center)
    return tuple(completed)


def detect_potential_stereo_loci(
    molecule: Chem.Mol,
    *,
    include_extended_ring_axes: bool = False,
) -> tuple[PotentialStereoLocus, ...]:
    """Return topology-supported, orientation-unspecified axial loci.

    The detector intentionally over-approximates. A returned locus is not a
    configured descriptor, a rotational-stability claim, or proof that the
    complete molecule is chiral.
    """
    loci = tuple(
        PotentialStereoLocus(
            locus_type=PotentialStereoLocusType(locus_type),
            support=support,
        )
        for locus_type, support in detect_potential_axis_supports(
            molecule,
            include_extended_ring_axes=include_extended_ring_axes,
        )
    )
    return tuple(sorted(loci, key=lambda locus: locus.identifier))


def _validate_material_frame(
    molecule: Chem.Mol,
    owner: int,
    references: tuple[Reference, ...],
) -> None:
    """Validate one zero-based molecular sidecar frame."""
    owner_atom = molecule.GetAtomWithIdx(owner)
    neighbors = {neighbor.GetIdx() for neighbor in owner_atom.GetNeighbors()}
    virtual_counts = {"H": 0, "LP": 0}
    for reference in references:
        if type(reference) is int:
            if reference < 0 or reference >= molecule.GetNumAtoms():
                raise ValueError(f"Extended stereo reference {reference} is absent.")
            if reference not in neighbors:
                raise ValueError(
                    f"Extended stereo reference {reference} is not adjacent to {owner}."
                )
            continue
        virtual = parse_virtual_reference(reference)
        if virtual is None or virtual.center != owner:
            raise ValueError(f"Invalid extended virtual reference: {reference!r}.")
        virtual_counts[virtual.kind] += 1
    available_h = int(owner_atom.GetNumExplicitHs()) + int(
        owner_atom.GetNumImplicitHs()
    )
    if virtual_counts["H"] > available_h:
        raise ValueError(f"Extended stereo requires unavailable hydrogen at {owner}.")


def _validate_molecular_configuration(
    molecule: Chem.Mol,
    configuration: MolecularStereoConfiguration,
) -> None:
    """Prove that declared zero-based descriptor supports exist in the molecule."""
    atom_count = molecule.GetNumAtoms()
    for descriptor in configuration.descriptors:
        material = descriptor.dependencies
        if any(atom < 0 or atom >= atom_count for atom in material):
            raise ValueError("Extended stereo support contains an absent atom index.")
        if isinstance(
            descriptor,
            (CumuleneAxisStereo, ExtendedCisTransStereo),
        ):
            path = (
                descriptor.axis_path
                if isinstance(descriptor, CumuleneAxisStereo)
                else descriptor.path
            )
            for left, right in zip(path, path[1:]):
                bond = molecule.GetBondBetweenAtoms(left, right)
                if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                    raise ValueError(
                        "Cumulene evidence requires a continuous double-bond path."
                    )
            for owner, frame in zip(
                (path[0], path[-1]),
                descriptor.terminal_frames,
            ):
                _validate_material_frame(molecule, owner, frame)
        elif isinstance(descriptor, AtropBondStereo):
            left, right = descriptor.atoms[2:4]
            if molecule.GetBondBetweenAtoms(left, right) is None:
                raise ValueError("Atrop evidence requires its central bond.")
            _validate_material_frame(molecule, left, descriptor.atoms[:2])
            _validate_material_frame(molecule, right, descriptor.atoms[4:])
        elif isinstance(descriptor, HelicalStereo):
            pairs = list(zip(descriptor.path, descriptor.path[1:]))
            if descriptor.cyclic:
                pairs.append((descriptor.path[-1], descriptor.path[0]))
            if any(
                molecule.GetBondBetweenAtoms(left, right) is None
                for left, right in pairs
            ):
                raise ValueError(
                    "Helical evidence requires a continuous molecular path."
                )


def _configuration_covers_locus(
    configuration: MolecularStereoConfiguration,
    locus: PotentialStereoLocus,
) -> bool:
    target = min(locus.support.path, tuple(reversed(locus.support.path)))
    for descriptor in configuration.descriptors:
        if isinstance(descriptor, (CumuleneAxisStereo, AtropBondStereo)):
            path = descriptor.support.path
            if min(path, tuple(reversed(path))) == target:
                return True
    return False


def _inject_molecular_configuration(
    registry: dict[str, Any],
    configuration: MolecularStereoConfiguration,
) -> None:
    for descriptor in configuration.graph_descriptors():
        key = descriptor_id(descriptor)
        previous = registry.get(key)
        if previous is not None and previous != descriptor:
            raise ValueError(f"Conflicting configured stereo evidence at {key}.")
        registry[key] = descriptor


def _configuration_result_metadata(
    configuration: MolecularStereoConfiguration | None,
) -> dict[str, Any]:
    if configuration is None:
        return {}
    return {
        "configured_extended_descriptor_count": len(configuration.descriptors),
        "stereo_evidence_source": configuration.evidence_source.value,
        "extended_stability_status": configuration.stability.value,
        "population_fraction": configuration.population_fraction,
    }


def classify_molecular_chirality(
    molecule: Chem.Mol,
    *,
    stereo_complete: bool = True,
    require_specified: bool = False,
    stereo_configuration: MolecularStereoConfiguration | None = None,
) -> MolecularChiralityResult:
    """Classify a molecule as globally chiral or achiral.

    The molecular identity profile uses element, isotope, total hydrogen
    count, and connectivity. It otherwise follows the published
    StereoMolGraph validation protocol and deliberately avoids raw charge and
    bond-order fields from a single Lewis/resonance form. Assigned local
    descriptors are retained. When ``stereo_complete`` is true, eligible
    unrepresented sp3 topologies receive the published protocol's completion
    probes before all parity-bearing configurations are reflected. Cumulene
    and biaryl topology is reported separately as orientation-unspecified
    potential loci and never injected into the descriptor registry.

    This is a molecule classifier. It does not extract, apply, or compare
    reaction rules. The identity profile is a graph-topology convention, not a
    quantum/geometric chirality proof: bond-order distinctions that do not
    change hydrogen topology can be collapsed, as in validation case VS170.
    ``stereo_complete`` cannot recover a stereochemical configuration erased
    from an input SMILES. In particular, a topological biaryl candidate proves
    neither orientation nor a high rotational barrier, and ordinary SMILES
    cannot distinguish helicene handedness. Set ``require_specified`` to reject
    RDKit-recognized unresolved input and detected unconfigured axial loci.
    ``stereo_configuration`` may add fixed extended descriptors only through
    the authorized evidence model; its references are zero-based RDKit indices.
    """
    if stereo_configuration is not None:
        _validate_molecular_configuration(molecule, stereo_configuration)
    working = _indexed_copy(molecule)
    unspecified = _unspecified_stereo_loci(
        working,
        () if stereo_configuration is None else (stereo_configuration,),
    )
    potential_loci = detect_potential_stereo_loci(working)
    potential_identifiers = tuple(
        locus.identifier
        for locus in potential_loci
        if stereo_configuration is None
        or not _configuration_covers_locus(stereo_configuration, locus)
    )
    unresolved = tuple(sorted((*unspecified, *potential_identifiers)))
    if require_specified and unresolved:
        raise UnspecifiedMolecularStereoError(unresolved)
    graph = MolToGraph(attr_profile="minimal").transform(
        working,
        use_index_as_atom_map=True,
    )
    registry = dict(graph.graph.get("stereo_descriptors", {}))
    completed = (
        _complete_tetrahedral_topology(working, registry) if stereo_complete else ()
    )
    if stereo_configuration is not None:
        _inject_molecular_configuration(registry, stereo_configuration)
    graph.graph["stereo_descriptors"] = registry

    mirror = graph.copy()
    mirror.graph["stereo_descriptors"] = {
        key: descriptor.invert() if descriptor.parity in {-1, 1} else descriptor
        for key, descriptor in registry.items()
    }
    if _molecular_stereo_form(graph) != _molecular_stereo_form(mirror):
        return MolecularChiralityResult(
            classification=MolecularChirality.CHIRAL,
            mirror_isomorphism=None,
            descriptor_count=len(registry),
            completed_tetrahedral_centers=completed,
            potential_stereo_loci=potential_loci,
            decision_method="stereo_colour_prefilter",
            input_stereo_status=("underspecified" if unresolved else "specified"),
            unspecified_stereo_loci=unresolved,
            **_configuration_result_metadata(stereo_configuration),
        )
    matcher = _MolecularStereoGraphMatcher(graph, mirror)
    mapping = dict(matcher.mapping) if matcher.is_isomorphic() else None
    classification = (
        MolecularChirality.ACHIRAL if mapping is not None else MolecularChirality.CHIRAL
    )
    return MolecularChiralityResult(
        classification=classification,
        mirror_isomorphism=(
            None if mapping is None else tuple(sorted(mapping.items()))
        ),
        descriptor_count=len(registry),
        completed_tetrahedral_centers=completed,
        potential_stereo_loci=potential_loci,
        input_stereo_status=("underspecified" if unresolved else "specified"),
        unspecified_stereo_loci=unresolved,
        **_configuration_result_metadata(stereo_configuration),
    )


def is_molecular_chiral(
    molecule: Chem.Mol,
    *,
    stereo_complete: bool = True,
    require_specified: bool = False,
    stereo_configuration: MolecularStereoConfiguration | None = None,
) -> bool:
    """Return the boolean whole-molecule chirality classification."""
    return classify_molecular_chirality(
        molecule,
        stereo_complete=stereo_complete,
        require_specified=require_specified,
        stereo_configuration=stereo_configuration,
    ).is_chiral


def _canonical_isomeric_smiles(molecule: Chem.Mol) -> str:
    working = Chem.Mol(molecule)
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(working, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=4096)
def _cached_isomer_classification(
    isomeric_smiles: str,
    stereo_complete: bool,
) -> MolecularChirality:
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected cached isomeric SMILES: {isomeric_smiles}")
    return classify_molecular_chirality(
        molecule,
        stereo_complete=stereo_complete,
        require_specified=False,
    ).classification


def clear_molecular_chirality_cache() -> None:
    """Clear the bounded cache used by configuration-aware enumeration."""
    _cached_isomer_classification.cache_clear()


@dataclass(frozen=True)
class _TetrahedralCompletionAnalysis:
    """Exact achiral-existence result over unresolved tetrahedral assignments."""

    achiral_witness: str | None
    baseline_smiles: str
    baseline_classification: MolecularChirality


def _parity_constraint_assignment(
    descriptors: Mapping[int, TetrahedralStereo],
    variable_centers: frozenset[int],
    mapping: Mapping[int, int],
) -> dict[int, int] | None:
    """Solve mirror-compatibility XOR constraints for one automorphism."""
    constraints: dict[int, list[tuple[int, int]]] = {
        center: [] for center in descriptors
    }
    for center, source in descriptors.items():
        target = descriptors.get(mapping[center])
        if target is None:
            return None
        transported = source.relabel(mapping)
        if transported == target.invert():
            relation = 0
        elif transported == target:
            relation = 1
        else:
            return None
        constraints[center].append((target.center, relation))
        constraints[target.center].append((center, relation))

    values = {
        center: 0 for center in descriptors if center not in variable_centers
    }
    for root in descriptors:
        if root not in values:
            values[root] = 0
        pending = [root]
        while pending:
            left = pending.pop()
            for right, relation in constraints[left]:
                expected = values[left] ^ relation
                if right in values:
                    if values[right] != expected:
                        return None
                    continue
                values[right] = expected
                pending.append(right)
    return values


def _flip_tetrahedral_assignments(
    baseline: Chem.Mol,
    variable_centers: frozenset[int],
    values: Mapping[int, int],
) -> Chem.Mol | None:
    """Apply a solved descriptor-parity assignment to an RDKit baseline."""
    witness = Chem.Mol(baseline)
    for center in variable_centers:
        if not values[center]:
            continue
        atom = witness.GetAtomWithIdx(center - 1)
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        else:
            return None
    return witness


def _analyze_unresolved_tetrahedral_completions(
    molecule: Chem.Mol,
    *,
    stereo_complete: bool,
    automorphism_limit: int = 4096,
) -> _TetrahedralCompletionAnalysis | None:
    """Prove whether any unresolved tetrahedral completion is achiral.

    The raw assignment space grows as ``2**n``.  Achirality, however, only
    requires one constitutional automorphism whose transported local
    configurations equal the mirror configurations.  For each automorphism,
    those requirements are binary XOR constraints.  Exhausting the much
    smaller automorphism set therefore proves nonexistence without enumerating
    every stereoisomer.

    ``None`` means this exact shortcut is outside its supported boundary or
    exceeded its automorphism guard; callers must retain fail-closed capped
    enumeration semantics.
    """
    unresolved = tuple(
        info
        for info in Chem.FindPotentialStereo(molecule)
        if info.specified != Chem.StereoSpecified.Specified
    )
    if not unresolved or any(
        info.type != Chem.StereoType.Atom_Tetrahedral for info in unresolved
    ):
        return None
    variable_centers = frozenset(
        int(info.centeredOn) + 1 for info in unresolved
    )
    options = StereoEnumerationOptions(
        tryEmbedding=False,
        onlyUnassigned=True,
        maxIsomers=1,
        rand=0x5A17,
        unique=True,
    )
    baseline = next(iter(EnumerateStereoisomers(molecule, options=options)), None)
    if baseline is None:
        return None
    baseline_smiles = _canonical_isomeric_smiles(baseline)
    baseline_classification = classify_molecular_chirality(
        baseline,
        stereo_complete=stereo_complete,
        require_specified=False,
    ).classification

    indexed = _indexed_copy(baseline)
    graph = MolToGraph(attr_profile="minimal").transform(
        indexed,
        use_index_as_atom_map=True,
    )
    registry = dict(graph.graph.get("stereo_descriptors", {}))
    if stereo_complete:
        _complete_tetrahedral_topology(indexed, registry)
    if any(not isinstance(value, TetrahedralStereo) for value in registry.values()):
        return None
    descriptors = {
        value.center: value for value in registry.values()
    }
    if not variable_centers.issubset(descriptors):
        return None

    graph.graph["stereo_descriptors"] = {}
    verification_failed = False
    matcher = _MolecularStereoGraphMatcher(graph, graph)
    for position, mapping in enumerate(matcher.isomorphisms_iter()):
        if position >= automorphism_limit:
            return None
        values = _parity_constraint_assignment(
            descriptors,
            variable_centers,
            mapping,
        )
        if values is None:
            continue
        witness = _flip_tetrahedral_assignments(
            baseline,
            variable_centers,
            values,
        )
        if witness is None:
            verification_failed = True
            continue
        witness_result = classify_molecular_chirality(
            witness,
            stereo_complete=stereo_complete,
            require_specified=False,
        )
        if witness_result.classification is MolecularChirality.ACHIRAL:
            return _TetrahedralCompletionAnalysis(
                achiral_witness=_canonical_isomeric_smiles(witness),
                baseline_smiles=baseline_smiles,
                baseline_classification=baseline_classification,
            )
        verification_failed = True

    if verification_failed or baseline_classification is MolecularChirality.ACHIRAL:
        return None
    return _TetrahedralCompletionAnalysis(
        achiral_witness=None,
        baseline_smiles=baseline_smiles,
        baseline_classification=baseline_classification,
    )


def _assessment_configuration_alternatives(
    molecule: Chem.Mol,
    configuration_set: MolecularStereoConfigurationSet | None,
) -> tuple[MolecularStereoConfiguration | None, ...]:
    if configuration_set is None:
        return (None,)
    for configuration in configuration_set.configurations:
        _validate_molecular_configuration(molecule, configuration)
    return configuration_set.configurations


def _covered_rdkit_binary_loci(
    molecule: Chem.Mol,
    configurations: tuple[MolecularStereoConfiguration | None, ...],
) -> int:
    configured = tuple(item for item in configurations if item is not None)
    if len(configured) != len(configurations):
        return 0
    return sum(
        info.specified != Chem.StereoSpecified.Specified
        and all(
            _configuration_covers_rdkit_locus(molecule, info, configuration)
            for configuration in configured
        )
        for info in Chem.FindPotentialStereo(molecule)
    )


def assess_molecular_chirality(
    molecule: Chem.Mol,
    *,
    max_isomers: int = 256,
    stereo_complete: bool = True,
    try_embedding: bool = False,
    use_cache: bool = True,
    stereo_configurations: MolecularStereoConfigurationSet | None = None,
) -> MolecularChiralityAssessment:
    """Assess chirality across supported completions of unresolved stereo.

    RDKit-supported unassigned tetrahedral atoms and double bonds are
    enumerated. A mixed chiral/achiral population proves
    ``configuration_dependent`` even when the search cap truncates the full
    population. For an oversized all-tetrahedral product, exact parity
    constraints over the constitutional automorphisms can independently prove
    that no achiral completion exists or construct an achiral witness. A
    one-sided truncated population without such a proof is never promoted to
    a necessary conclusion. Unsupported unresolved stereo types also fail
    closed as ``unsupported_or_incomplete``.
    """
    if molecule is None:
        raise ValueError("Molecular chirality assessment requires a molecule.")
    if type(max_isomers) is not int or max_isomers < 1:
        raise ValueError("max_isomers must be a positive integer.")

    working = Chem.Mol(molecule)
    configurations = _assessment_configuration_alternatives(
        working, stereo_configurations
    )
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(working, cleanIt=False, force=True)
    configured = tuple(
        configuration for configuration in configurations if configuration is not None
    )
    unspecified = _unspecified_stereo_loci(working, configured)
    potential_loci = detect_potential_stereo_loci(working)
    unsupported_potential = tuple(
        locus.identifier
        for locus in potential_loci
        if any(
            configuration is None
            or not _configuration_covers_locus(configuration, locus)
            for configuration in configurations
        )
    )
    unsupported = tuple(
        sorted((*_unsupported_stereo_loci(working),) + unsupported_potential)
    )
    if unsupported:
        return MolecularChiralityAssessment(
            outcome=MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE,
            observed_classifications=(),
            input_stereo_status="underspecified",
            unspecified_stereo_loci=unspecified,
            unsupported_stereo_loci=unsupported,
            theoretical_isomer_upper_bound=0,
            evaluated_isomer_count=0,
            enumeration_complete=False,
            max_isomers=max_isomers,
            configured_alternative_count=(
                0 if stereo_configurations is None else len(configurations)
            ),
            configured_population_status=(
                None
                if stereo_configurations is None
                else stereo_configurations.population_status.value
            ),
            decision_method="unsupported_locus_detection",
        )

    options = StereoEnumerationOptions(
        tryEmbedding=try_embedding,
        onlyUnassigned=True,
        maxIsomers=max_isomers,
        rand=0x5A17,
        unique=True,
    )
    raw_rdkit_theoretical = int(GetStereoisomerCount(working, options=options))
    covered_binary = _covered_rdkit_binary_loci(working, configurations)
    rdkit_theoretical = max(1, raw_rdkit_theoretical // (2**covered_binary))
    theoretical = rdkit_theoretical * len(configurations)
    classifications: set[MolecularChirality] = set()
    representatives: dict[MolecularChirality, str] = {}
    evaluated = 0
    stopped_after_decisive_mixture = False
    stopped_at_cap = False
    decision_method = "stereo_completion_enumeration"

    tetrahedral_analysis = None
    if theoretical > max_isomers and configurations == (None,):
        tetrahedral_analysis = _analyze_unresolved_tetrahedral_completions(
            working,
            stereo_complete=stereo_complete,
        )
    if tetrahedral_analysis is not None:
        baseline_classification = tetrahedral_analysis.baseline_classification
        classifications.add(baseline_classification)
        representatives[baseline_classification] = (
            tetrahedral_analysis.baseline_smiles
        )
        evaluated = 1
        if tetrahedral_analysis.achiral_witness is None:
            decision_method = "automorphism_parity_nonexistence_proof"
        elif baseline_classification is MolecularChirality.CHIRAL:
            classifications.add(MolecularChirality.ACHIRAL)
            representatives[MolecularChirality.ACHIRAL] = (
                tetrahedral_analysis.achiral_witness
            )
            evaluated += 1
            stopped_after_decisive_mixture = True
            decision_method = "automorphism_parity_achiral_witness"

    analysis_is_decisive = (
        tetrahedral_analysis is not None
        and (
            tetrahedral_analysis.achiral_witness is None
            or len(classifications) > 1
        )
    )
    if not analysis_is_decisive:
        classifications.clear()
        representatives.clear()
        evaluated = 0
        for isomer in EnumerateStereoisomers(working, options=options):
            isomeric_smiles = _canonical_isomeric_smiles(isomer)
            for configuration in configurations:
                if evaluated >= max_isomers:
                    stopped_at_cap = True
                    break
                classification = (
                    _cached_isomer_classification(isomeric_smiles, stereo_complete)
                    if use_cache and configuration is None
                    else classify_molecular_chirality(
                        isomer,
                        stereo_complete=stereo_complete,
                        stereo_configuration=configuration,
                    ).classification
                )
                evaluated += 1
                classifications.add(classification)
                representatives.setdefault(classification, isomeric_smiles)
                if len(classifications) > 1:
                    stopped_after_decisive_mixture = True
                    break
            if stopped_after_decisive_mixture or stopped_at_cap:
                break

    observed = tuple(sorted(classifications, key=lambda value: value.value))
    evidence = tuple(
        (representatives[classification], classification) for classification in observed
    )
    enumeration_exhausted = (
        not analysis_is_decisive
        and not stopped_after_decisive_mixture
        and not stopped_at_cap
        and (
            theoretical <= max_isomers
            or evaluated < max_isomers
        )
    )
    if len(classifications) > 1:
        outcome = MolecularChiralityOutcome.CONFIGURATION_DEPENDENT
        if decision_method == "stereo_completion_enumeration":
            decision_method = "enumerated_mixed_population"
    elif (
        decision_method == "automorphism_parity_nonexistence_proof"
        and classifications == {MolecularChirality.CHIRAL}
    ):
        outcome = MolecularChiralityOutcome.NECESSARILY_CHIRAL
    elif not enumeration_exhausted or not classifications:
        outcome = MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE
        decision_method = "capped_stereo_completion_enumeration"
    elif MolecularChirality.CHIRAL in classifications:
        outcome = MolecularChiralityOutcome.NECESSARILY_CHIRAL
        decision_method = "complete_stereo_completion_enumeration"
    else:
        outcome = MolecularChiralityOutcome.NECESSARILY_ACHIRAL
        decision_method = "complete_stereo_completion_enumeration"

    return MolecularChiralityAssessment(
        outcome=outcome,
        observed_classifications=observed,
        input_stereo_status=("underspecified" if unspecified else "specified"),
        unspecified_stereo_loci=unspecified,
        unsupported_stereo_loci=(),
        theoretical_isomer_upper_bound=theoretical,
        evaluated_isomer_count=evaluated,
        enumeration_complete=enumeration_exhausted,
        max_isomers=max_isomers,
        representative_isomers=evidence,
        configured_alternative_count=(
            0 if stereo_configurations is None else len(configurations)
        ),
        configured_population_status=(
            None
            if stereo_configurations is None
            else stereo_configurations.population_status.value
        ),
        decision_method=decision_method,
    )
