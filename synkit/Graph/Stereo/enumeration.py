"""Exact perception refinement and configuration enumeration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import hashlib
from math import prod
from typing import Any, Hashable

from networkx.utils import UnionFind

from synkit.Graph.Canon.exact import (
    ExactCanonicalResult,
    ExactColoredGraphCanonicalizer,
)

from .canonical import (
    AtomVertex,
    AttributeSelector,
    StereoLocusVertex,
    StereoPortVertex,
    StereoSlotVertex,
    StereoTupleVertex,
    StereographMirrorStatus,
    _DEFAULT_ATOM_COLOR_KEYS,
    _DEFAULT_BOND_COLOR_KEYS,
    _rdkit_graph_and_registry,
)
from .configured import (
    CONFIGURED_DESCRIPTOR_TYPES,
    ConfiguredDescriptor,
    canonicalize_configured_registry,
    classify_configured_stereograph_mirror,
    expand_configured_stereograph,
)
from .descriptors import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)
from .extended_descriptors import HelicalStereo, PlanarChiralityStereo
from .orbits import SHAPE_DEFINITIONS, StereoSpecification

FOCAL_STEREOGRAPH_SCHEMA = "synkit.focal-stereograph/1"
STEREO_ENUMERATION_SCHEMA = "synkit.stereograph-enumeration/1"


class StereoEnumerationLimitError(RuntimeError):
    """Raised before enumeration when the complete product exceeds its cap."""

    def __init__(self, theoretical_count: int, limit: int) -> None:
        self.theoretical_count = theoretical_count
        self.limit = limit
        super().__init__(
            f"Complete stereo enumeration requires {theoretical_count} "
            f"assignments, exceeding limit {limit}."
        )


@dataclass(frozen=True)
class FocalStereoEvidence:
    """Exact symmetry evidence after omitting one focal configuration."""

    schema: str
    focal_identifier: str
    descriptor_class: str
    port_orbits: tuple[tuple[int, ...], ...]
    atom_orbits: tuple[frozenset[Hashable], ...]
    canonical_code: str
    canonical_digest: str
    auxiliary: ExactCanonicalResult

    @property
    def all_ports_distinct(self) -> bool:
        return all(len(orbit) == 1 for orbit in self.port_orbits)


@dataclass(frozen=True)
class StereoAssignment:
    """One exact whole-stereograph assignment class."""

    registry_items: tuple[tuple[str, StereoValue], ...]
    canonical_code: str
    canonical_digest: str
    mirror_status: StereographMirrorStatus

    def registry(self) -> dict[str, StereoValue]:
        return dict(self.registry_items)


@dataclass(frozen=True)
class StereoEnumerationResult:
    """Complete assignment quotient under exact stereo-aware automorphisms."""

    schema: str
    theoretical_assignment_count: int
    exact_assignment_count: int
    enantiomer_class_count: int
    unresolved_loci: tuple[str, ...]
    focal_evidence: tuple[FocalStereoEvidence, ...]
    assignments: tuple[StereoAssignment, ...]
    complete: bool = True
    exact: bool = True


def _fixed_seed(descriptor: ConfiguredDescriptor) -> ConfiguredDescriptor:
    if descriptor.specification is StereoSpecification.FIXED:
        return descriptor
    if isinstance(descriptor, SquarePlanarStereo):
        return SquarePlanarStereo(descriptor.atoms, 0, descriptor.provenance)
    if isinstance(
        descriptor,
        (
            TetrahedralStereo,
            TrigonalBipyramidalStereo,
            OctahedralStereo,
            AtropBondStereo,
        ),
    ):
        return type(descriptor)(descriptor.atoms, 1, descriptor.provenance)
    if isinstance(descriptor, PlanarBondStereo):
        return PlanarBondStereo(descriptor.atoms, 0, descriptor.provenance)
    if isinstance(descriptor, CumuleneAxisStereo):
        return CumuleneAxisStereo(
            descriptor.axis_path,
            descriptor.terminal_frames,
            1,
            descriptor.provenance,
        )
    if isinstance(descriptor, ExtendedCisTransStereo):
        return ExtendedCisTransStereo(
            descriptor.path,
            descriptor.terminal_frames,
            0,
            descriptor.provenance,
        )
    if isinstance(descriptor, HelicalStereo):
        return HelicalStereo(
            descriptor.path,
            1,
            descriptor.provenance,
            descriptor.cyclic,
            descriptor.reported_positions,
            descriptor.coupling_id,
        )
    return PlanarChiralityStereo(
        descriptor.plane_atoms,
        descriptor.pilot,
        1,
        descriptor.provenance,
    )


def _shape_candidates(
    descriptor: ConfiguredDescriptor,
) -> tuple[ConfiguredDescriptor, ...]:
    seed = _fixed_seed(descriptor)
    if isinstance(
        seed,
        (
            CumuleneAxisStereo,
            ExtendedCisTransStereo,
            HelicalStereo,
            PlanarChiralityStereo,
        ),
    ):
        candidates = (seed, seed.invert())
    else:
        definition = SHAPE_DEFINITIONS[seed.descriptor_class]
        parity = (
            0
            if isinstance(
                seed,
                (
                    SquarePlanarStereo,
                    PlanarBondStereo,
                    ExtendedCisTransStereo,
                ),
            )
            else 1
        )
        candidates = tuple(
            type(seed)(
                permutation.apply(seed.configuration.frame),
                parity,
                seed.provenance,
            )
            for permutation in definition.unspecified_group.elements
        )
    unique: dict[tuple[Any, ...], ConfiguredDescriptor] = {}
    for candidate in candidates:
        form = candidate.canonical_form()
        unique.setdefault(form, candidate)
    return tuple(unique[key] for key in sorted(unique, key=repr))


def local_configuration_classes(
    descriptor: ConfiguredDescriptor,
) -> tuple[ConfiguredDescriptor, ...]:
    """Return every local configuration class for one declared support."""
    if not isinstance(descriptor, CONFIGURED_DESCRIPTOR_TYPES):
        raise TypeError("Unsupported configured stereo descriptor.")
    return _shape_candidates(descriptor)


def _focal_index(
    items: tuple[tuple[str, ConfiguredDescriptor], ...],
    focal_identifier: str,
) -> int:
    for index, (identifier, _descriptor) in enumerate(items):
        if identifier == focal_identifier:
            return index
    raise KeyError(f"Focal stereo locus is absent: {focal_identifier!r}.")


def _orbit_partition(
    auxiliary: ExactCanonicalResult,
    focal_index: int,
) -> tuple[tuple[int, ...], ...]:
    positions = {
        node: node.position
        for orbit in auxiliary.orbits
        for node in orbit
        if isinstance(node, StereoPortVertex) and node.locus == focal_index
    }
    union = UnionFind(positions)
    for orbit in auxiliary.orbits:
        focal = [
            node
            for node in orbit
            if isinstance(node, StereoPortVertex) and node.locus == focal_index
        ]
        for node in focal[1:]:
            union.union(focal[0], node)
    classes: dict[Hashable, list[int]] = {}
    for node, position in positions.items():
        classes.setdefault(union[node], []).append(position)
    return tuple(
        sorted(
            (tuple(sorted(values)) for values in classes.values()),
            key=lambda values: values,
        )
    )


def focal_stereo_evidence(
    base_graph,
    registry: Mapping[str, StereoValue],
    focal_identifier: str,
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> FocalStereoEvidence:
    """Canonize after omitting only the focal local configuration relation."""
    items = tuple(sorted(registry.items()))
    if any(not isinstance(value, CONFIGURED_DESCRIPTOR_TYPES) for _, value in items):
        raise TypeError("Focal evidence requires configured-catalogue descriptors.")
    index = _focal_index(items, focal_identifier)
    omitted = tuple(
        position
        for position, (_key, value) in enumerate(items)
        if position == index or value.specification is StereoSpecification.UNSPECIFIED
    )
    seeded = tuple(
        _fixed_seed(value) if position in omitted else value
        for position, (_key, value) in enumerate(items)
    )
    graph = expand_configured_stereograph(
        base_graph,
        seeded,  # type: ignore[arg-type]
        atom_color=atom_color,
        bond_color=bond_color,
    )
    focal = seeded[index]
    for omitted_index in omitted:
        omitted_descriptor = seeded[omitted_index]
        omitted_locus = StereoLocusVertex(omitted_index)
        if isinstance(omitted_descriptor, HelicalStereo):
            color = graph.nodes[omitted_locus]["color"]
            graph.nodes[omitted_locus]["color"] = (
                *color[:2],
                "configuration_omitted",
                *color[3:],
            )
            continue
        remove = [
            node
            for node in graph
            if (isinstance(node, StereoTupleVertex) and node.locus == omitted_index)
            or (isinstance(node, StereoSlotVertex) and node.locus == omitted_index)
        ]
        graph.remove_nodes_from(remove)
        if isinstance(omitted_descriptor, PlanarChiralityStereo):
            for reference in omitted_descriptor.dependencies:
                graph.add_edge(omitted_locus, AtomVertex(reference))
    auxiliary = ExactColoredGraphCanonicalizer(
        graph,
        node_color="color",
        edge_color=None,
    ).canonicalize()
    code = f"{FOCAL_STEREOGRAPH_SCHEMA}\n{auxiliary.canonical_code}"
    atom_orbits = tuple(
        frozenset(node.reference for node in orbit if isinstance(node, AtomVertex))
        for orbit in auxiliary.orbits
        if any(isinstance(node, AtomVertex) for node in orbit)
    )
    return FocalStereoEvidence(
        FOCAL_STEREOGRAPH_SCHEMA,
        focal_identifier,
        focal.descriptor_class,
        _orbit_partition(auxiliary, index),
        atom_orbits,
        code,
        hashlib.sha256(code.encode("utf-8")).hexdigest(),
        auxiliary,
    )


def enumerate_stereograph_assignments(
    base_graph,
    registry: Mapping[str, StereoValue],
    *,
    max_assignments: int = 4096,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> StereoEnumerationResult:
    """Enumerate and exactly quotient all declared unknown configurations."""
    if type(max_assignments) is not int or max_assignments < 1:
        raise ValueError("Stereo enumeration limit must be a positive integer.")
    items = tuple(sorted(registry.items()))
    if any(not isinstance(value, CONFIGURED_DESCRIPTOR_TYPES) for _, value in items):
        raise TypeError("Enumeration requires configured-catalogue descriptors.")
    unresolved = tuple(
        key
        for key, value in items
        if value.specification is StereoSpecification.UNSPECIFIED
    )
    choices = tuple(
        (
            local_configuration_classes(value)  # type: ignore[arg-type]
            if key in unresolved
            else (value,)
        )
        for key, value in items
    )
    theoretical = prod(map(len, choices))
    if theoretical > max_assignments:
        raise StereoEnumerationLimitError(theoretical, max_assignments)
    focal = tuple(
        focal_stereo_evidence(
            base_graph,
            registry,
            key,
            atom_color=atom_color,
            bond_color=bond_color,
        )
        for key in unresolved
    )
    exact: dict[str, StereoAssignment] = {}
    mirror_classes: set[tuple[str, str]] = set()
    for values in product(*choices):
        assignment_registry = {
            key: value for (key, _source), value in zip(items, values)
        }
        canonical = canonicalize_configured_registry(
            base_graph,
            assignment_registry,
            atom_color=atom_color,
            bond_color=bond_color,
        )
        mirror = classify_configured_stereograph_mirror(
            base_graph,
            assignment_registry,
            atom_color=atom_color,
            bond_color=bond_color,
        )
        mirror_code = (
            canonical.canonical_code
            if mirror.mirror is None
            else mirror.mirror.canonical_code
        )
        mirror_classes.add(tuple(sorted((canonical.canonical_code, mirror_code))))
        exact.setdefault(
            canonical.canonical_code,
            StereoAssignment(
                tuple(sorted(assignment_registry.items())),
                canonical.canonical_code,
                canonical.canonical_digest,
                mirror.status,
            ),
        )
    assignments = tuple(exact[key] for key in sorted(exact))
    return StereoEnumerationResult(
        STEREO_ENUMERATION_SCHEMA,
        theoretical,
        len(assignments),
        len(mirror_classes),
        unresolved,
        focal,
        assignments,
    )


def _rdkit_identifier_map(molecule: Any) -> dict[int, int]:
    maps = tuple(int(atom.GetAtomMapNum()) for atom in molecule.GetAtoms())
    fully_mapped = all(value > 0 for value in maps) and len(set(maps)) == len(maps)
    return {
        atom.GetIdx(): (
            int(atom.GetAtomMapNum()) if fully_mapped else atom.GetIdx() + 1
        )
        for atom in molecule.GetAtoms()
    }


def _translate_reference(reference, identifiers: Mapping[int, int]):
    if type(reference) is int:
        return identifiers[reference]
    virtual = parse_virtual_reference(reference)
    if virtual is None:
        return reference
    return virtual_reference(virtual.kind, identifiers[virtual.center])


def _double_bond_unknown(molecule, support, identifiers):
    left_index, right_index = support.endpoints
    frames = []
    for owner_index, other_index in (
        (left_index, right_index),
        (right_index, left_index),
    ):
        atom = molecule.GetAtomWithIdx(owner_index)
        references = [
            identifiers[neighbor.GetIdx()]
            for neighbor in atom.GetNeighbors()
            if neighbor.GetIdx() != other_index
        ]
        if len(references) == 1:
            from .rdkit_adapter import _virtual_ligand

            references.append(_virtual_ligand(atom, identifiers[owner_index]))
        if len(references) != 2:
            raise ValueError("Potential planar bond lacks two endpoint references.")
        frames.append(tuple(references))
    left, right = identifiers[left_index], identifiers[right_index]
    return PlanarBondStereo(
        (*frames[0], left, right, *frames[1]),
        None,
        "rdkit_potential_stereo",
    )


def enumerate_rdkit_stereographs(
    molecule: Any,
    *,
    max_assignments: int = 4096,
) -> StereoEnumerationResult:
    """Perceive unresolved RDKit supports, then enumerate exact assignments."""
    from synkit.Chem.Molecule.stereo_perception import (
        StereoConfigurationState,
        StereoElementType,
        detect_potential_stereo_elements,
    )

    if tuple(molecule.GetStereoGroups()):
        raise TypeError("Enhanced stereo groups require population enumeration.")
    graph, configured = _rdkit_graph_and_registry(molecule)
    registry = dict(configured)
    identifiers = _rdkit_identifier_map(molecule)
    for element in detect_potential_stereo_elements(molecule):
        if element.configuration_state is not StereoConfigurationState.UNSPECIFIED:
            continue
        if element.element_type is StereoElementType.TETRAHEDRAL:
            evidence = element.constitutional_evidence
            if evidence is None or evidence.canonical_frame is None:
                continue
            atoms = tuple(
                _translate_reference(reference, identifiers)
                for reference in evidence.canonical_frame
            )
            descriptor: StereoValue = TetrahedralStereo(
                atoms,  # type: ignore[arg-type]
                None,
                "exact_focal_perception",
            )
        elif element.element_type is StereoElementType.DOUBLE_BOND:
            descriptor = _double_bond_unknown(
                molecule,
                element.support,
                identifiers,
            )
        elif element.element_type is StereoElementType.CUMULENE_AXIS:
            support = element.support
            descriptor = CumuleneAxisStereo(
                tuple(identifiers[atom] for atom in support.path),
                tuple(
                    tuple(
                        _translate_reference(reference, identifiers)
                        for reference in frame
                    )
                    for frame in support.terminal_frames
                ),  # type: ignore[arg-type]
                None,
                "exact_focal_perception",
            )
        elif element.element_type is StereoElementType.EXTENDED_CIS_TRANS:
            support = element.support
            descriptor = ExtendedCisTransStereo(
                tuple(identifiers[atom] for atom in support.path),
                tuple(
                    tuple(
                        _translate_reference(reference, identifiers)
                        for reference in frame
                    )
                    for frame in support.terminal_frames
                ),  # type: ignore[arg-type]
                None,
                "exact_focal_perception",
            )
        else:
            support = element.support
            descriptor = AtropBondStereo(
                (
                    *tuple(
                        _translate_reference(reference, identifiers)
                        for reference in support.terminal_frames[0]
                    ),
                    identifiers[support.path[0]],
                    identifiers[support.path[-1]],
                    *tuple(
                        _translate_reference(reference, identifiers)
                        for reference in support.terminal_frames[1]
                    ),
                ),  # type: ignore[arg-type]
                None,
                "exact_focal_perception",
            )
        key = descriptor_id(descriptor)
        registry.setdefault(key, descriptor)
    return enumerate_stereograph_assignments(
        graph,
        registry,
        max_assignments=max_assignments,
    )


__all__ = [
    "FOCAL_STEREOGRAPH_SCHEMA",
    "STEREO_ENUMERATION_SCHEMA",
    "FocalStereoEvidence",
    "StereoAssignment",
    "StereoEnumerationLimitError",
    "StereoEnumerationResult",
    "enumerate_stereograph_assignments",
    "enumerate_rdkit_stereographs",
    "focal_stereo_evidence",
    "local_configuration_classes",
]
