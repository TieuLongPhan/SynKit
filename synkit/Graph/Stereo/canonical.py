"""Exact canonicalization of configured permutation-labelled stereographs.

Version 1 supports fixed tetrahedral and planar-bond descriptors.  It
faithfully expands each complete local configuration orbit into a coloured
incidence graph and delegates only graph canonical labeling to SynKit's native
exact kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Hashable

import networkx as nx

from synkit.Graph.Canon.exact import (
    AutomorphismWitness,
    ExactCanonicalResult,
    ExactColoredGraphCanonicalizer,
)

from .descriptors import (
    PlanarBondStereo,
    Reference,
    StereoValue,
    TetrahedralStereo,
    parse_virtual_reference,
)
from .orbits import SHAPE_DEFINITIONS, StereoSpecification

STEREOGRAPH_SCHEMA = "synkit.canonical-stereograph/1"

_DEFAULT_ATOM_COLOR_KEYS = (
    "element",
    "isotope",
    "charge",
    "formal_charge",
    "radical",
    "radical_electrons",
    "aromatic",
    "hcount",
    "lone_pairs",
)
_DEFAULT_BOND_COLOR_KEYS = (
    "order",
    "standard_order",
    "bond_type",
    "aromatic",
)

AttributeSelector = str | Sequence[str] | Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AtomVertex:
    """Auxiliary-graph handle for one base-graph atom."""

    reference: Hashable


@dataclass(frozen=True)
class BondVertex:
    """Auxiliary-graph handle for one base-graph bond/resource."""

    index: int


@dataclass(frozen=True)
class StereoLocusVertex:
    """Auxiliary-graph handle for one configured stereo locus."""

    index: int


@dataclass(frozen=True)
class StereoPortVertex:
    """One owner-specific ligand incidence at a stereo locus."""

    locus: int
    position: int


@dataclass(frozen=True)
class StereoTupleVertex:
    """One allowed ordered tuple in a local configuration orbit."""

    locus: int
    index: int


@dataclass(frozen=True)
class StereoSlotVertex:
    """Position-coloured connection between a tuple and one port."""

    locus: int
    tuple_index: int
    position: int


@dataclass(frozen=True)
class VirtualResourceVertex:
    """Owner-scoped hidden hydrogen or lone-pair resource."""

    locus: int
    position: int
    kind: str


@dataclass(frozen=True)
class CanonicalStereographResult:
    """Complete configured-stereograph identity and projected witnesses.

    ``canonical_code`` is the deterministic, versioned exact certificate.
    Atom/bond orders and projected automorphisms retain input handles and are
    witnesses modulo configured-stereograph automorphism, not unique choices
    among constitutionally symmetric atoms or ports.
    """

    schema: str
    canonical_code: str
    canonical_digest: str
    atom_order: tuple[Hashable, ...]
    bond_order: tuple[tuple[Hashable, Hashable], ...]
    locus_order: tuple[int, ...]
    port_order: tuple[StereoPortVertex, ...]
    atom_automorphisms: tuple[AutomorphismWitness, ...]
    locus_automorphisms: tuple[AutomorphismWitness, ...]
    port_automorphisms: tuple[AutomorphismWitness, ...]
    auxiliary: ExactCanonicalResult

    @property
    def complete(self) -> bool:
        return self.auxiliary.complete

    @property
    def exact(self) -> bool:
        return self.auxiliary.exact

    @property
    def canonical_certificate(self) -> str:
        """Return the exact, versioned stereograph serialization."""
        return self.canonical_code

    def same_stereograph(self, other: object) -> bool:
        """Compare exact versioned certificates, not finite digests."""
        return (
            isinstance(other, CanonicalStereographResult)
            and self.schema == other.schema
            and self.auxiliary.same_canonical_graph(other.auxiliary)
        )


class StereographMirrorStatus(str, Enum):
    """Exact mirror-comparison outcome for a configured stereograph."""

    ACHIRAL = "achiral"
    CHIRAL = "chiral"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StereographMirrorResult:
    """Evidence for exact comparison with a geometry-specific mirror.

    Definitive results retain both complete exact certificates.  Incomplete
    or unsupported inputs deliberately return no plausible partial identity.
    """

    status: StereographMirrorStatus
    descriptor_count: int
    original: CanonicalStereographResult | None = None
    mirror: CanonicalStereographResult | None = None
    mirrored_descriptors: tuple[StereoValue, ...] = ()
    atom_mirror_isomorphism: tuple[tuple[Hashable, Hashable], ...] | None = None
    incomplete_loci: tuple[str, ...] = ()
    unsupported_loci: tuple[str, ...] = ()
    unsupported_families: tuple[str, ...] = ()
    method: str = "exact_canonical_stereograph_mirror_comparison"

    @property
    def is_definitive(self) -> bool:
        return self.status in {
            StereographMirrorStatus.ACHIRAL,
            StereographMirrorStatus.CHIRAL,
        }

    @property
    def is_chiral(self) -> bool | None:
        if not self.is_definitive:
            return None
        return self.status is StereographMirrorStatus.CHIRAL


@dataclass(frozen=True)
class _Expansion:
    graph: nx.Graph
    bond_endpoints: tuple[tuple[Hashable, Hashable], ...]


def _selected(
    attributes: Mapping[str, Any],
    selector: AttributeSelector,
) -> Any:
    if isinstance(selector, str):
        return attributes.get(selector)
    if callable(selector):
        return selector(attributes)
    return tuple(attributes.get(key) for key in selector)


def _add_coloured_node(
    graph: nx.Graph,
    node: Hashable,
    color: tuple[Any, ...],
) -> None:
    graph.add_node(node, color=color)


def _validate_base_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError(
            "Configured molecular stereographs currently require an "
            "undirected NetworkX Graph."
        )
    if graph.is_multigraph():
        raise TypeError(
            "Parallel resources must already be represented as typed "
            "incidence vertices."
        )


def _material_port_resource(
    base: nx.Graph,
    bonds: Mapping[tuple[Hashable, Hashable], BondVertex],
    owner: Hashable,
    reference: Reference,
    *,
    geometry: str,
) -> BondVertex:
    if reference not in base:
        raise ValueError(f"{geometry} ligand reference {reference!r} is absent.")
    if not base.has_edge(owner, reference):
        raise ValueError(
            f"{geometry} ligand {reference!r} is not bonded to owner {owner!r}."
        )
    return bonds[(owner, reference)]


def _add_configuration_orbit(
    auxiliary: nx.Graph,
    locus: StereoLocusVertex,
    descriptor: TetrahedralStereo | PlanarBondStereo,
    locus_index: int,
    targets: Mapping[Reference, Hashable],
    frame_positions: Sequence[int],
) -> None:
    definition = SHAPE_DEFINITIONS[descriptor.descriptor_class]
    orbit = definition.preserving_group.orbit(descriptor.configuration.frame)
    for tuple_index, frame in enumerate(orbit):
        relation = StereoTupleVertex(locus_index, tuple_index)
        _add_coloured_node(
            auxiliary,
            relation,
            ("stereo_tuple", descriptor.descriptor_class),
        )
        auxiliary.add_edge(locus, relation)
        for slot_position, frame_position in enumerate(frame_positions):
            reference = frame[frame_position]
            slot = StereoSlotVertex(locus_index, tuple_index, slot_position)
            _add_coloured_node(
                auxiliary,
                slot,
                (
                    "stereo_slot",
                    descriptor.descriptor_class,
                    slot_position,
                ),
            )
            auxiliary.add_edge(relation, slot)
            auxiliary.add_edge(slot, targets[reference])


def _add_tetrahedral_locus(
    auxiliary: nx.Graph,
    base: nx.Graph,
    atom_vertices: Mapping[Hashable, AtomVertex],
    bond_vertices: Mapping[tuple[Hashable, Hashable], BondVertex],
    descriptor: TetrahedralStereo,
    locus_index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError(
            "Canonical stereograph Version 1 requires fixed tetrahedral "
            "configuration."
        )
    center = descriptor.center
    if center not in atom_vertices:
        raise ValueError(f"Tetrahedral center {center!r} is absent.")
    locus = StereoLocusVertex(locus_index)
    _add_coloured_node(
        auxiliary,
        locus,
        ("stereo_locus", descriptor.descriptor_class),
    )
    auxiliary.add_edge(locus, atom_vertices[center])

    targets: dict[Reference, Hashable] = {center: atom_vertices[center]}
    for position, reference in enumerate(descriptor.atoms[1:]):
        port = StereoPortVertex(locus_index, position)
        targets[reference] = port
        _add_coloured_node(
            auxiliary,
            port,
            ("stereo_port", descriptor.descriptor_class),
        )
        auxiliary.add_edge(locus, port)
        virtual = parse_virtual_reference(reference)
        if virtual is None:
            resource = _material_port_resource(
                base,
                bond_vertices,
                center,
                reference,
                geometry="Tetrahedral",
            )
        else:
            if virtual.center != center:
                raise ValueError(
                    f"Virtual ligand {reference!r} does not belong to "
                    f"center {center!r}."
                )
            resource = VirtualResourceVertex(
                locus_index,
                position,
                virtual.kind,
            )
            _add_coloured_node(
                auxiliary,
                resource,
                ("virtual_resource", virtual.kind),
            )
            auxiliary.add_edge(atom_vertices[center], resource)
        auxiliary.add_edge(port, resource)

    _add_configuration_orbit(
        auxiliary,
        locus,
        descriptor,
        locus_index,
        targets,
        (1, 2, 3, 4),
    )


def _add_planar_bond_locus(
    auxiliary: nx.Graph,
    base: nx.Graph,
    atom_vertices: Mapping[Hashable, AtomVertex],
    bond_vertices: Mapping[tuple[Hashable, Hashable], BondVertex],
    descriptor: PlanarBondStereo,
    locus_index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError(
            "Canonical stereograph Version 1 requires fixed planar-bond "
            "configuration."
        )
    left, right = descriptor.atoms[2:4]
    if left not in atom_vertices or right not in atom_vertices:
        raise ValueError(
            f"Planar-bond locus {left!r}-{right!r} has an absent endpoint."
        )
    if not base.has_edge(left, right):
        raise ValueError(
            f"Planar-bond locus {left!r}-{right!r} is not a base-graph bond."
        )
    locus = StereoLocusVertex(locus_index)
    _add_coloured_node(
        auxiliary,
        locus,
        ("stereo_locus", descriptor.descriptor_class),
    )
    auxiliary.add_edge(locus, bond_vertices[(left, right)])
    targets: dict[Reference, Hashable] = {
        left: atom_vertices[left],
        right: atom_vertices[right],
    }
    ligand_positions = ((0, left), (1, left), (4, right), (5, right))
    for port_position, (frame_position, owner) in enumerate(ligand_positions):
        reference = descriptor.atoms[frame_position]
        port = StereoPortVertex(locus_index, port_position)
        targets[reference] = port
        _add_coloured_node(
            auxiliary,
            port,
            ("stereo_port", descriptor.descriptor_class),
        )
        auxiliary.add_edge(locus, port)
        virtual = parse_virtual_reference(reference)
        if virtual is None:
            resource = _material_port_resource(
                base,
                bond_vertices,
                owner,
                reference,
                geometry="Planar-bond",
            )
        else:
            if virtual.center != owner:
                raise ValueError(
                    f"Virtual ligand {reference!r} does not belong to "
                    f"planar-bond endpoint {owner!r}."
                )
            resource = VirtualResourceVertex(
                locus_index,
                port_position,
                virtual.kind,
            )
            _add_coloured_node(
                auxiliary,
                resource,
                ("virtual_resource", virtual.kind),
            )
            auxiliary.add_edge(atom_vertices[owner], resource)
        auxiliary.add_edge(port, resource)
    _add_configuration_orbit(
        auxiliary,
        locus,
        descriptor,
        locus_index,
        targets,
        (0, 1, 2, 3, 4, 5),
    )


def expand_tetrahedral_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[TetrahedralStereo],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> nx.Graph:
    """Return the blueprint auxiliary graph for fixed tetrahedral stereo."""
    fixed = _require_tetrahedral(tuple(descriptors))
    return _expand(
        base_graph,
        fixed,
        atom_color=atom_color,
        bond_color=bond_color,
    ).graph


def expand_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[TetrahedralStereo | PlanarBondStereo],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> nx.Graph:
    """Expand a fixed tetrahedral/planar-bond Version 1 stereograph."""
    return _expand(
        base_graph,
        tuple(descriptors),
        atom_color=atom_color,
        bond_color=bond_color,
    ).graph


def _require_tetrahedral(
    descriptors: tuple[StereoValue, ...],
) -> tuple[TetrahedralStereo, ...]:
    if any(not isinstance(item, TetrahedralStereo) for item in descriptors):
        raise TypeError(
            "This tetrahedral compatibility API accepts only "
            "TetrahedralStereo descriptors."
        )
    return descriptors  # type: ignore[return-value]


def _expand(
    base_graph: nx.Graph,
    descriptors: tuple[TetrahedralStereo | PlanarBondStereo, ...],
    *,
    atom_color: AttributeSelector,
    bond_color: AttributeSelector,
) -> _Expansion:
    _validate_base_graph(base_graph)
    base = base_graph.copy()
    auxiliary = nx.Graph()
    atom_vertices = {reference: AtomVertex(reference) for reference in base}
    for reference, vertex in atom_vertices.items():
        _add_coloured_node(
            auxiliary,
            vertex,
            ("atom", _selected(base.nodes[reference], atom_color)),
        )

    incidence: dict[tuple[Hashable, Hashable], BondVertex] = {}
    endpoints = []
    for index, (left, right, attributes) in enumerate(base.edges(data=True)):
        vertex = BondVertex(index)
        endpoints.append((left, right))
        incidence[(left, right)] = vertex
        incidence[(right, left)] = vertex
        _add_coloured_node(
            auxiliary,
            vertex,
            ("bond_resource", _selected(attributes, bond_color)),
        )
        auxiliary.add_edge(atom_vertices[left], vertex)
        auxiliary.add_edge(vertex, atom_vertices[right])

    seen_loci: set[tuple[str, Hashable]] = set()
    for locus_index, descriptor in enumerate(descriptors):
        if isinstance(descriptor, TetrahedralStereo):
            locus_key = ("tetrahedral", descriptor.center)
        elif isinstance(descriptor, PlanarBondStereo):
            locus_key = ("planar_bond", frozenset(descriptor.atoms[2:4]))
        else:
            raise TypeError(
                "Canonical stereograph Version 1 accepts only fixed "
                "tetrahedral and planar-bond descriptors."
            )
        if locus_key in seen_loci:
            raise ValueError(f"Duplicate configured stereo locus at {locus_key[1]!r}.")
        seen_loci.add(locus_key)
        if isinstance(descriptor, TetrahedralStereo):
            _add_tetrahedral_locus(
                auxiliary,
                base,
                atom_vertices,
                incidence,
                descriptor,
                locus_index,
            )
        else:
            _add_planar_bond_locus(
                auxiliary,
                base,
                atom_vertices,
                incidence,
                descriptor,
                locus_index,
            )
    return _Expansion(auxiliary, tuple(endpoints))


def _project_witnesses(
    result: ExactCanonicalResult,
    vertex_type: type[AtomVertex] | type[StereoLocusVertex] | type[StereoPortVertex],
) -> tuple[AutomorphismWitness, ...]:
    projected = []
    for witness in result.automorphisms:
        mapping = witness.as_dict()
        pairs = []
        for source, target in mapping.items():
            if not isinstance(source, vertex_type):
                continue
            if not isinstance(target, vertex_type):
                raise RuntimeError(
                    "Auxiliary automorphism changed a semantic vertex role."
                )
            if isinstance(source, AtomVertex):
                pairs.append((source.reference, target.reference))
            elif isinstance(source, StereoLocusVertex):
                pairs.append((source.index, target.index))
            else:
                pairs.append((source, target))
        projected.append(AutomorphismWitness(tuple(pairs)))
    return tuple(projected)


def canonicalize_tetrahedral_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[TetrahedralStereo],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> CanonicalStereographResult:
    """Return exact Version 1 identity for configured tetrahedral stereo."""
    fixed = _require_tetrahedral(tuple(descriptors))
    return canonicalize_stereograph(
        base_graph,
        fixed,
        atom_color=atom_color,
        bond_color=bond_color,
    )


def canonicalize_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[TetrahedralStereo | PlanarBondStereo],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> CanonicalStereographResult:
    """Return exact Version 1 identity for configured organic stereo."""
    expansion = _expand(
        base_graph,
        tuple(descriptors),
        atom_color=atom_color,
        bond_color=bond_color,
    )
    auxiliary = ExactColoredGraphCanonicalizer(
        expansion.graph,
        node_color="color",
        edge_color=None,
    ).canonicalize()
    atom_order = tuple(
        node.reference
        for node in auxiliary.canonical_order
        if isinstance(node, AtomVertex)
    )
    bond_order = tuple(
        expansion.bond_endpoints[node.index]
        for node in auxiliary.canonical_order
        if isinstance(node, BondVertex)
    )
    locus_order = tuple(
        node.index
        for node in auxiliary.canonical_order
        if isinstance(node, StereoLocusVertex)
    )
    port_order = tuple(
        node for node in auxiliary.canonical_order if isinstance(node, StereoPortVertex)
    )
    canonical_code = f"{STEREOGRAPH_SCHEMA}\n{auxiliary.canonical_code}"
    return CanonicalStereographResult(
        schema=STEREOGRAPH_SCHEMA,
        canonical_code=canonical_code,
        canonical_digest=hashlib.sha256(canonical_code.encode("utf-8")).hexdigest(),
        atom_order=atom_order,
        bond_order=bond_order,
        locus_order=locus_order,
        port_order=port_order,
        atom_automorphisms=_project_witnesses(auxiliary, AtomVertex),
        locus_automorphisms=_project_witnesses(
            auxiliary,
            StereoLocusVertex,
        ),
        port_automorphisms=_project_witnesses(
            auxiliary,
            StereoPortVertex,
        ),
        auxiliary=auxiliary,
    )


def canonicalize_tetrahedral_registry(
    base_graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> CanonicalStereographResult:
    """Canonicalize a registry only when every entry is tetrahedral.

    Rejecting other configured families is deliberate: silently omitting an
    E/Z, axial, or coordination descriptor would produce an incomplete
    stereochemical identity.
    """
    descriptors = tuple(registry.values())
    unsupported = tuple(
        descriptor.descriptor_class
        for descriptor in descriptors
        if not isinstance(descriptor, TetrahedralStereo)
    )
    if unsupported:
        families = ", ".join(sorted(set(unsupported)))
        raise TypeError(
            "Tetrahedral canonicalization cannot omit configured stereo "
            f"families: {families}."
        )
    return canonicalize_tetrahedral_stereograph(
        base_graph,
        descriptors,
        atom_color=atom_color,
        bond_color=bond_color,
    )


def canonicalize_stereo_registry(
    base_graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> CanonicalStereographResult:
    """Canonicalize a registry containing only Version 1 stereo families."""
    descriptors = tuple(registry.values())
    unsupported = tuple(
        descriptor.descriptor_class
        for descriptor in descriptors
        if not isinstance(descriptor, (TetrahedralStereo, PlanarBondStereo))
    )
    if unsupported:
        families = ", ".join(sorted(set(unsupported)))
        raise TypeError(
            "Canonical stereograph Version 1 cannot omit configured stereo "
            f"families: {families}."
        )
    return canonicalize_stereograph(
        base_graph,
        descriptors,
        atom_color=atom_color,
        bond_color=bond_color,
    )


def mirror_stereo_descriptor(
    descriptor: StereoValue,
) -> TetrahedralStereo | PlanarBondStereo:
    """Apply the Version 1 spatial-reflection action to one descriptor.

    Tetrahedral handedness changes to its binary opposite.  Planar-bond E/Z
    configuration is retained: reflection does not turn an E alkene into Z.
    Unknown configurations remain unknown for this standalone transform.
    """
    if isinstance(descriptor, TetrahedralStereo):
        return descriptor.opposite()
    if isinstance(descriptor, PlanarBondStereo):
        return descriptor
    raise TypeError(
        "Canonical stereograph Version 1 has no mirror action for "
        f"{descriptor.descriptor_class!r}."
    )


def mirror_stereo_registry(
    registry: Mapping[str, StereoValue],
) -> dict[str, TetrahedralStereo | PlanarBondStereo]:
    """Return a geometry-specific Version 1 mirror without changing loci."""
    return {
        identifier: mirror_stereo_descriptor(descriptor)
        for identifier, descriptor in registry.items()
    }


def _nondefinitive_mirror_result(
    status: StereographMirrorStatus,
    descriptor_count: int,
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
    unsupported_families: Iterable[str] = (),
) -> StereographMirrorResult:
    return StereographMirrorResult(
        status=status,
        descriptor_count=descriptor_count,
        incomplete_loci=tuple(sorted(set(incomplete_loci))),
        unsupported_loci=tuple(sorted(set(unsupported_loci))),
        unsupported_families=tuple(sorted(set(unsupported_families))),
    )


def classify_stereograph_mirror(
    base_graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> StereographMirrorResult:
    """Compare a complete Version 1 stereograph with its exact mirror.

    ``incomplete_loci`` and ``unsupported_loci`` let a perception boundary
    declare missing configured information without inventing descriptors.
    Unsupported evidence takes precedence over incomplete evidence.
    """
    descriptors = tuple(registry.values())
    unsupported_entries = tuple(
        (identifier, descriptor.descriptor_class)
        for identifier, descriptor in registry.items()
        if not isinstance(descriptor, (TetrahedralStereo, PlanarBondStereo))
    )
    declared_unsupported = tuple(unsupported_loci)
    if unsupported_entries or declared_unsupported:
        return _nondefinitive_mirror_result(
            StereographMirrorStatus.UNSUPPORTED,
            len(descriptors),
            unsupported_loci=(
                *declared_unsupported,
                *(identifier for identifier, _family in unsupported_entries),
            ),
            unsupported_families=(
                family for _identifier, family in unsupported_entries
            ),
        )
    declared_incomplete = tuple(incomplete_loci)
    unknown = tuple(
        identifier
        for identifier, descriptor in registry.items()
        if descriptor.specification is StereoSpecification.UNSPECIFIED
    )
    if unknown or declared_incomplete:
        return _nondefinitive_mirror_result(
            StereographMirrorStatus.INCOMPLETE,
            len(descriptors),
            incomplete_loci=(*declared_incomplete, *unknown),
        )

    original = canonicalize_stereo_registry(
        base_graph,
        registry,
        atom_color=atom_color,
        bond_color=bond_color,
    )
    mirrored_registry = mirror_stereo_registry(registry)
    mirror = canonicalize_stereo_registry(
        base_graph,
        mirrored_registry,
        atom_color=atom_color,
        bond_color=bond_color,
    )
    achiral = original.same_stereograph(mirror)
    atom_mapping = (
        tuple(zip(original.atom_order, mirror.atom_order)) if achiral else None
    )
    return StereographMirrorResult(
        status=(
            StereographMirrorStatus.ACHIRAL
            if achiral
            else StereographMirrorStatus.CHIRAL
        ),
        descriptor_count=len(descriptors),
        original=original,
        mirror=mirror,
        mirrored_descriptors=tuple(mirrored_registry.values()),
        atom_mirror_isomorphism=atom_mapping,
    )


def _rdkit_graph_and_registry(
    molecule: Any,
) -> tuple[nx.Graph, Mapping[str, StereoValue]]:
    """Return one defensive RDKit graph/descriptor identifier namespace."""
    from rdkit import Chem

    from synkit.IO.mol_to_graph import MolToGraph

    from .rdkit_adapter import descriptors_from_rdkit

    if not isinstance(molecule, Chem.Mol):
        raise TypeError("RDKit stereograph canonicalization requires Chem.Mol.")
    working = Chem.Mol(molecule)
    atom_maps = tuple(int(atom.GetAtomMapNum()) for atom in working.GetAtoms())
    fully_mapped = all(value > 0 for value in atom_maps) and len(set(atom_maps)) == len(
        atom_maps
    )
    graph = MolToGraph(include_stereo_descriptors=False).transform(
        working,
        use_index_as_atom_map=fully_mapped,
    )
    registry = descriptors_from_rdkit(
        working,
        require_atom_maps=fully_mapped,
    )
    return graph, registry


def canonicalize_rdkit_tetrahedral_stereograph(
    molecule: Any,
) -> CanonicalStereographResult:
    """Canonicalize RDKit constitution plus its configured tetrahedral stereo.

    A defensive molecule copy is converted with atom-map node IDs only when
    every atom has a unique positive map.  Otherwise both the molecular graph
    and descriptors use one-based RDKit-index IDs.  Atom maps, CIP labels,
    coordinates, and RDKit local chiral tags do not enter semantic colours.
    """
    graph, registry = _rdkit_graph_and_registry(molecule)
    return canonicalize_tetrahedral_registry(graph, registry)


def canonicalize_rdkit_stereograph(
    molecule: Any,
) -> CanonicalStereographResult:
    """Canonicalize RDKit constitution plus configured tetrahedral/E/Z stereo."""
    graph, registry = _rdkit_graph_and_registry(molecule)
    return canonicalize_stereo_registry(graph, registry)


def classify_rdkit_stereograph_mirror(
    molecule: Any,
) -> StereographMirrorResult:
    """Classify configured RDKit input through exact Version 1 mirror identity.

    RDKit and SynKit perception are used only to report supplied-information
    state.  They do not provide E/Z or CIP labels to the exact certificate.
    """
    from synkit.Chem.Molecule.stereo_perception import (
        StereoConfigurationState,
        StereoElementType,
        detect_potential_stereo_elements,
    )

    graph, registry = _rdkit_graph_and_registry(molecule)
    elements = detect_potential_stereo_elements(molecule)
    version_one = {
        StereoElementType.TETRAHEDRAL,
        StereoElementType.DOUBLE_BOND,
    }
    incomplete = tuple(
        element.identifier
        for element in elements
        if element.element_type in version_one
        and element.configuration_state is StereoConfigurationState.UNSPECIFIED
    )
    unsupported = tuple(
        element.identifier
        for element in elements
        if element.element_type not in version_one
    )
    stereo_groups = tuple(molecule.GetStereoGroups())
    enhanced_groups = tuple(
        f"enhanced_stereo_group:{index}:{group.GetGroupType()}"
        for index, group in enumerate(stereo_groups)
    )
    return classify_stereograph_mirror(
        graph,
        registry,
        incomplete_loci=incomplete,
        unsupported_loci=(*unsupported, *enhanced_groups),
    )


__all__ = [
    "AtomVertex",
    "BondVertex",
    "CanonicalStereographResult",
    "STEREOGRAPH_SCHEMA",
    "StereographMirrorResult",
    "StereographMirrorStatus",
    "StereoLocusVertex",
    "StereoPortVertex",
    "StereoSlotVertex",
    "StereoTupleVertex",
    "VirtualResourceVertex",
    "classify_rdkit_stereograph_mirror",
    "classify_stereograph_mirror",
    "canonicalize_rdkit_stereograph",
    "canonicalize_rdkit_tetrahedral_stereograph",
    "canonicalize_stereo_registry",
    "canonicalize_stereograph",
    "canonicalize_tetrahedral_registry",
    "canonicalize_tetrahedral_stereograph",
    "expand_stereograph",
    "expand_tetrahedral_stereograph",
    "mirror_stereo_descriptor",
    "mirror_stereo_registry",
]
