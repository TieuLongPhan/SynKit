"""Exact canonicalization of configured permutation-labelled stereographs.

The public entry points in this module cover SynKit's complete configured
stereo catalogue.  The tetrahedral-specific functions are retained as narrow
compatibility helpers, not as a separate stereograph model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Hashable, Literal

import networkx as nx

from synkit.Graph.Canon.exact import (
    AutomorphismWitness,
    ExactCanonicalResult,
)

from .descriptors import (
    Reference,
    StereoValue,
    TetrahedralStereo,
)
from .orbits import SHAPE_DEFINITIONS

# ``/2`` is the persisted wire-schema revision of the complete configured
# certificate.  It is deliberately not exposed as a second scientific model.
STEREOGRAPH_SCHEMA = "synkit.canonical-stereograph/2"

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
    descriptor: StereoValue,
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


def expand_tetrahedral_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[TetrahedralStereo],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> nx.Graph:
    """Expand tetrahedral stereo through the unified stereograph engine."""
    fixed = _require_tetrahedral(tuple(descriptors))
    return expand_stereograph(
        base_graph,
        fixed,
        atom_color=atom_color,
        bond_color=bond_color,
    )


def expand_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[StereoValue],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> nx.Graph:
    """Expand a complete configured stereograph."""
    from .configured import expand_configured_stereograph

    return expand_configured_stereograph(
        base_graph,
        tuple(descriptors),
        atom_color=atom_color,
        bond_color=bond_color,
    )


def _require_tetrahedral(
    descriptors: tuple[StereoValue, ...],
) -> tuple[TetrahedralStereo, ...]:
    if any(not isinstance(item, TetrahedralStereo) for item in descriptors):
        raise TypeError(
            "This tetrahedral compatibility API accepts only "
            "TetrahedralStereo descriptors."
        )
    return descriptors  # type: ignore[return-value]


def _expand_base_graph(
    base_graph: nx.Graph,
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

    endpoints = []
    for index, (left, right, attributes) in enumerate(base.edges(data=True)):
        vertex = BondVertex(index)
        endpoints.append((left, right))
        _add_coloured_node(
            auxiliary,
            vertex,
            ("bond_resource", _selected(attributes, bond_color)),
        )
        auxiliary.add_edge(atom_vertices[left], vertex)
        auxiliary.add_edge(vertex, atom_vertices[right])
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
    """Return exact stereograph identity for configured tetrahedral stereo."""
    fixed = _require_tetrahedral(tuple(descriptors))
    return canonicalize_stereograph(
        base_graph,
        fixed,
        atom_color=atom_color,
        bond_color=bond_color,
    )


def canonicalize_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[StereoValue],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
    enumerate_automorphism_group: bool = True,
) -> CanonicalStereographResult:
    """Return exact identity for a complete configured stereograph."""
    from .configured import canonicalize_configured_stereograph

    return canonicalize_configured_stereograph(
        base_graph,
        tuple(descriptors),
        atom_color=atom_color,
        bond_color=bond_color,
        enumerate_automorphism_group=enumerate_automorphism_group,
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
    enumerate_automorphism_group: bool = True,
) -> CanonicalStereographResult:
    """Canonicalize a registry across the complete configured catalogue."""
    from .configured import canonicalize_configured_registry

    return canonicalize_configured_registry(
        base_graph,
        registry,
        atom_color=atom_color,
        bond_color=bond_color,
        enumerate_automorphism_group=enumerate_automorphism_group,
    )


def mirror_stereo_descriptor(
    descriptor: StereoValue,
) -> StereoValue:
    """Apply the geometry-specific spatial-reflection action."""
    from .configured import mirror_configured_descriptor

    return mirror_configured_descriptor(descriptor)  # type: ignore[arg-type]


def mirror_stereo_registry(
    registry: Mapping[str, StereoValue],
) -> dict[str, StereoValue]:
    """Return the geometry-specific mirror without changing loci."""
    return {
        identifier: mirror_stereo_descriptor(descriptor)
        for identifier, descriptor in registry.items()
    }


def classify_stereograph_mirror(
    base_graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> StereographMirrorResult:
    """Compare a complete configured stereograph with its exact mirror.

    ``incomplete_loci`` and ``unsupported_loci`` let a perception boundary
    declare missing configured information without inventing descriptors.
    Unsupported evidence takes precedence over incomplete evidence.
    """
    from .configured import classify_configured_stereograph_mirror

    return classify_configured_stereograph_mirror(
        base_graph,
        registry,
        incomplete_loci=incomplete_loci,
        unsupported_loci=unsupported_loci,
        atom_color=atom_color,
        bond_color=bond_color,
    )


def _rdkit_graph(molecule: Any) -> nx.Graph:
    """Return a defensive attributed graph in the RDKit atom namespace."""
    from rdkit import Chem

    from synkit.IO.mol_to_graph import MolToGraph

    if not isinstance(molecule, Chem.Mol):
        raise TypeError("RDKit stereograph canonicalization requires Chem.Mol.")
    working = Chem.Mol(molecule)
    atom_maps = tuple(int(atom.GetAtomMapNum()) for atom in working.GetAtoms())
    fully_mapped = all(value > 0 for value in atom_maps) and len(set(atom_maps)) == len(
        atom_maps
    )
    return MolToGraph(include_stereo_descriptors=False).transform(
        working,
        use_index_as_atom_map=fully_mapped,
    )


def _rdkit_graph_and_registry(
    molecule: Any,
) -> tuple[nx.Graph, Mapping[str, StereoValue]]:
    """Return one RDKit graph/descriptor identifier namespace."""
    from rdkit import Chem

    from .rdkit_adapter import descriptors_from_rdkit

    graph = _rdkit_graph(molecule)
    working = Chem.Mol(molecule)
    atom_maps = tuple(int(atom.GetAtomMapNum()) for atom in working.GetAtoms())
    fully_mapped = all(value > 0 for value in atom_maps) and len(set(atom_maps)) == len(
        atom_maps
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
    """Canonicalize RDKit constitution plus all supported configured stereo."""
    from .configured import canonicalize_rdkit_configured_stereograph

    return canonicalize_rdkit_configured_stereograph(molecule)


def classify_rdkit_stereograph_mirror(
    molecule: Any,
    *,
    require_complete: bool = True,
    identity_profile: Literal[
        "chemical",
        "lewis_state",
        "acs_topology",
    ] = "chemical",
) -> StereographMirrorResult:
    """Classify configured RDKit input through exact molecular mirror identity.

    The chemical default compares the exact resonance family while preserving
    genuine bond-order distinctions. ``lewis_state`` audits one supplied
    Lewis drawing; ``acs_topology`` is connectivity-only benchmark
    compatibility.
    """
    from .configured import classify_rdkit_configured_stereograph_mirror

    return classify_rdkit_configured_stereograph_mirror(
        molecule,
        require_complete=require_complete,
        identity_profile=identity_profile,
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
