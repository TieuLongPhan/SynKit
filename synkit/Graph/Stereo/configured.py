"""Exact canonicalization for the complete configured stereo catalogue."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
from typing import Any, Hashable, Literal

import networkx as nx

from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

from .canonical import (
    AtomVertex,
    AttributeSelector,
    BondVertex,
    CanonicalStereographResult,
    StereoLocusVertex,
    StereoPortVertex,
    StereoSlotVertex,
    StereoTupleVertex,
    STEREOGRAPH_SCHEMA,
    StereographMirrorResult,
    StereographMirrorStatus,
    VirtualResourceVertex,
    _DEFAULT_ATOM_COLOR_KEYS,
    _DEFAULT_BOND_COLOR_KEYS,
    _Expansion,
    _add_coloured_node,
    _add_configuration_orbit,
    _expand_base_graph,
    _material_port_resource,
    _project_witnesses,
    _rdkit_graph,
    _rdkit_graph_and_registry,
)
from .descriptors import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    OctahedralStereo,
    PlanarBondStereo,
    Reference,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    parse_virtual_reference,
)
from .extended_descriptors import HelicalStereo, PlanarChiralityStereo
from .global_configured import add_framework as _add_framework
from .global_stereo import FrameworkStereo
from .orbits import StereoSpecification

# Compatibility alias for callers that adopted the former configured-prefixed
# API.  SynKit now exposes one canonical stereograph model.
CONFIGURED_STEREOGRAPH_SCHEMA = STEREOGRAPH_SCHEMA

AtomDescriptor = (
    TetrahedralStereo
    | SquarePlanarStereo
    | TrigonalBipyramidalStereo
    | OctahedralStereo
)
BondDescriptor = PlanarBondStereo | AtropBondStereo
ConfiguredDescriptor = StereoValue
CONFIGURED_DESCRIPTOR_TYPES = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
    PlanarBondStereo,
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    PlanarChiralityStereo,
    FrameworkStereo,
)

MirrorIdentityProfile = Literal["chemical", "lewis_state", "acs_topology"]
_ACS_TOPOLOGY_ATOM_COLOR = ("element", "isotope", "hcount")
_MAX_RESONANCE_FORMS = 4096


def _connectivity_bond_color(_attributes: Mapping[str, Any]) -> str:
    """Collapse bond placement for the explicit ACS topology profile."""
    return "bond"


def _mirror_profile_options(
    identity_profile: MirrorIdentityProfile,
) -> dict[str, AttributeSelector]:
    if identity_profile == "acs_topology":
        return {
            "atom_color": _ACS_TOPOLOGY_ATOM_COLOR,
            "bond_color": _connectivity_bond_color,
        }
    if identity_profile in {"chemical", "lewis_state"}:
        return {}
    raise ValueError(
        "Mirror identity profile must be 'chemical', 'lewis_state', "
        "or 'acs_topology'."
    )


def _record_mirror_profile(
    result: StereographMirrorResult,
    identity_profile: MirrorIdentityProfile,
) -> StereographMirrorResult:
    return replace(
        result,
        method=f"{result.method}:{identity_profile}",
    )


@dataclass(frozen=True)
class _Prepared:
    expansion: _Expansion
    base: nx.Graph
    atoms: Mapping[Hashable, AtomVertex]
    bonds: Mapping[tuple[Hashable, Hashable], BondVertex]


def _prepare(
    base_graph: nx.Graph,
    atom_color: AttributeSelector,
    bond_color: AttributeSelector,
) -> _Prepared:
    base = base_graph.copy()
    expansion = _expand_base_graph(
        base,
        atom_color=atom_color,
        bond_color=bond_color,
    )
    atoms = {
        node.reference: node for node in expansion.graph if isinstance(node, AtomVertex)
    }
    bonds: dict[tuple[Hashable, Hashable], BondVertex] = {}
    for node in expansion.graph:
        if not isinstance(node, BondVertex):
            continue
        left, right = expansion.bond_endpoints[node.index]
        bonds[(left, right)] = node
        bonds[(right, left)] = node
    return _Prepared(expansion, base, atoms, bonds)


def _port(
    prepared: _Prepared,
    locus: StereoLocusVertex,
    locus_index: int,
    port_position: int,
    owner: Hashable,
    reference: Reference,
    geometry: str,
) -> StereoPortVertex:
    port = StereoPortVertex(locus_index, port_position)
    _add_coloured_node(
        prepared.expansion.graph,
        port,
        ("stereo_port", geometry),
    )
    prepared.expansion.graph.add_edge(locus, port)
    virtual = parse_virtual_reference(reference)
    if virtual is None:
        resource: Hashable = _material_port_resource(
            prepared.base,
            prepared.bonds,
            owner,
            reference,
            geometry=geometry,
        )
    else:
        if virtual.center != owner:
            raise ValueError(
                f"Virtual ligand {reference!r} does not belong to {owner!r}."
            )
        resource = VirtualResourceVertex(
            locus_index,
            port_position,
            virtual.kind,
        )
        _add_coloured_node(
            prepared.expansion.graph,
            resource,
            ("virtual_resource", virtual.kind),
        )
        prepared.expansion.graph.add_edge(prepared.atoms[owner], resource)
    prepared.expansion.graph.add_edge(port, resource)
    return port


def _locus(
    prepared: _Prepared,
    index: int,
    geometry: str,
) -> StereoLocusVertex:
    locus = StereoLocusVertex(index)
    _add_coloured_node(
        prepared.expansion.graph,
        locus,
        ("stereo_locus", geometry),
    )
    return locus


def _add_atom_descriptor(
    prepared: _Prepared,
    descriptor: AtomDescriptor,
    index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError(f"Configured {descriptor.descriptor_class} must be fixed.")
    center = descriptor.center
    if center not in prepared.atoms:
        raise ValueError(f"Stereo center {center!r} is absent.")
    locus = _locus(prepared, index, descriptor.descriptor_class)
    prepared.expansion.graph.add_edge(locus, prepared.atoms[center])
    targets: dict[Reference, Hashable] = {center: prepared.atoms[center]}
    for position, reference in enumerate(descriptor.atoms[1:]):
        targets[reference] = _port(
            prepared,
            locus,
            index,
            position,
            center,
            reference,
            descriptor.descriptor_class,
        )
    _add_configuration_orbit(
        prepared.expansion.graph,
        locus,
        descriptor,  # type: ignore[arg-type]
        index,
        targets,
        tuple(range(1, len(descriptor.atoms))),
    )


def _add_bond_descriptor(
    prepared: _Prepared,
    descriptor: BondDescriptor,
    index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError(f"Configured {descriptor.descriptor_class} must be fixed.")
    left, right = descriptor.atoms[2:4]
    if (left, right) not in prepared.bonds:
        raise ValueError(f"Stereo bond {left!r}-{right!r} is absent.")
    locus = _locus(prepared, index, descriptor.descriptor_class)
    prepared.expansion.graph.add_edge(locus, prepared.bonds[(left, right)])
    targets: dict[Reference, Hashable] = {
        left: prepared.atoms[left],
        right: prepared.atoms[right],
    }
    owners = (left, left, right, right)
    for port_position, (frame_position, owner) in enumerate(zip((0, 1, 4, 5), owners)):
        reference = descriptor.atoms[frame_position]
        targets[reference] = _port(
            prepared,
            locus,
            index,
            port_position,
            owner,
            reference,
            descriptor.descriptor_class,
        )
    _add_configuration_orbit(
        prepared.expansion.graph,
        locus,
        descriptor,
        index,
        targets,
        tuple(range(6)),
    )


def _validate_path(
    prepared: _Prepared,
    path: Sequence[int],
    *,
    cyclic: bool = False,
) -> tuple[BondVertex, ...]:
    if any(atom not in prepared.atoms for atom in path):
        raise ValueError("Configured stereo path contains an absent atom.")
    pairs = list(zip(path, path[1:]))
    if cyclic:
        pairs.append((path[-1], path[0]))
    try:
        return tuple(prepared.bonds[pair] for pair in pairs)
    except KeyError as exc:
        raise ValueError("Configured stereo path is not continuous.") from exc


def _add_cumulene_path(
    prepared: _Prepared,
    descriptor: CumuleneAxisStereo | ExtendedCisTransStereo,
    index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError("Configured cumulene path must be fixed.")
    path = (
        descriptor.axis_path
        if isinstance(descriptor, CumuleneAxisStereo)
        else descriptor.path
    )
    path_bonds = _validate_path(prepared, path)
    locus = _locus(prepared, index, descriptor.descriptor_class)
    for bond in path_bonds:
        prepared.expansion.graph.add_edge(locus, bond)
    left, right = path[0], path[-1]
    targets: dict[Reference, Hashable] = {
        left: prepared.atoms[left],
        right: prepared.atoms[right],
    }
    references = (*descriptor.terminal_frames[0], *descriptor.terminal_frames[1])
    for position, (owner, reference) in enumerate(
        zip((left, left, right, right), references)
    ):
        targets[reference] = _port(
            prepared,
            locus,
            index,
            position,
            owner,
            reference,
            descriptor.descriptor_class,
        )
    _add_configuration_orbit(
        prepared.expansion.graph,
        locus,
        descriptor,  # type: ignore[arg-type]
        index,
        targets,
        tuple(range(6)),
    )


def _add_frames(
    graph: nx.Graph,
    locus: StereoLocusVertex,
    index: int,
    geometry: str,
    frames: Sequence[Sequence[int]],
    atoms: Mapping[Hashable, AtomVertex],
) -> None:
    for tuple_index, frame in enumerate(frames):
        relation = StereoTupleVertex(index, tuple_index)
        _add_coloured_node(graph, relation, ("stereo_tuple", geometry))
        graph.add_edge(locus, relation)
        for position, reference in enumerate(frame):
            slot = StereoSlotVertex(index, tuple_index, position)
            _add_coloured_node(
                graph,
                slot,
                ("stereo_slot", geometry, position),
            )
            graph.add_edge(relation, slot)
            graph.add_edge(slot, atoms[reference])


def _path_variants(
    path: tuple[int, ...],
    cyclic: bool,
) -> tuple[tuple[int, ...], ...]:
    if not cyclic:
        return path, tuple(reversed(path))
    reverse = tuple(reversed(path))
    return tuple(
        sequence[offset:] + sequence[:offset]
        for sequence in (path, reverse)
        for offset in range(len(path))
    )


def _add_helical(
    prepared: _Prepared,
    descriptor: HelicalStereo,
    index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError("Configured helical path must be fixed.")
    _validate_path(prepared, descriptor.path, cyclic=descriptor.cyclic)
    locus = StereoLocusVertex(index)
    _add_coloured_node(
        prepared.expansion.graph,
        locus,
        (
            "stereo_locus",
            descriptor.descriptor_class,
            descriptor.parity,
            descriptor.cyclic,
            descriptor.coupling_id,
        ),
    )
    _add_frames(
        prepared.expansion.graph,
        locus,
        index,
        descriptor.descriptor_class,
        _path_variants(descriptor.path, descriptor.cyclic),
        prepared.atoms,
    )


def _add_planar_chirality(
    prepared: _Prepared,
    descriptor: PlanarChiralityStereo,
    index: int,
) -> None:
    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError("Configured planar chirality must be fixed.")
    _validate_path(prepared, descriptor.plane_atoms, cyclic=True)
    if descriptor.pilot not in prepared.atoms:
        raise ValueError("Planar-chirality pilot atom is absent.")
    locus = _locus(prepared, index, descriptor.descriptor_class)
    plane = (
        descriptor.plane_atoms
        if descriptor.parity == 1
        else tuple(reversed(descriptor.plane_atoms))
    )
    frames = tuple(
        (descriptor.pilot, *(plane[offset:] + plane[:offset]))
        for offset in range(len(plane))
    )
    _add_frames(
        prepared.expansion.graph,
        locus,
        index,
        descriptor.descriptor_class,
        frames,
        prepared.atoms,
    )


def _locus_key(descriptor: ConfiguredDescriptor) -> tuple[str, Hashable]:
    if isinstance(descriptor, FrameworkStereo):
        return "global", descriptor.support_atoms
    if isinstance(
        descriptor,
        (
            TetrahedralStereo,
            SquarePlanarStereo,
            TrigonalBipyramidalStereo,
            OctahedralStereo,
        ),
    ):
        return "atom", descriptor.center
    if isinstance(descriptor, (PlanarBondStereo, AtropBondStereo)):
        return "bond", frozenset(descriptor.atoms[2:4])
    if isinstance(descriptor, (CumuleneAxisStereo, ExtendedCisTransStereo)):
        path = (
            descriptor.axis_path
            if isinstance(descriptor, CumuleneAxisStereo)
            else descriptor.path
        )
        return descriptor.descriptor_class, min(
            path,
            tuple(reversed(path)),
        )
    if isinstance(descriptor, HelicalStereo):
        return "path", (descriptor.canonical_path, descriptor.coupling_id)
    return "plane", (descriptor.canonical_plane, descriptor.pilot)


def _expanded(
    base_graph: nx.Graph,
    descriptors: tuple[ConfiguredDescriptor, ...],
    *,
    atom_color: AttributeSelector,
    bond_color: AttributeSelector,
) -> _Expansion:
    prepared = _prepare(base_graph, atom_color, bond_color)
    seen: set[tuple[str, Hashable]] = set()
    global_supports: list[frozenset[int]] = []
    for index, descriptor in enumerate(descriptors):
        key = _locus_key(descriptor)
        if key in seen:
            raise ValueError(f"Duplicate configured stereo locus at {key[1]!r}.")
        seen.add(key)
        if isinstance(descriptor, FrameworkStereo):
            if any(descriptor.support_atoms & support for support in global_supports):
                raise ValueError("Overlapping framework stereo supports are invalid.")
            global_supports.append(descriptor.support_atoms)
            _add_framework(prepared, descriptor, index)
        elif isinstance(
            descriptor,
            (
                TetrahedralStereo,
                SquarePlanarStereo,
                TrigonalBipyramidalStereo,
                OctahedralStereo,
            ),
        ):
            _add_atom_descriptor(prepared, descriptor, index)
        elif isinstance(descriptor, (PlanarBondStereo, AtropBondStereo)):
            _add_bond_descriptor(prepared, descriptor, index)
        elif isinstance(
            descriptor,
            (CumuleneAxisStereo, ExtendedCisTransStereo),
        ):
            _add_cumulene_path(prepared, descriptor, index)
        elif isinstance(descriptor, HelicalStereo):
            _add_helical(prepared, descriptor, index)
        else:
            _add_planar_chirality(prepared, descriptor, index)
    return prepared.expansion


def expand_configured_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[ConfiguredDescriptor],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> nx.Graph:
    return _expanded(
        base_graph,
        tuple(descriptors),
        atom_color=atom_color,
        bond_color=bond_color,
    ).graph


def canonicalize_configured_stereograph(
    base_graph: nx.Graph,
    descriptors: Iterable[ConfiguredDescriptor],
    *,
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
    enumerate_automorphism_group: bool = True,
) -> CanonicalStereographResult:
    expansion = _expanded(
        base_graph,
        tuple(descriptors),
        atom_color=atom_color,
        bond_color=bond_color,
    )
    auxiliary = ExactColoredGraphCanonicalizer(
        expansion.graph,
        node_color="color",
        edge_color=None,
        prune_automorphisms=True,
        enumerate_automorphism_group=enumerate_automorphism_group,
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
    code = f"{STEREOGRAPH_SCHEMA}\n{auxiliary.canonical_code}"
    return CanonicalStereographResult(
        schema=STEREOGRAPH_SCHEMA,
        canonical_code=code,
        canonical_digest=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        atom_order=atom_order,
        bond_order=bond_order,
        locus_order=locus_order,
        port_order=port_order,
        atom_automorphisms=_project_witnesses(auxiliary, AtomVertex),
        locus_automorphisms=_project_witnesses(
            auxiliary,
            StereoLocusVertex,
        ),
        port_automorphisms=_project_witnesses(auxiliary, StereoPortVertex),
        auxiliary=auxiliary,
    )


def canonicalize_configured_registry(
    base_graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    **options: Any,
) -> CanonicalStereographResult:
    unsupported = tuple(
        descriptor.descriptor_class
        for descriptor in registry.values()
        if not isinstance(descriptor, CONFIGURED_DESCRIPTOR_TYPES)
    )
    if unsupported:
        raise TypeError(
            "Configured canonicalization cannot omit families: "
            + ", ".join(sorted(set(unsupported)))
        )
    return canonicalize_configured_stereograph(
        base_graph,
        registry.values(),  # type: ignore[arg-type]
        **options,
    )


def mirror_configured_descriptor(
    descriptor: ConfiguredDescriptor,
) -> ConfiguredDescriptor:
    if isinstance(descriptor, FrameworkStereo):
        return descriptor.invert()
    if isinstance(descriptor, TetrahedralStereo):
        return descriptor.opposite()
    if isinstance(
        descriptor,
        (
            SquarePlanarStereo,
            PlanarBondStereo,
            ExtendedCisTransStereo,
        ),
    ):
        return descriptor
    if isinstance(
        descriptor,
        (
            TrigonalBipyramidalStereo,
            OctahedralStereo,
            AtropBondStereo,
            CumuleneAxisStereo,
            HelicalStereo,
            PlanarChiralityStereo,
        ),
    ):
        return descriptor.invert()
    raise TypeError(f"Unsupported configured mirror family: {type(descriptor)!r}.")


def _configured_mirror_boundary(
    registry: Mapping[str, StereoValue],
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
) -> StereographMirrorResult | None:
    """Return a fail-closed boundary result, or ``None`` when definitive."""
    unsupported_entries = tuple(
        (key, value.descriptor_class)
        for key, value in registry.items()
        if not isinstance(value, CONFIGURED_DESCRIPTOR_TYPES)
    )
    declared_unsupported = tuple(unsupported_loci)
    if unsupported_entries or declared_unsupported:
        return StereographMirrorResult(
            StereographMirrorStatus.UNSUPPORTED,
            len(registry),
            unsupported_loci=tuple(
                sorted(
                    {
                        *declared_unsupported,
                        *(key for key, _family in unsupported_entries),
                    }
                )
            ),
            unsupported_families=tuple(
                sorted({family for _key, family in unsupported_entries})
            ),
            method="exact_canonical_stereograph_mirror_comparison",
        )
    unknown = tuple(
        key
        for key, descriptor in registry.items()
        if descriptor.specification is StereoSpecification.UNSPECIFIED
    )
    missing = tuple(sorted({*incomplete_loci, *unknown}))
    if missing:
        return StereographMirrorResult(
            StereographMirrorStatus.INCOMPLETE,
            len(registry),
            incomplete_loci=missing,
            method="exact_canonical_stereograph_mirror_comparison",
        )
    return None


def classify_configured_stereograph_mirror(
    base_graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> StereographMirrorResult:
    boundary = _configured_mirror_boundary(
        registry,
        incomplete_loci=incomplete_loci,
        unsupported_loci=unsupported_loci,
    )
    if boundary is not None:
        return boundary
    original = canonicalize_configured_registry(
        base_graph,
        registry,
        atom_color=atom_color,
        bond_color=bond_color,
    )
    mirrored = {
        key: mirror_configured_descriptor(value)  # type: ignore[arg-type]
        for key, value in registry.items()
    }
    mirror = canonicalize_configured_registry(
        base_graph,
        mirrored,
        atom_color=atom_color,
        bond_color=bond_color,
    )
    achiral = original.same_stereograph(mirror)
    return StereographMirrorResult(
        (
            StereographMirrorStatus.ACHIRAL
            if achiral
            else StereographMirrorStatus.CHIRAL
        ),
        len(registry),
        original,
        mirror,
        tuple(mirrored.values()),
        (tuple(zip(original.atom_order, mirror.atom_order)) if achiral else None),
        method="exact_canonical_stereograph_mirror_comparison",
    )


def _resonance_graphs(molecule: Any) -> tuple[nx.Graph, ...]:
    """Return every RDKit resonance form as an attributed molecular graph."""
    from rdkit import Chem

    supplier = Chem.ResonanceMolSupplier(
        molecule,
        flags=Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION,
        maxStructs=_MAX_RESONANCE_FORMS,
    )
    forms = tuple(supplier)
    if len(forms) >= _MAX_RESONANCE_FORMS:
        raise RuntimeError(
            "Exact resonance-family enumeration reached the "
            f"{_MAX_RESONANCE_FORMS}-form safety limit."
        )
    if not forms:
        forms = (Chem.Mol(molecule),)
    graphs = []
    for form in forms:
        normalized = Chem.Mol(form)
        Chem.SetAromaticity(normalized)
        graphs.append(_rdkit_graph(normalized))
    return tuple(graphs)


def _classify_resonance_family_mirror(
    molecule: Any,
    registry: Mapping[str, StereoValue],
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
) -> StereographMirrorResult:
    """Compare the complete Lewis-resonance family with its mirror exactly."""
    boundary = _configured_mirror_boundary(
        registry,
        incomplete_loci=incomplete_loci,
        unsupported_loci=unsupported_loci,
    )
    if boundary is not None:
        return boundary

    mirrored = {
        key: mirror_configured_descriptor(value)  # type: ignore[arg-type]
        for key, value in registry.items()
    }
    originals = []
    mirrors = []
    for graph in _resonance_graphs(molecule):
        originals.append(canonicalize_configured_registry(graph, registry))
        mirrors.append(canonicalize_configured_registry(graph, mirrored))
    originals.sort(key=lambda result: result.canonical_code)
    mirrors.sort(key=lambda result: result.canonical_code)
    mirror_by_code = {result.canonical_code: result for result in mirrors}
    matched = next(
        (
            (original, mirror_by_code[original.canonical_code])
            for original in originals
            if original.canonical_code in mirror_by_code
        ),
        None,
    )
    achiral = matched is not None
    original, mirror = matched or (originals[0], mirrors[0])
    return StereographMirrorResult(
        (
            StereographMirrorStatus.ACHIRAL
            if achiral
            else StereographMirrorStatus.CHIRAL
        ),
        len(registry),
        original,
        mirror,
        tuple(mirrored.values()),
        (tuple(zip(original.atom_order, mirror.atom_order)) if achiral else None),
        method="exact_resonance_family_stereograph_mirror_comparison",
    )


def canonicalize_rdkit_configured_stereograph(
    molecule: Any,
) -> CanonicalStereographResult:
    graph, registry = _rdkit_graph_and_registry(molecule)
    return canonicalize_configured_registry(graph, registry)


def _is_terminal_phosphate_resonance_locus(molecule: Any, element: Any) -> bool:
    """Return whether an apparent P center differs only by localized resonance.

    A neutral tetra-coordinate phosphorus with terminal ``P=O`` and
    ``P-[O-]`` ligands contains two drawings of the same delocalized phosphate
    oxygen environment.  Treating bond order and formal charge as ligand
    identity makes this look tetrahedral even though exchanging those terminal
    oxygens is resonance-equivalent in the molecular-identity task.
    """
    from rdkit import Chem

    support = element.support
    if not hasattr(support, "center"):
        return False
    atom = molecule.GetAtomWithIdx(support.center)
    if atom.GetAtomicNum() != 15 or atom.GetFormalCharge() != 0:
        return False
    terminal_oxygen_bonds = [
        (neighbor, molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()))
        for neighbor in atom.GetNeighbors()
        if neighbor.GetAtomicNum() == 8 and neighbor.GetDegree() == 1
    ]
    has_neutral_double = any(
        oxygen.GetFormalCharge() == 0
        and bond.GetBondType() is Chem.BondType.DOUBLE
        for oxygen, bond in terminal_oxygen_bonds
    )
    has_anionic_single = any(
        oxygen.GetFormalCharge() == -1
        and bond.GetBondType() is Chem.BondType.SINGLE
        for oxygen, bond in terminal_oxygen_bonds
    )
    return has_neutral_double and has_anionic_single


def _classify_rdkit_identity_profile(
    molecule: Any,
    graph: nx.Graph,
    registry: Mapping[str, StereoValue],
    identity_profile: MirrorIdentityProfile,
    *,
    incomplete_loci: Iterable[str] = (),
    unsupported_loci: Iterable[str] = (),
) -> StereographMirrorResult:
    if identity_profile == "chemical":
        result = _classify_resonance_family_mirror(
            molecule,
            registry,
            incomplete_loci=incomplete_loci,
            unsupported_loci=unsupported_loci,
        )
    else:
        result = classify_configured_stereograph_mirror(
            graph,
            registry,
            incomplete_loci=incomplete_loci,
            unsupported_loci=unsupported_loci,
            **_mirror_profile_options(identity_profile),
        )
    return _record_mirror_profile(result, identity_profile)


def classify_rdkit_configured_stereograph_mirror(
    molecule: Any,
    *,
    require_complete: bool = True,
    identity_profile: MirrorIdentityProfile = "chemical",
) -> StereographMirrorResult:
    """Classify an RDKit molecule with the exact stereograph mirror test.

    The ``chemical`` default compares the complete enumerated resonance family:
    genuine bond-order differences remain distinct, while alternative
    resonance/Kekule drawings do not create false chirality.
    ``lewis_state`` audits the precise supplied charge/bond-order drawing.
    ``acs_topology`` reproduces the external benchmark's connectivity-only
    identity convention and must not be reported as literal Lewis-structure
    chirality.

    Perception is used only to identify supported loci whose configuration is
    absent.  It does not supply a configuration or a handedness label.
    Set ``require_complete=False`` to classify exactly the source-declared
    configured stereograph while leaving every undeclared locus unconstrained.
    Enhanced stereo groups remain outside the single-stereoisomer contract.
    A chiral verdict on the supplied descriptors remains exact when additional
    loci are unresolved: adding stereo constraints can only remove candidate
    mirror isomorphisms, never create one.
    """
    from rdkit import Chem

    from synkit.Chem.Molecule.stereo_perception import (
        StereoConfigurationState,
        detect_potential_stereo_elements,
    )

    _mirror_profile_options(identity_profile)
    graph, registry = _rdkit_graph_and_registry(molecule)
    enhanced_groups = tuple(
        f"enhanced_stereo_group:{index}:{group.GetGroupType()}"
        for index, group in enumerate(molecule.GetStereoGroups())
    )
    if not require_complete:
        return _classify_rdkit_identity_profile(
            molecule,
            graph,
            registry,
            identity_profile,
            unsupported_loci=enhanced_groups,
        )
    configured_tetrahedral_centers = tuple(
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetChiralTag()
        in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
    )
    incomplete = tuple(
        element.identifier
        for element in detect_potential_stereo_elements(
            molecule,
            excluded_tetrahedral_centers=configured_tetrahedral_centers,
        )
        if (
            element.configuration_state is StereoConfigurationState.UNSPECIFIED
            and not _is_terminal_phosphate_resonance_locus(molecule, element)
        )
    )
    if incomplete and not enhanced_groups:
        relaxed = _classify_rdkit_identity_profile(
            molecule,
            graph,
            registry,
            identity_profile,
        )
        if relaxed.status is StereographMirrorStatus.CHIRAL:
            return replace(
                relaxed,
                incomplete_loci=tuple(sorted(set(incomplete))),
                method=(
                    "exact_stereograph_monotone_chiral_mirror_proof:"
                    f"{identity_profile}"
                ),
            )
    return _classify_rdkit_identity_profile(
        molecule,
        graph,
        registry,
        identity_profile,
        incomplete_loci=incomplete,
        unsupported_loci=enhanced_groups,
    )


__all__ = [
    "CONFIGURED_DESCRIPTOR_TYPES",
    "CONFIGURED_STEREOGRAPH_SCHEMA",
    "ConfiguredDescriptor",
    "MirrorIdentityProfile",
    "canonicalize_configured_registry",
    "canonicalize_configured_stereograph",
    "canonicalize_rdkit_configured_stereograph",
    "classify_configured_stereograph_mirror",
    "classify_rdkit_configured_stereograph_mirror",
    "expand_configured_stereograph",
    "mirror_configured_descriptor",
]
