"""Evidence boundary for coupled whole-framework stereochemistry."""

from __future__ import annotations

from rdkit import Chem
import networkx as nx

from synkit.Graph.Stereo.configured import (
    classify_configured_stereograph_mirror,
)
from synkit.Graph.Stereo.global_stereo import (
    FrameworkFrame,
    FrameworkStereo,
    GlobalStereoCertificate,
    GlobalStereoInformationState,
)
from synkit.Graph.Stereo.canonical import StereographMirrorStatus
from synkit.IO.mol_to_graph import MolToGraph

from .chirality import _complete_tetrahedral_topology, _indexed_copy

MAX_FRAMEWORK_SUPPORT_ATOMS = 256
MAX_COUPLED_FRAMES = 64


def _topology_framework(
    molecule: Chem.Mol,
) -> tuple[object, FrameworkStereo | None, tuple[int, ...], str | None]:
    working = _indexed_copy(molecule)
    graph = MolToGraph(attr_profile="minimal").transform(
        working,
        use_index_as_atom_map=True,
    )
    registry = dict(graph.graph.get("stereo_descriptors", {}))
    completed = _complete_tetrahedral_topology(working, registry)
    frames = tuple(
        FrameworkFrame(
            descriptor.center,
            tuple(descriptor.atoms[1:]),
            int(descriptor.parity),
        )
        for descriptor in registry.values()
        if descriptor.descriptor_class == "tetrahedral"
        and descriptor.center in completed
    )
    if not frames:
        return graph, None, completed, None
    components = tuple(nx.connected_components(graph))
    frame_components = {
        index
        for index, component in enumerate(components)
        if any(frame.center in component for frame in frames)
    }
    if len(frame_components) != 1:
        return (
            graph,
            None,
            completed,
            "Coupled frames span more than one disconnected component.",
        )
    support = frozenset(components[next(iter(frame_components))])
    selected_frames = tuple(frame for frame in frames if frame.center in support)
    if len(support) > MAX_FRAMEWORK_SUPPORT_ATOMS:
        return (
            graph,
            None,
            completed,
            "Framework support exceeds the 256-atom safety limit.",
        )
    if len(selected_frames) > MAX_COUPLED_FRAMES:
        return (
            graph,
            None,
            completed,
            "Coupled framework exceeds the 64-frame safety limit.",
        )
    descriptor = FrameworkStereo(
        support,
        selected_frames,
        None,
        "molecular_chirality:coupled_topology_certificate",
    )
    return graph, descriptor, completed, None


def analyze_global_stereo_support(
    molecule: Chem.Mol,
) -> GlobalStereoCertificate:
    """Return a map-independent coupled-frame topology certificate.

    The returned descriptor is orientation-unspecified.  Topology may prove
    that every compatible configured member is chiral, but it never selects
    the positive or negative member of that mirror pair.
    """
    if molecule is None:
        return GlobalStereoCertificate(
            None,
            GlobalStereoInformationState.UNSUPPORTED,
            False,
            "exact_framework_stereograph_mirror_comparison",
            unsupported_reason="A molecule is required.",
        )
    graph, descriptor, _completed, unsupported_reason = _topology_framework(molecule)
    if unsupported_reason is not None:
        return GlobalStereoCertificate(
            None,
            GlobalStereoInformationState.UNSUPPORTED,
            False,
            "exact_framework_stereograph_mirror_comparison",
            unsupported_reason=unsupported_reason,
        )
    if descriptor is None:
        return GlobalStereoCertificate(
            None,
            GlobalStereoInformationState.POTENTIAL,
            False,
            "exact_framework_stereograph_mirror_comparison",
        )
    positive = FrameworkStereo(
        descriptor.support_atoms,
        descriptor.frames,
        1,
        descriptor.provenance,
    )
    result = classify_configured_stereograph_mirror(
        graph,  # type: ignore[arg-type]
        {"global:framework": positive},
        atom_color=("element", "isotope", "hcount"),
        bond_color=lambda _attributes: "bond",
    )
    necessarily_chiral = result.status is StereographMirrorStatus.CHIRAL
    return GlobalStereoCertificate(
        descriptor,
        (
            GlobalStereoInformationState.NECESSARILY_CHIRAL
            if necessarily_chiral
            else GlobalStereoInformationState.POTENTIAL
        ),
        necessarily_chiral,
        "exact_framework_stereograph_mirror_comparison:acs_topology",
        None if result.original is None else result.original.canonical_digest,
        None if result.mirror is None else result.mirror.canonical_digest,
        result.atom_mirror_isomorphism,
    )


def configured_framework_from_certificate(
    certificate: GlobalStereoCertificate,
    orientation: int,
    *,
    provenance: str,
) -> FrameworkStereo:
    """Select a global orientation only from explicit authorized evidence."""
    if orientation not in {-1, 1}:
        raise ValueError("Configured framework orientation must be -1 or 1.")
    if not provenance.strip():
        raise ValueError("Configured framework orientation requires provenance.")
    if certificate.descriptor is None:
        raise ValueError("The certificate has no coupled framework support.")
    if not certificate.necessarily_chiral:
        raise ValueError("The certificate does not establish a global mirror pair.")
    return FrameworkStereo(
        certificate.descriptor.support_atoms,
        certificate.descriptor.frames,
        orientation,
        provenance,
    )


__all__ = [
    "MAX_COUPLED_FRAMES",
    "MAX_FRAMEWORK_SUPPORT_ATOMS",
    "analyze_global_stereo_support",
    "configured_framework_from_certificate",
]
