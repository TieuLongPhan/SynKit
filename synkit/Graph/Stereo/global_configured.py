"""Auxiliary-graph expansion for configured coupled frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import networkx as nx

from .canonical import _add_coloured_node, _material_port_resource
from .descriptors import TetrahedralStereo, parse_virtual_reference
from .global_stereo import FrameworkStereo
from .orbits import SHAPE_DEFINITIONS, StereoSpecification


@dataclass(frozen=True)
class _FrameworkSupportVertex:
    locus: int
    atom: int


@dataclass(frozen=True)
class _FrameworkFrameVertex:
    locus: int
    frame: int


@dataclass(frozen=True)
class _FrameworkRepresentationVertex:
    locus: int
    frame: int
    representation: int


@dataclass(frozen=True)
class _FrameworkSlotVertex:
    locus: int
    frame: int
    representation: int
    slot: int


@dataclass(frozen=True)
class _FrameworkVirtualVertex:
    locus: int
    frame: int
    representation: int
    slot: int
    kind: str


def add_framework(prepared: Any, descriptor: FrameworkStereo, index: int) -> None:
    """Expand one coupled global orientation without local-parity promotion."""
    from .configured import _locus

    if descriptor.specification is not StereoSpecification.FIXED:
        raise ValueError("Configured framework orientation must be fixed.")
    absent = descriptor.support_atoms - prepared.base.nodes
    if absent:
        raise ValueError(
            "Framework support contains absent atoms: "
            + ", ".join(map(str, sorted(absent)))
        )
    if not nx.is_connected(prepared.base.subgraph(descriptor.support_atoms)):
        raise ValueError("Framework stereo support must be connected.")

    locus = _locus(prepared, index, descriptor.descriptor_class)
    for atom in sorted(descriptor.support_atoms):
        incidence = _FrameworkSupportVertex(index, atom)
        _add_coloured_node(
            prepared.expansion.graph,
            incidence,
            ("framework_support",),
        )
        prepared.expansion.graph.add_edge(locus, incidence)
        prepared.expansion.graph.add_edge(incidence, prepared.atoms[atom])

    definition = SHAPE_DEFINITIONS["tetrahedral"]
    for frame_index, frame in enumerate(descriptor.frames):
        frame_vertex = _FrameworkFrameVertex(index, frame_index)
        _add_coloured_node(
            prepared.expansion.graph,
            frame_vertex,
            ("framework_frame",),
        )
        prepared.expansion.graph.add_edge(locus, frame_vertex)
        prepared.expansion.graph.add_edge(frame_vertex, prepared.atoms[frame.center])
        configured = TetrahedralStereo(
            (frame.center, *frame.references),
            frame.relation * descriptor.orientation,  # type: ignore[operator]
        )
        orbit = definition.preserving_group.orbit(configured.configuration.frame)
        for representation_index, representation in enumerate(orbit):
            representation_vertex = _FrameworkRepresentationVertex(
                index,
                frame_index,
                representation_index,
            )
            _add_coloured_node(
                prepared.expansion.graph,
                representation_vertex,
                ("framework_representation",),
            )
            prepared.expansion.graph.add_edge(frame_vertex, representation_vertex)
            for slot_index, reference in enumerate(representation[1:]):
                slot = _FrameworkSlotVertex(
                    index,
                    frame_index,
                    representation_index,
                    slot_index,
                )
                _add_coloured_node(
                    prepared.expansion.graph,
                    slot,
                    ("framework_slot", slot_index),
                )
                prepared.expansion.graph.add_edge(representation_vertex, slot)
                virtual = parse_virtual_reference(reference)
                if virtual is None:
                    resource: Hashable = _material_port_resource(
                        prepared.base,
                        prepared.bonds,
                        frame.center,
                        reference,
                        geometry=descriptor.descriptor_class,
                    )
                else:
                    if virtual.center != frame.center:
                        raise ValueError(
                            f"Virtual ligand {reference!r} does not belong to "
                            f"framework center {frame.center!r}."
                        )
                    resource = _FrameworkVirtualVertex(
                        index,
                        frame_index,
                        representation_index,
                        slot_index,
                        virtual.kind,
                    )
                    _add_coloured_node(
                        prepared.expansion.graph,
                        resource,
                        ("virtual_resource", virtual.kind),
                    )
                    prepared.expansion.graph.add_edge(
                        prepared.atoms[frame.center],
                        resource,
                    )
                prepared.expansion.graph.add_edge(slot, resource)


__all__ = ["add_framework"]
