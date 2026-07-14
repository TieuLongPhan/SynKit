"""Descriptor registry comparison for molecules, ITS graphs, and rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx

from .descriptors import (
    OctahedralStereo,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
)

_ATOM_STEREO_TYPES = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
)


def _peripheral_references(descriptor: StereoValue) -> tuple[Any, ...]:
    if isinstance(descriptor, _ATOM_STEREO_TYPES):
        return descriptor.atoms[1:]
    return (*descriptor.atoms[:2], *descriptor.atoms[4:])


def _align_single_reference_replacement(
    old: StereoValue,
    new: StereoValue,
) -> StereoValue | None:
    """Align one leaving/entering reference in the old descriptor frame."""
    if type(old) is not type(new) or descriptor_id(old) != descriptor_id(new):
        return None
    old_refs = _peripheral_references(old)
    new_refs = _peripheral_references(new)
    removed = list(set(old_refs) - set(new_refs))
    added = list(set(new_refs) - set(old_refs))
    if len(removed) != 1 or len(added) != 1:
        return None
    atoms = list(old.atoms)
    atoms[atoms.index(removed[0])] = added[0]
    return type(old)(tuple(atoms), old.parity, old.provenance)


@dataclass(frozen=True)
class StereoChange:
    """One rule-local stereochemical state transition.

    ``before`` and ``after`` are the stable molecular states. ``transition``
    is optional ITS/transition-state stereo and is deliberately separate from
    either endpoint.  The textual ``change`` remains a derived, readable
    classification for compatibility with existing SynKit rule consumers.
    """

    change: str
    before: StereoValue | None
    after: StereoValue | None
    transition: StereoValue | None = None

    @property
    def dependencies(self) -> frozenset[int]:
        values = set()
        if self.before:
            values.update(self.before.dependencies)
        if self.after:
            values.update(self.after.dependencies)
        if self.transition:
            values.update(self.transition.dependencies)
        return frozenset(values)

    @property
    def fleeting(self) -> StereoValue | None:
        """Compatibility term for stereo present only at the transition state."""
        return self.transition

    def reverse(self) -> "StereoChange":
        """Return the same state transition in the opposite direction."""
        return StereoChange(
            classify_stereo_change(self.after, self.before, self.transition),
            self.after,
            self.before,
            self.transition,
        )


def classify_stereo_change(
    old: StereoValue | None,
    new: StereoValue | None,
    transition: StereoValue | None = None,
) -> str:
    """Derive SynKit's chemical label from endpoint/transition descriptors."""
    if old is None and new is None:
        return "FLEETING" if transition is not None else "UNSPECIFIED"
    if old is None:
        return "FORMED"
    if new is None:
        return "BROKEN"
    if old == new:
        return "RETAINED"
    if type(old) is type(new) and old.invert() == new:
        return "INVERTED"
    aligned = _align_single_reference_replacement(old, new)
    if aligned is not None:
        if aligned == new:
            return "RETAINED"
        if aligned.invert() == new:
            return "INVERTED"
    return "UNSPECIFIED"


def stereo_registry(graph: nx.Graph) -> Mapping[str, StereoValue]:
    return graph.graph.get("stereo_descriptors", {})


def compare_stereo_registries(
    before: nx.Graph,
    after: nx.Graph,
    transition: nx.Graph | None = None,
) -> dict[str, StereoChange]:
    left, right = stereo_registry(before), stereo_registry(after)
    middle = stereo_registry(transition) if transition is not None else {}
    changes: dict[str, StereoChange] = {}
    for key in sorted(set(left) | set(right) | set(middle)):
        old, new, transient = left.get(key), right.get(key), middle.get(key)
        changes[key] = StereoChange(
            classify_stereo_change(old, new, transient),
            old,
            new,
            transient,
        )
    return changes


def annotate_its_stereo(
    its: nx.Graph,
    before: nx.Graph,
    after: nx.Graph,
    transition: nx.Graph | None = None,
) -> None:
    changes = compare_stereo_registries(before, after, transition)
    its.graph["stereo_changes"] = changes
    its.graph["stereo_descriptors"] = {
        "reactant": dict(stereo_registry(before)),
        "product": dict(stereo_registry(after)),
    }
    if transition is not None:
        its.graph["stereo_descriptors"]["transition"] = dict(
            stereo_registry(transition)
        )


def stereo_complete_reaction_center_nodes(its: nx.Graph) -> set[Any]:
    changes = its.graph.get("stereo_changes", {})
    if not changes:
        return set()
    atom_map_nodes = {}
    for node, attrs in its.nodes(data=True):
        atom_map = attrs.get("atom_map", node)
        if isinstance(atom_map, int):
            atom_map_nodes[atom_map] = node
        elif isinstance(atom_map, tuple) and len(atom_map) == 2:
            for side_map in atom_map:
                if isinstance(side_map, int) and side_map > 0:
                    atom_map_nodes[side_map] = node
    return {
        atom_map_nodes[atom_map]
        for change in changes.values()
        if change.change != "RETAINED"
        for atom_map in change.dependencies
        if atom_map in atom_map_nodes
    }
