"""Stepwise lifecycle validation for supplied relative stereochemistry."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import networkx as nx

from .model import StereoDescriptor, StereoEffect, VerificationIssue


def descriptor_key(descriptor: StereoDescriptor) -> str:
    if descriptor.descriptor_class in {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }:
        return f"atom:{descriptor.atoms[0]}"
    if descriptor.descriptor_class in {"planar_bond", "atrop_bond"}:
        return f"bond:{min(descriptor.atoms[2:4])}-{max(descriptor.atoms[2:4])}"
    return "unknown"


def _maps(graph: nx.Graph) -> set[int]:
    return {
        int(data["atom_map"])
        for _, data in graph.nodes(data=True)
        if isinstance(data.get("atom_map"), int) and data["atom_map"] > 0
    }


def _descriptor_is_supported(graph: nx.Graph, descriptor: StereoDescriptor) -> bool:
    integer_refs = {value for value in descriptor.atoms if isinstance(value, int)}
    if not integer_refs <= _maps(graph):
        return False
    if descriptor.descriptor_class not in {"planar_bond", "atrop_bond"}:
        return True
    left, right = descriptor.atoms[2:4]
    lookup = {
        int(data["atom_map"]): node
        for node, data in graph.nodes(data=True)
        if isinstance(data.get("atom_map"), int)
    }
    if not isinstance(left, int) or not isinstance(right, int):
        return False
    if not graph.has_edge(lookup[left], lookup[right]):
        return False
    if descriptor.descriptor_class == "atrop_bond":
        return True
    return float(graph.edges[lookup[left], lookup[right]].get("pi_order", 0.0)) >= 1.0


def apply_stereo_effects(
    graph: nx.Graph,
    effects: Iterable[StereoEffect],
    *,
    step_id: str,
) -> tuple[nx.Graph, tuple[VerificationIssue, ...]]:
    """Apply declared effects without inferring reaction stereochemistry."""
    result = deepcopy(graph)
    registry = dict(result.graph.get("mechanism_stereo_descriptors", {}))
    issues: list[VerificationIssue] = []
    for effect in effects:
        target = f"{effect.descriptor_target[0]}:{effect.descriptor_target[1]}"
        present = registry.get(target)
        code = effect.effect
        if code in {"PRESERVE", "INVERT", "BREAK"} and present is None:
            issues.append(
                VerificationIssue(
                    "STEREO_TRANSITION_FROM_ABSENT",
                    f"{code} requires a present descriptor at {target}.",
                    step_id=step_id,
                )
            )
            continue
        if code == "PRESERVE":
            candidate = effect.after or present
            if candidate is None or not _descriptor_is_supported(result, candidate):
                issues.append(
                    VerificationIssue(
                        "INVALID_STEREO_PRESERVATION",
                        f"Descriptor references are not valid after step {step_id}.",
                        step_id=step_id,
                    )
                )
            else:
                registry[target] = candidate
        elif code == "INVERT":
            candidate = effect.after
            if candidate is None and present is not None:
                candidate = StereoDescriptor(
                    present.descriptor_class,
                    present.atoms,
                    -present.parity if present.parity else present.parity,
                    present.state,
                    present.provenance,
                )
            if candidate is None or not _descriptor_is_supported(result, candidate):
                issues.append(
                    VerificationIssue(
                        "INVALID_STEREO_INVERSION",
                        f"Inverted descriptor is incomplete at {target}.",
                        step_id=step_id,
                    )
                )
            else:
                registry[target] = candidate
        elif code == "BREAK":
            registry.pop(target, None)
        elif code == "FORM":
            if effect.after is None or effect.after.state != "specified":
                issues.append(
                    VerificationIssue(
                        "INCOMPLETE_STEREO_FORMATION",
                        f"Specified formation requires a complete descriptor at {target}.",
                        step_id=step_id,
                    )
                )
            elif not _descriptor_is_supported(result, effect.after):
                issues.append(
                    VerificationIssue(
                        "INVALID_STEREO_FORMATION",
                        f"Formed descriptor references invalid topology at {target}.",
                        step_id=step_id,
                    )
                )
            else:
                registry[target] = effect.after
        elif code == "UNSPECIFIED":
            unknown = effect.after or effect.before
            if unknown is not None:
                registry[target] = StereoDescriptor(
                    unknown.descriptor_class,
                    unknown.atoms,
                    None,
                    "unknown",
                    unknown.provenance,
                )
        elif code == "FLEETING":
            if effect.after is not None:
                registry[target] = effect.after
        else:
            issues.append(
                VerificationIssue(
                    "UNSUPPORTED_STEREO_EFFECT",
                    f"Unsupported stereo effect {code!r}.",
                    step_id=step_id,
                )
            )
    for key, descriptor in tuple(registry.items()):
        if not _descriptor_is_supported(result, descriptor):
            registry.pop(key)
            issues.append(
                VerificationIssue(
                    "UNDECLARED_STEREO_DESTRUCTION",
                    f"Topology no longer supports descriptor {key}; declare BREAK or UNSPECIFIED.",
                    step_id=step_id,
                )
            )
    result.graph["mechanism_stereo_descriptors"] = registry
    return result, tuple(issues)


def stereo_timeline(states: Iterable[nx.Graph]) -> dict[str, list[dict | None]]:
    """Return a rectangular descriptor-state history for MTG serialization."""
    registries = [
        state.graph.get("mechanism_stereo_descriptors", {}) for state in states
    ]
    keys = sorted({key for registry in registries for key in registry})
    result = {}
    for key in keys:
        history = [
            registry[key].to_dict() if key in registry else None
            for registry in registries
        ]
        if (
            history
            and history[0] is None
            and history[-1] is None
            and any(history[1:-1])
        ):
            for value in history:
                if value is not None:
                    value["lifecycle"] = "FLEETING"
        result[key] = history
    return result
