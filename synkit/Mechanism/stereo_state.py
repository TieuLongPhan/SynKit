"""Stepwise lifecycle validation for supplied relative stereochemistry."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import networkx as nx

from synkit.Graph.Stereo import (
    classify_stereo_change,
    descriptor_graph_support_errors,
    stereo_from_dict,
)

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


def stereo_effect_target_key(effect: StereoEffect) -> str:
    """Return the registry key addressed by one validated stereo effect."""
    kind, reference = effect.descriptor_target
    if kind == "atom":
        return f"atom:{reference}"
    left, right = reference
    return f"bond:{left}-{right}"


def descriptor_support_errors(
    graph: nx.Graph,
    descriptor: StereoDescriptor,
    *,
    registry_key: str | None = None,
) -> tuple[str, ...]:
    """Return topology/resource reasons a descriptor is unsupported."""
    if descriptor.descriptor_class == "unknown":
        return ("unknown descriptor class has no executable locus",)
    return descriptor_graph_support_errors(
        graph,
        stereo_from_dict(descriptor.to_dict()),
        registry_key=registry_key,
    )


def descriptor_is_supported(
    graph: nx.Graph,
    descriptor: StereoDescriptor,
    *,
    registry_key: str | None = None,
) -> bool:
    return not descriptor_support_errors(
        graph,
        descriptor,
        registry_key=registry_key,
    )


def _descriptor_identity(descriptor: StereoDescriptor) -> tuple[object, ...]:
    if descriptor.descriptor_class == "unknown":
        return ("unknown", descriptor.state)
    native = stereo_from_dict(descriptor.to_dict())
    return (descriptor.state, *native.canonical_form())


def _descriptors_equal(
    left: StereoDescriptor | None,
    right: StereoDescriptor | None,
) -> bool:
    return (
        left is None
        and right is None
        or (
            left is not None
            and right is not None
            and _descriptor_identity(left) == _descriptor_identity(right)
        )
    )


def _inverted(descriptor: StereoDescriptor) -> StereoDescriptor:
    return descriptor.inverted()


def _change_kind(
    before: StereoDescriptor,
    after: StereoDescriptor,
) -> str:
    """Classify relative stereo, including one mapped ligand replacement."""
    return classify_stereo_change(
        stereo_from_dict(before.to_dict()),
        stereo_from_dict(after.to_dict()),
    )


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
        target = stereo_effect_target_key(effect)
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
        if code in {"PRESERVE", "INVERT", "BREAK"}:
            if effect.before is None:
                issues.append(
                    VerificationIssue(
                        "MISSING_STEREO_BEFORE",
                        f"{code} requires the declared pre-step descriptor at {target}.",
                        step_id=step_id,
                    )
                )
                continue
            if not _descriptors_equal(effect.before, present):
                issues.append(
                    VerificationIssue(
                        "STEREO_BEFORE_MISMATCH",
                        f"Declared stereo pre-state does not match {target}.",
                        step_id=step_id,
                        expected=present.to_dict() if present else None,
                        observed=effect.before.to_dict(),
                    )
                )
                continue
        if code == "PRESERVE":
            candidate = effect.after or present
            if (
                candidate is None
                or _change_kind(present, candidate) != "RETAINED"
                or not descriptor_is_supported(
                    result,
                    candidate,
                    registry_key=target,
                )
            ):
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
            candidate = effect.after or _inverted(present)
            if _change_kind(
                present, candidate
            ) != "INVERTED" or not descriptor_is_supported(
                result,
                candidate,
                registry_key=target,
            ):
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
            if effect.after is not None and effect.after.state != "absent":
                issues.append(
                    VerificationIssue(
                        "INVALID_STEREO_BREAK",
                        f"BREAK cannot retain an endpoint descriptor at {target}.",
                        step_id=step_id,
                    )
                )
                continue
            registry.pop(target, None)
        elif code == "FORM":
            if present is not None:
                issues.append(
                    VerificationIssue(
                        "STEREO_TRANSITION_FROM_PRESENT",
                        f"FORM requires an absent descriptor at {target}.",
                        step_id=step_id,
                    )
                )
            elif effect.after is None or effect.after.state != "specified":
                issues.append(
                    VerificationIssue(
                        "INCOMPLETE_STEREO_FORMATION",
                        f"Specified formation requires a complete descriptor at {target}.",
                        step_id=step_id,
                    )
                )
            elif not descriptor_is_supported(
                result,
                effect.after,
                registry_key=target,
            ):
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
            if effect.before is not None and not _descriptors_equal(
                effect.before,
                present,
            ):
                issues.append(
                    VerificationIssue(
                        "STEREO_BEFORE_MISMATCH",
                        f"Declared stereo pre-state does not match {target}.",
                        step_id=step_id,
                        expected=present.to_dict() if present else None,
                        observed=effect.before.to_dict(),
                    )
                )
            elif effect.before is None and present is not None:
                issues.append(
                    VerificationIssue(
                        "MISSING_STEREO_BEFORE",
                        f"UNSPECIFIED requires the present descriptor at {target}.",
                        step_id=step_id,
                    )
                )
            elif unknown is None:
                issues.append(
                    VerificationIssue(
                        "INCOMPLETE_STEREO_FORMATION",
                        f"UNSPECIFIED requires descriptor topology at {target}.",
                        step_id=step_id,
                    )
                )
            else:
                candidate = StereoDescriptor(
                    unknown.descriptor_class,
                    unknown.atoms,
                    None,
                    "unknown",
                    unknown.provenance,
                )
                if not descriptor_is_supported(
                    result,
                    candidate,
                    registry_key=target,
                ):
                    issues.append(
                        VerificationIssue(
                            "INVALID_STEREO_FORMATION",
                            f"Unknown descriptor references are invalid at {target}.",
                            step_id=step_id,
                        )
                    )
                    continue
                registry[target] = StereoDescriptor(
                    unknown.descriptor_class,
                    unknown.atoms,
                    None,
                    "unknown",
                    unknown.provenance,
                )
        elif code == "FLEETING":
            if effect.before is not None and not _descriptors_equal(
                effect.before,
                present,
            ):
                issues.append(
                    VerificationIssue(
                        "STEREO_BEFORE_MISMATCH",
                        f"Declared fleeting pre-state does not match {target}.",
                        step_id=step_id,
                    )
                )
        else:
            issues.append(
                VerificationIssue(
                    "UNSUPPORTED_STEREO_EFFECT",
                    f"Unsupported stereo effect {code!r}.",
                    step_id=step_id,
                )
            )
    for key, descriptor in tuple(registry.items()):
        if not descriptor_is_supported(result, descriptor, registry_key=key):
            registry.pop(key)
            issues.append(
                VerificationIssue(
                    "UNDECLARED_STEREO_DESTRUCTION",
                    f"Topology no longer supports descriptor {key}; declare BREAK or UNSPECIFIED.",
                    step_id=step_id,
                )
            )
    if issues:
        return deepcopy(graph), tuple(issues)
    result.graph["mechanism_stereo_descriptors"] = registry
    return result, ()


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
