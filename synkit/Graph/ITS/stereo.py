"""Strict stereo-layer validation for ITS/DPO construction."""

from __future__ import annotations

from typing import Mapping

import networkx as nx

from synkit.Graph.Stereo import (
    StereoRefusal,
    StereoRefusalCode,
    StereoValue,
    descriptor_graph_support_errors,
)


class StereoITSValidationError(ValueError):
    """Raised when a stereo-bearing graph violates its support contract."""

    def __init__(self, refusals: tuple[StereoRefusal, ...]) -> None:
        self.refusals = refusals
        detail = "; ".join(f"{item.targets}: {item.detail}" for item in refusals)
        super().__init__(f"Invalid ITS stereo layer: {detail}")


def stereo_support_refusals(
    graph: nx.Graph,
    *,
    side: str,
    registry: Mapping[str, StereoValue] | None = None,
) -> tuple[StereoRefusal, ...]:
    """Return deterministic support/refusal evidence for one stereo layer."""
    values = (
        registry if registry is not None else graph.graph.get("stereo_descriptors", {})
    )
    refusals = []
    for target, descriptor in sorted(values.items()):
        errors = descriptor_graph_support_errors(
            graph,
            descriptor,
            registry_key=target,
        )
        if errors:
            refusals.append(
                StereoRefusal(
                    StereoRefusalCode.INVALID_REFERENCE,
                    f"{side}: " + "; ".join(errors),
                    (target,),
                )
            )
    return tuple(refusals)


def validate_stereo_support(
    graph: nx.Graph,
    *,
    side: str,
    registry: Mapping[str, StereoValue] | None = None,
) -> None:
    """Raise one aggregate error when any descriptor support is invalid."""
    refusals = stereo_support_refusals(
        graph,
        side=side,
        registry=registry,
    )
    if refusals:
        raise StereoITSValidationError(refusals)


__all__ = [
    "StereoITSValidationError",
    "stereo_support_refusals",
    "validate_stereo_support",
]
