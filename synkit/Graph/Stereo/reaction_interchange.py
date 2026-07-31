"""Loss-accounted interchange for complete reaction-stereo values.

Molecule and reaction line notations do not have fields for executable rule
guards, stereo changes, branch populations, coupled outcomes, or typed
refusals.  This module therefore distinguishes lossless sidecar-capable
formats from endpoint-only carriers and makes every omitted semantic axis a
stable machine-readable issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import networkx as nx

from .wire import StereoReactionValue

REACTION_STEREO_INTERCHANGE_SCHEMA = "synkit.reaction-stereo-interchange/1"
REACTION_STEREO_SIDECAR_KEY = "reaction_stereo_json"
REACTION_STEREO_DIGEST_KEY = "reaction_stereo_sha256"

_SIDECAR_FORMATS = frozenset({"internal_graph", "gml"})
_WIRE_FORMATS = frozenset({"canonical_dict", "canonical_json"})
_ENDPOINT_FORMATS = frozenset({"reaction_smiles", "cxsmiles", "mol_v3000"})
SUPPORTED_REACTION_STEREO_FORMATS = _SIDECAR_FORMATS | _WIRE_FORMATS | _ENDPOINT_FORMATS
_AXES = (
    "guards",
    "effects",
    "outcomes",
    "couplings",
    "assertions",
    "refusals",
)


class ReactionStereoInterchangeError(ValueError):
    """Raised when strict projection would discard reaction semantics."""

    def __init__(self, report: "ReactionStereoInterchangeReport") -> None:
        self.report = report
        super().__init__(
            f"{report.target_format} cannot losslessly encode "
            f"{', '.join(issue.axis for issue in report.issues)}"
        )


@dataclass(frozen=True)
class ReactionStereoInterchangeIssue:
    """One semantic axis not representable in the target carrier."""

    code: str
    axis: str
    targets: tuple[str, ...]
    severity: str = "ERROR"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "axis": self.axis,
            "targets": list(self.targets),
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReactionStereoInterchangeReport:
    """Stable loss/refusal ledger for one projection."""

    target_format: str
    issues: tuple[ReactionStereoInterchangeIssue, ...] = ()
    sidecar_available: bool = False
    schema: str = REACTION_STEREO_INTERCHANGE_SCHEMA

    @property
    def lossless(self) -> bool:
        return not self.issues

    @property
    def status(self) -> str:
        return "PRESERVED" if self.lossless else "REFUSED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_format": self.target_format,
            "status": self.status,
            "lossless": self.lossless,
            "sidecar_available": self.sidecar_available,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _axis_targets(
    value: StereoReactionValue,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("guards", tuple(value.guards)),
        ("effects", tuple(value.effects)),
        ("outcomes", tuple(value.outcomes)),
        ("couplings", tuple(value.couplings)),
        ("assertions", tuple(value.assertions)),
        (
            "refusals",
            tuple(
                target
                for refusal in value.refusals
                for target in (refusal.targets or ("reaction",))
            ),
        ),
    )


def _endpoint_report(
    value: StereoReactionValue,
    target_format: str,
) -> ReactionStereoInterchangeReport:
    issues = tuple(
        ReactionStereoInterchangeIssue(
            "REACTION_AXIS_NOT_REPRESENTABLE",
            axis,
            tuple(sorted(set(targets))),
            detail=(
                f"{target_format} is an endpoint carrier and has no normative "
                f"field for reaction-stereo {axis}."
            ),
        )
        for axis, targets in _axis_targets(value)
        if targets
    )
    return ReactionStereoInterchangeReport(
        target_format,
        issues,
        sidecar_available=True,
    )


def project_reaction_stereo(
    value: StereoReactionValue,
    target_format: str,
    *,
    carrier: Any | None = None,
    strict: bool = True,
) -> tuple[Any, ReactionStereoInterchangeReport]:
    """Project a complete value, preserving it or reporting every lost axis.

    ``carrier`` is the endpoint text/value supplied by the caller for
    reaction SMILES, CXSMILES, or MOL V3000.  SynKit never claims that such a
    carrier contains rule semantics merely because the endpoints retain local
    stereochemical labels.
    """
    target_format = target_format.lower()
    if target_format not in SUPPORTED_REACTION_STEREO_FORMATS:
        raise ValueError(f"Unsupported reaction-stereo format: {target_format!r}.")
    if target_format == "canonical_dict":
        return value.to_dict(), ReactionStereoInterchangeReport(target_format)
    if target_format == "canonical_json":
        return (
            value.normalized_json(),
            ReactionStereoInterchangeReport(target_format),
        )
    if target_format == "internal_graph":
        if carrier is not None and not isinstance(carrier, nx.Graph):
            raise TypeError("internal_graph carrier must be a NetworkX graph.")
        graph = nx.Graph() if carrier is None else nx.Graph(carrier)
        _attach_sidecar(graph, value)
        return graph, ReactionStereoInterchangeReport(
            target_format,
            sidecar_available=True,
        )
    if target_format == "gml":
        if carrier is not None and not isinstance(carrier, nx.Graph):
            raise TypeError("gml carrier must be a NetworkX graph.")
        graph = nx.Graph() if carrier is None else nx.Graph(carrier)
        _attach_sidecar(graph, value)
        text = "\n".join(nx.generate_gml(graph))
        return text, ReactionStereoInterchangeReport(
            target_format,
            sidecar_available=True,
        )

    report = _endpoint_report(value, target_format)
    if strict and not report.lossless:
        raise ReactionStereoInterchangeError(report)
    return carrier, report


def _attach_sidecar(graph: nx.Graph, value: StereoReactionValue) -> None:
    payload = value.normalized_json()
    graph.graph[REACTION_STEREO_SIDECAR_KEY] = payload
    graph.graph[REACTION_STEREO_DIGEST_KEY] = sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _value_from_sidecar(
    attributes: Mapping[str, Any],
) -> StereoReactionValue:
    try:
        payload = str(attributes[REACTION_STEREO_SIDECAR_KEY])
        observed = str(attributes[REACTION_STEREO_DIGEST_KEY])
    except KeyError as error:
        raise ValueError("Reaction-stereo sidecar or digest is missing.") from error
    expected = sha256(payload.encode("utf-8")).hexdigest()
    if observed != expected:
        raise ValueError("Reaction-stereo sidecar digest mismatch.")
    return StereoReactionValue.from_json(payload)


def reaction_stereo_from_graph(
    graph: nx.Graph,
) -> StereoReactionValue:
    """Read and authenticate a reaction-stereo internal-graph sidecar."""
    return _value_from_sidecar(graph.graph)


def reaction_stereo_from_gml(
    text: str,
) -> tuple[nx.Graph, StereoReactionValue]:
    """Read GML and authenticate its complete reaction-stereo sidecar."""
    graph = nx.parse_gml(text.splitlines())
    return graph, _value_from_sidecar(graph.graph)


def reaction_stereo_interchange_schema() -> dict[str, Any]:
    """Return the dependency-free schema for the loss ledger."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": REACTION_STEREO_INTERCHANGE_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "target_format",
            "status",
            "lossless",
            "sidecar_available",
            "issues",
        ],
        "properties": {
            "schema": {"const": REACTION_STEREO_INTERCHANGE_SCHEMA},
            "target_format": {"enum": sorted(SUPPORTED_REACTION_STEREO_FORMATS)},
            "status": {"enum": ["PRESERVED", "REFUSED"]},
            "lossless": {"type": "boolean"},
            "sidecar_available": {"type": "boolean"},
            "issues": {"type": "array"},
        },
    }


__all__ = [
    "REACTION_STEREO_DIGEST_KEY",
    "REACTION_STEREO_INTERCHANGE_SCHEMA",
    "REACTION_STEREO_SIDECAR_KEY",
    "SUPPORTED_REACTION_STEREO_FORMATS",
    "ReactionStereoInterchangeError",
    "ReactionStereoInterchangeIssue",
    "ReactionStereoInterchangeReport",
    "project_reaction_stereo",
    "reaction_stereo_from_gml",
    "reaction_stereo_from_graph",
    "reaction_stereo_interchange_schema",
]
