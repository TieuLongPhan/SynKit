"""Loss-reporting JSON and graph interchange helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import networkx as nx

from synkit.Graph.Stereo import stereo_from_dict

from .model import MechanismRecord


@dataclass(frozen=True)
class ConversionLossReport:
    source_format: str
    target_format: str
    discarded_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def lossless(self) -> bool:
        return not self.discarded_fields and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "lossless": self.lossless,
            "discarded_fields": list(self.discarded_fields),
            "warnings": list(self.warnings),
        }


def _record_projection_losses(
    record: MechanismRecord,
    *,
    include_fishhook: bool,
) -> list[str]:
    """Return fields omitted by a flattened mechanism projection."""
    discarded = ["event_groups", "provenance"]
    optional_fields = (
        ("metadata", bool(record.metadata)),
        ("endpoint_stereo", bool(record.endpoint_stereo)),
        ("stereo_effects", any(step.stereo_effects for step in record.steps)),
        ("stereo_motions", any(step.stereo_motions for step in record.steps)),
    )
    discarded.extend(name for name, present in optional_fields if present)
    has_fishhook = include_fishhook and any(
        move.electron_count == 1
        for step in record.steps
        for group in step.groups
        for move in group.moves
    )
    if has_fishhook:
        discarded.extend(("fishhook_events", "fishhook_coupling"))
    return discarded


def project_record(
    record: MechanismRecord, target_format: str
) -> tuple[Any, ConversionLossReport]:
    """Project a record only when losses are explicitly disclosed."""
    if target_format == "json":
        return record.to_dict(), ConversionLossReport("MechanismRecord", "json")
    if target_format == "mapped_reaction_smiles":
        discarded = _record_projection_losses(record, include_fishhook=True)
        return record.mapped_reaction, ConversionLossReport(
            "MechanismRecord", target_format, tuple(discarded)
        )
    if target_format == "legacy_epd":
        from .adapters import legacy_epd_from_group

        rows = []
        for step in record.steps:
            for group in step.groups:
                rows.extend(legacy_epd_from_group(group))
        discarded = _record_projection_losses(record, include_fishhook=False)
        return rows, ConversionLossReport(
            "MechanismRecord", target_format, tuple(discarded)
        )
    raise ValueError(f"Unsupported target format: {target_format!r}")


def record_from_json_value(value: Mapping[str, Any]) -> MechanismRecord:
    return MechanismRecord.from_dict(value)


def stereo_graph_to_gml(graph: nx.Graph) -> tuple[str, ConversionLossReport]:
    """Serialize graph-level descriptors through a JSON-valued GML attribute."""
    serializable = nx.Graph(graph)
    registry = graph.graph.get("stereo_descriptors", {})
    discarded = tuple(
        f"graph.{key}" for key in sorted(graph.graph) if key != "stereo_descriptors"
    )
    serializable.graph.clear()
    serializable.graph["stereo_descriptors_json"] = json.dumps(
        {key: value.to_dict() for key, value in registry.items()},
        ensure_ascii=True,
        sort_keys=True,
    )
    return "\n".join(nx.generate_gml(serializable)), ConversionLossReport(
        "networkx", "gml", discarded
    )


def stereo_graph_from_gml(text: str) -> nx.Graph:
    """Read GML emitted by :func:`stereo_graph_to_gml`."""
    graph = nx.parse_gml(text.splitlines())
    encoded = graph.graph.pop("stereo_descriptors_json", "{}")
    graph.graph["stereo_descriptors"] = {
        key: stereo_from_dict(value) for key, value in json.loads(encoded).items()
    }
    return graph


def project_stereo_graph(graph: nx.Graph) -> tuple[nx.Graph, ConversionLossReport]:
    """Project connectivity and relative stereo for an optional external backend."""
    projected = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        projected.add_node(
            node, element=attrs.get("element", "*"), atom_map=attrs.get("atom_map", 0)
        )
    projected.add_edges_from(graph.edges)
    projected.graph["stereo_descriptors"] = dict(
        graph.graph.get("stereo_descriptors", {})
    )
    discarded = (
        "charge",
        "radical",
        "lone_pairs",
        "sigma_order",
        "pi_order",
        "event_groups",
        "provenance",
    )
    return projected, ConversionLossReport(
        "SynKit LLG", "stereo connectivity projection", discarded
    )
