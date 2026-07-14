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


def project_record(
    record: MechanismRecord, target_format: str
) -> tuple[Any, ConversionLossReport]:
    """Project a record only when losses are explicitly disclosed."""
    if target_format == "json":
        return record.to_dict(), ConversionLossReport("MechanismRecord", "json")
    if target_format == "mapped_reaction_smiles":
        discarded = ["event_groups", "provenance"]
        if any(step.stereo_effects for step in record.steps):
            discarded.append("stereo_effects")
        if any(
            move.electron_count == 1
            for step in record.steps
            for group in step.groups
            for move in group.moves
        ):
            discarded.extend(("fishhook_events", "fishhook_coupling"))
        return record.mapped_reaction, ConversionLossReport(
            "MechanismRecord", target_format, tuple(discarded)
        )
    if target_format == "legacy_epd":
        from .adapters import legacy_epd_from_group

        rows = []
        for step in record.steps:
            for group in step.groups:
                rows.extend(legacy_epd_from_group(group))
        discarded = ["event_groups", "provenance"]
        if any(step.stereo_effects for step in record.steps):
            discarded.append("stereo_effects")
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
    serializable.graph.clear()
    serializable.graph["stereo_descriptors_json"] = json.dumps(
        {key: value.to_dict() for key, value in registry.items()},
        ensure_ascii=True,
        sort_keys=True,
    )
    return "\n".join(nx.generate_gml(serializable)), ConversionLossReport(
        "networkx", "gml"
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
        "SynKit LSG", "stereo connectivity projection", discarded
    )
