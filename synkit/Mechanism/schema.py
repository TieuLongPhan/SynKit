"""Small, dependency-free JSON-schema view of the v2 public record."""

from __future__ import annotations

from typing import Any

from .model import SCHEMA_VERSION


def mechanism_record_schema() -> dict[str, Any]:
    """Return the frozen draft-1 JSON schema used by public serializers."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SynKit MechanismRecord",
        "type": "object",
        "required": ["schema_version", "mapped_reaction", "steps"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "mapped_reaction": {"type": "string", "pattern": ">>"},
            "steps": {
                "type": "array",
                "items": {"$ref": "#/$defs/mechanisticStep"},
            },
            "provenance": {"type": "object"},
            "metadata": {"type": "object"},
        },
        "additionalProperties": False,
        "$defs": {
            "electronLocus": {
                "type": "object",
                "required": ["locus", "atom_maps"],
                "properties": {
                    "locus": {"enum": ["lp", "σ", "π", "∙"]},
                    "atom_maps": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                },
                "additionalProperties": False,
            },
            "electronMove": {
                "type": "object",
                "required": [
                    "source",
                    "target",
                    "electron_count",
                    "arrow_type",
                    "group_id",
                ],
                "properties": {
                    "event_id": {"type": ["string", "null"]},
                    "source": {"$ref": "#/$defs/electronLocus"},
                    "target": {"$ref": "#/$defs/electronLocus"},
                    "electron_count": {"enum": [1, 2]},
                    "arrow_type": {"enum": ["fishhook", "curved"]},
                    "group_id": {"type": "string", "minLength": 1},
                    "coupling_id": {"type": ["string", "null"]},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "electronMoveGroup": {
                "type": "object",
                "required": ["group_id", "moves"],
                "properties": {
                    "group_id": {"type": "string", "minLength": 1},
                    "macro": {"type": ["string", "null"]},
                    "moves": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/electronMove"},
                    },
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "mechanisticStep": {
                "type": "object",
                "required": ["step_id", "groups"],
                "properties": {
                    "step_id": {"type": "string", "minLength": 1},
                    "groups": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/electronMoveGroup"},
                    },
                    "stereo_effects": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/stereoEffect"},
                    },
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "stereoDescriptor": {
                "type": "object",
                "required": [
                    "descriptor_class",
                    "atoms",
                    "parity",
                    "state",
                    "provenance",
                ],
                "properties": {
                    "descriptor_class": {
                        "enum": [
                            "tetrahedral",
                            "square_planar",
                            "trigonal_bipyramidal",
                            "octahedral",
                            "planar_bond",
                            "atrop_bond",
                            "unknown",
                        ]
                    },
                    "atoms": {
                        "type": "array",
                        "items": {"type": ["integer", "string", "null"]},
                    },
                    "parity": {"type": ["integer", "null"], "enum": [-1, 0, 1, None]},
                    "state": {
                        "enum": ["specified", "unknown", "unspecified", "absent"]
                    },
                    "provenance": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            "stereoEffect": {
                "type": "object",
                "required": [
                    "descriptor_target",
                    "effect",
                    "before",
                    "after",
                    "provenance",
                ],
                "properties": {
                    "descriptor_target": {
                        "type": "array",
                        "prefixItems": [{"type": "string"}, {"type": "integer"}],
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "effect": {
                        "enum": [
                            "PRESERVE",
                            "INVERT",
                            "BREAK",
                            "FORM",
                            "FLEETING",
                            "UNSPECIFIED",
                        ]
                    },
                    "before": {
                        "anyOf": [
                            {"$ref": "#/$defs/stereoDescriptor"},
                            {"type": "null"},
                        ]
                    },
                    "after": {
                        "anyOf": [
                            {"$ref": "#/$defs/stereoDescriptor"},
                            {"type": "null"},
                        ]
                    },
                    "provenance": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    }
