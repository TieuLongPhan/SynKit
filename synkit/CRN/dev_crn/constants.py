from __future__ import annotations

from enum import Enum
from typing import Dict


class NodeKind(str, Enum):
    """Node category in the bipartite CRN graph."""

    SPECIES = "species"
    REACTION = "reaction"


class EdgeRole(str, Enum):
    """Edge role connecting species↔reaction nodes."""

    REACTANT = "reactant"
    PRODUCT = "product"


class RenderEngine(str, Enum):
    """Graphviz layout engine options."""

    DOT = "dot"
    NEATO = "neato"
    FDP = "fdp"
    SFDP = "sfdp"


class RenderFormat(str, Enum):
    """Rendering output format."""

    SVG = "svg"
    PNG = "png"
    PDF = "pdf"


# Defaults & tolerances
DEFAULT_MIN_OVERLAP: int = 1
DEFAULT_MAX_DEPTH: int = 12  # conservative default for enumerations
DEFAULT_GRAPH_ATTRS: Dict[str, str] = {
    "rankdir": "LR",
    "splines": "true",
    "nodesep": "0.6",
    "ranksep": "0.6",
    "overlap": "false",
}
