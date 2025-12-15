"""
Public API for :mod:`crn`.

This package is under active development; APIs may change without notice.

Re-exported
-----------
- :class:`crn.constants.NodeKind`, :class:`crn.constants.EdgeRole`
- :class:`crn.exceptions.CRNError`
- :class:`crn.properties.SpeciesProperty`, :class:`crn.properties.ReactionProperty`
- :class:`crn.reaction.Reaction`
- :class:`crn.network.ReactionNetwork`
- :class:`crn.pathway.Pathway`
- :class:`crn.explorer.ReactionPathwayExplorer`
- :class:`crn.viz.CRNVisualizer`
"""

from __future__ import annotations
import warnings


from .constants import NodeKind, EdgeRole, RenderEngine, RenderFormat
from .exceptions import (
    CRNError,
    InvalidReactionError,
    StandardizationError,
    VisualizationError,
    SearchError,
)
from .properties import SpeciesProperty, ReactionProperty, NodeProperty, EdgeProperty
from .reaction import Reaction
from .network import ReactionNetwork
from .pathway import Pathway
from .explorer import ReactionPathwayExplorer
from .viz import CRNVisualizer
from .helpers import (
    dedupe_pathways_by_canonical,
    pretty_print_pathway,
    replay_pathway_and_collect_inferred,
)
from .enumerator import (
    MotifEnumerator,
)

warnings.warn(
    "⚠️  This module is under active development. "
    "APIs may change without notice and stability is not guaranteed.",
    category=UserWarning,
    stacklevel=2,
)


__all__ = [
    # constants
    "NodeKind",
    "EdgeRole",
    "RenderEngine",
    "RenderFormat",
    # exceptions
    "CRNError",
    "InvalidReactionError",
    "StandardizationError",
    "VisualizationError",
    "SearchError",
    # property layer
    "SpeciesProperty",
    "ReactionProperty",
    "NodeProperty",
    "EdgeProperty",
    # core classes
    "Reaction",
    "ReactionNetwork",
    "Pathway",
    "ReactionPathwayExplorer",
    "CRNVisualizer",
    # utilities
    "MotifEnumerator",
    "dedupe_pathways_by_canonical",
    "pretty_print_pathway",
    "replay_pathway_and_collect_inferred",
]

__version__ = "0.2.0-dev"
