"""
Public API for :mod:`CRN`.

.. warning::

   This package is currently **under development** and the public API is **not stable**.
   Interfaces, class names, function signatures and behaviors may change without notice.
   Use with caution for production systems.

Re-exported classes
-------------------
- :class:`~CRN.reaction.Reaction`
- :class:`~CRN.network.ReactionNetwork`
- :class:`~CRN.pathway.Pathway`
- :class:`~CRN.explorer.ReactionPathwayExplorer`
"""

from __future__ import annotations
import warnings
from typing import List

from .reaction import Reaction
from .network import ReactionNetwork
from .pathway import Pathway
from .explorer import ReactionPathwayExplorer

__all__: List[str] = [
    "Reaction",
    "ReactionNetwork",
    "Pathway",
    "ReactionPathwayExplorer",
]

__version__ = "0.0.0-dev"

# Emit a warning on import (only warnings, no logging)
warnings.warn(
    "CRN is under development and the API is not stable. "
    "It may change without notice.",
    UserWarning,
    stacklevel=2,
)
