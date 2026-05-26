"""Functional-group detection on SynKit molecular graphs."""

from .catalog import default_registry
from .audit import FunctionalGroupAudit, audit_reaction_smiles
from .api import FunctionalGroupLabels, smiles_to_graph_and_functional_groups
from .detector import FunctionalGroupDetector
from .model import FunctionalGroupMatch, FunctionalGroupPattern, FunctionalGroupRegistry

__all__ = [
    "FunctionalGroupDetector",
    "FunctionalGroupLabels",
    "FunctionalGroupAudit",
    "FunctionalGroupMatch",
    "FunctionalGroupPattern",
    "FunctionalGroupRegistry",
    "default_registry",
    "audit_reaction_smiles",
    "smiles_to_graph_and_functional_groups",
]
