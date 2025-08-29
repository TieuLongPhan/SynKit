from __future__ import annotations


class CRNError(RuntimeError):
    """Base class for all CRN-specific errors."""


class InvalidReactionError(CRNError):
    """Raised when a reaction string is malformed or cannot be parsed."""


class StandardizationError(CRNError):
    """Raised when reaction standardization fails irrecoverably."""


class VisualizationError(CRNError):
    """Raised when visualization backends fail (Graphviz/matplotlib)."""


class SearchError(CRNError):
    """Raised for search/enumeration issues (invalid arguments, overflow, etc.)."""
