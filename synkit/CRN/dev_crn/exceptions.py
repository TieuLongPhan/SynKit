from __future__ import annotations

__all__ = [
    "CRNError",
    "InvalidReactionError",
    "StandardizationError",
    "VisualizationError",
    "SearchError",
    "NoPathwaysError",
    "EnumeratorError",
]


class CRNError(RuntimeError):
    """
    Base class for all CRN-specific errors.

    Use this as the universal catch-all for CRN-related failures.
    """

    pass


class InvalidReactionError(CRNError):
    """
    Raised when a reaction string is malformed or cannot be parsed.

    :param message: Optional error message describing the invalid reaction.
    :type message: str
    :raises InvalidReactionError: Always raised when parsing fails.
    """

    pass


class StandardizationError(CRNError):
    """
    Raised when reaction standardization or canonicalization fails irrecoverably.

    :param message: Optional error message describing the failure.
    :type message: str
    :raises StandardizationError: Always raised when standardization fails.
    """

    pass


class VisualizationError(CRNError):
    """
    Raised when visualization backends fail (Graphviz, matplotlib, file I/O).

    :param message: Optional error message describing the visualization failure.
    :type message: str
    :raises VisualizationError: Always raised when rendering/exporting fails.
    """

    pass


class SearchError(CRNError):
    """
    Raised for search/enumeration issues (invalid arguments, overflow, etc.).

    :param message: Optional error message describing the search issue.
    :type message: str
    :raises SearchError: Always raised when search-related problems occur.
    """

    pass


class NoPathwaysError(CRNError):
    """
    Raised when the enumerator finds no (unique) pathways for a motif.

    :param motif_name: Name of the motif that failed enumeration.
    :type motif_name: str
    :param raw_count: Number of raw pathways found (usually 0).
    :type raw_count: int
    :param hint: Optional diagnostic hint for debugging (e.g. missing sources).
    :type hint: str
    :raises NoPathwaysError: Always raised when no pathways are found.
    """

    pass


class EnumeratorError(CRNError):
    """
    General enumerator error (non-specific).

    :param message: Optional error message describing the enumerator issue.
    :type message: str
    :raises EnumeratorError: Raised for non-standard failures in enumeration.
    """

    pass
