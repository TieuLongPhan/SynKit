from typing import TYPE_CHECKING

from .fusion_validation import (
    FusionIssue,
    FusionIssueCode,
    FusionValidation,
    WildcardRole,
    validate_endpoint_preservation,
    validate_fusion_rsmi,
    validate_rbl_candidate,
    validate_wildcard_mapping_roles,
)
from .rbl_policy import RBLSearchPolicy, SearchScope, TerminationPolicy
from .rbl_state import RBL_RESULT_SCHEMA
from .assignment import StereoBranchLimitError, StereoWildcardAssignmentLimitError
from .serialization_policy import RawITSApplicationSerializationWarning

if TYPE_CHECKING:
    from .rbl_engine import RBLEngine

__all__ = [
    "FusionIssue",
    "FusionIssueCode",
    "FusionValidation",
    "WildcardRole",
    "validate_endpoint_preservation",
    "validate_fusion_rsmi",
    "validate_rbl_candidate",
    "validate_wildcard_mapping_roles",
    "RBLSearchPolicy",
    "SearchScope",
    "TerminationPolicy",
    "RBLEngine",
    "RBL_RESULT_SCHEMA",
    "StereoBranchLimitError",
    "StereoWildcardAssignmentLimitError",
    "RawITSApplicationSerializationWarning",
]


def __getattr__(name: str) -> object:
    """Load heavyweight public reactor classes on first access.

    :param name: Requested package attribute.
    :type name: str
    :return: Lazily imported public object.
    :rtype: object
    :raises AttributeError: If ``name`` is not a lazy public export.
    """
    if name == "RBLEngine":
        from .rbl_engine import RBLEngine

        return RBLEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
