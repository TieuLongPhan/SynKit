"""Radical-based linking search, validation, and replayable proofs."""

from typing import TYPE_CHECKING

from .overlap import (
    TypedOverlapCertificate,
    TypedOverlapLimits,
    TypedOverlapResult,
    enumerate_typed_overlaps,
)
from .policy import (
    AcceptanceTask,
    OverlapScope,
    ProofLevel,
    RBLSearchPolicy,
    SearchOutcomeStatus,
    SearchScope,
    TerminationPolicy,
)
from .proof import (
    RBL_PROOF_SCHEMA,
    RBLProofReplay,
    RBLReplayCertificate,
    read_rbl_proof,
)
from .state import RBL_RESULT_SCHEMA
from .validation import (
    FusionIssue,
    FusionIssueCode,
    FusionValidation,
    WildcardRole,
    certify_fusion_postprocessing,
    validate_endpoint_preservation,
    validate_fusion_rsmi,
    validate_rbl_candidate,
    validate_strict_rbl_candidate,
    validate_wildcard_mapping_roles,
)

if TYPE_CHECKING:
    from .engine import RBLEngine

__all__ = [
    "AcceptanceTask",
    "FusionIssue",
    "FusionIssueCode",
    "FusionValidation",
    "OverlapScope",
    "ProofLevel",
    "RBLEngine",
    "RBLProofReplay",
    "RBLReplayCertificate",
    "RBLSearchPolicy",
    "RBL_PROOF_SCHEMA",
    "RBL_RESULT_SCHEMA",
    "SearchOutcomeStatus",
    "SearchScope",
    "TerminationPolicy",
    "TypedOverlapCertificate",
    "TypedOverlapLimits",
    "TypedOverlapResult",
    "WildcardRole",
    "certify_fusion_postprocessing",
    "enumerate_typed_overlaps",
    "read_rbl_proof",
    "validate_endpoint_preservation",
    "validate_fusion_rsmi",
    "validate_rbl_candidate",
    "validate_strict_rbl_candidate",
    "validate_wildcard_mapping_roles",
]


def __getattr__(name: str) -> object:
    """Load the engine lazily so lightweight policy imports stay cheap."""
    if name == "RBLEngine":
        from .engine import RBLEngine

        return RBLEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
