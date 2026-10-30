"""
Exact mapping methods, enumeration, and optimality certificates.

Full-space search and kernel refinement are distinct workflows. PABS
(Propagation and Assignment-Bounded Search) selects Python propagation or
optional C++ full-space search. The kernel solvers optimize a reduced
uncertainty region supplied by the SLAP workflow. A LAP (linear assignment
problem) provides a bound or a matching step in several of these methods.

Submodules
----------
search
    PABS public interface with explicit Python or C++ backend selection.
distance_search
    Full atom-compatible distance enumeration, replay certificates, admissible
    bounds, and the standalone small-graph exhaustive mapper.
kernel_solvers
    WL/SLAP uncertainty-kernel extraction, orbital branching, optional MILP,
    and enumeration within the supplied kernel restrictions.
pabs
    Python reversible-domain propagation search and residual cost spectra.
native
    Optional C++ search, canonicalization, orbit scheduling, and explicit
    compilation of the adjacent ``native_distance.cpp`` source. Execution
    requires a supplied library path.
assignment
    Shared LAP domains, incremental solves, and integer primal-dual proofs.
seeds
    Feasible-incumbent heuristics; these do not establish optimality.
hybrid, edit_support
    Exact-backend selection and binary edit-support enumeration.
core, hydrogen, symmetry, certificate
    Reference-component reduction, hydrogen lifting, permutation-group helpers,
    and optimality-gap reporting.

Historical flat modules remain aliases to these canonical implementations.
"""

from .kernel_solvers.kernel import Kernel, extract_kernel, apply_kernel_solution
from .kernel_solvers.branching import KernelSolution, solve_kernel, solve_kernel_blockwise
from .distance_search.exhaustive import ExactResult, ExactMapper
from .certificate import Certificate, certify_result, certify_results_exact
from .kernel_solvers.enumerate import (
    EnumerationResult,
    canonical_mapping_key,
    canonical_partial_mapping_key,
    dedup_mappings_by_automorphism,
    enumerate_kernel_optima,
    enumerate_symmetry_distinct_optima,
)
from .distance_search.search import (
    CertificateVerificationError,
    DistanceEnumerationCertificate,
    DistanceEnumerationResult,
    EnumerationStatus,
    ExactEnumerationLimitError,
    enumerate_distance_mappings,
    verify_distance_enumeration_certificate,
)
from .core import ITSComponentReduction, reduce_to_reference_its_components
from .hydrogen import (
    HydrogenEnumerationResult,
    HydrogenLiftSummary,
    HydrogenTransferPlan,
    enumerate_lgp_hydrogen_transfers,
    enumerate_minimal_hydrogen_transfers,
    summarize_minimal_hydrogen_lifts,
)
from .edit_support import (
    BinaryEditBudget,
    BinaryEditSupportResult,
    binary_edit_budget,
    enumerate_binary_edit_support_mappings,
)
from .hybrid import (
    HybridBackend,
    HybridBackendDecision,
    enumerate_hybrid_distance_mappings,
)
from .pabs.propagation import PropagationConfig, enumerate_synister_cp_mappings
from .search import enumerate_pabs_mappings
from .pabs.multi_shell import enumerate_pabs_shells

from ..._compat import (
    EXACT_MODULE_ALIASES as _EXACT_MODULE_ALIASES,
    register_module_aliases as _register_module_aliases,
)

_register_module_aliases(__name__, _EXACT_MODULE_ALIASES)

__all__ = [
    # kernel
    "Kernel",
    "extract_kernel",
    "apply_kernel_solution",
    # branching
    "KernelSolution",
    "solve_kernel",
    "solve_kernel_blockwise",
    # exhaustive
    "ExactResult",
    "ExactMapper",
    # certificate
    "Certificate",
    "certify_result",
    "certify_results_exact",
    # enumerate
    "EnumerationResult",
    "canonical_mapping_key",
    "canonical_partial_mapping_key",
    "dedup_mappings_by_automorphism",
    "enumerate_kernel_optima",
    "enumerate_symmetry_distinct_optima",
    # distance
    "CertificateVerificationError",
    "DistanceEnumerationCertificate",
    "DistanceEnumerationResult",
    "EnumerationStatus",
    "ExactEnumerationLimitError",
    "enumerate_distance_mappings",
    "verify_distance_enumeration_certificate",
    "ITSComponentReduction",
    "reduce_to_reference_its_components",
    "HydrogenEnumerationResult",
    "HydrogenLiftSummary",
    "HydrogenTransferPlan",
    "enumerate_lgp_hydrogen_transfers",
    "enumerate_minimal_hydrogen_transfers",
    "summarize_minimal_hydrogen_lifts",
    "BinaryEditBudget",
    "BinaryEditSupportResult",
    "binary_edit_budget",
    "enumerate_binary_edit_support_mappings",
    "HybridBackend",
    "HybridBackendDecision",
    "enumerate_hybrid_distance_mappings",
    "PropagationConfig",
    "enumerate_synister_cp_mappings",
    "enumerate_pabs_mappings",
    "enumerate_pabs_shells",
]
