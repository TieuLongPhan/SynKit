"""
mapper.exact — exact kernel solvers, enumeration, and optimality certificates.

Submodules
----------
kernel
    :class:`Kernel` dataclass and helpers to extract the uncertainty region
    from SLAP results and stitch a sub-mapping back into a full mapping.
branching
    Orbital-branching DFS solver (:func:`solve_kernel`) and block-decomposed
    variant (:func:`solve_kernel_blockwise`).
exhaustive
    Standalone exhaustive DFS mapper (:class:`ExactMapper`) for small graphs.
milp
    MILP/QAP formulation solved with PuLP/CBC (:func:`solve_kernel_milp`).
enumerate
    Symmetry-distinct enumeration of all exact optima (:func:`enumerate_kernel_optima`).
certificate
    Optimality certificates (:class:`Certificate`, :func:`certify_result`).
"""

from .kernel import Kernel, extract_kernel, apply_kernel_solution
from .branching import KernelSolution, solve_kernel, solve_kernel_blockwise
from .exhaustive import ExactResult, ExactMapper
from .certificate import Certificate, certify_result, certify_results_exact
from .enumerate import (
    EnumerationResult,
    canonical_mapping_key,
    canonical_partial_mapping_key,
    dedup_mappings_by_automorphism,
    enumerate_kernel_optima,
    enumerate_symmetry_distinct_optima,
)

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
]
