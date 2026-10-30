"""Thin prepared-graph and reaction adapters for optional native solver engines.

The common core loads prebuilt libraries explicitly. No engine is compiled on
import, and an unavailable requested C++ backend never falls back to Python.
"""

from __future__ import annotations

from .core import enumerate_core_mappings, map_reaction_core

SOLVERS = frozenset({"cp_sat", "scip", "maxsat", "kmilp"})


def _backend(solver):
    if solver not in SOLVERS:
        raise ValueError("solver must be 'cp_sat', 'scip', 'maxsat', or 'kmilp'")
    return solver


def enumerate_solver_mappings(lgp, *, library_path, solver="cp_sat", **options):
    """Run a full-space or explicit-kernel engine through the doubled-CD ABI.

    CP-SAT enumerates the objective-free satisfaction model after minimum
    proof. SCIP and Open-WBO enumerate projected assignments with no-goods.
    Native class construction is requested separately from raw enumeration.
    """
    return enumerate_core_mappings(
        lgp, library_path=library_path, backend=_backend(solver), **options
    )


def map_reaction_solver(reaction, *, library_path, solver="cp_sat", **options):
    """Use the common chemistry frontend with an explicitly selected engine."""
    return map_reaction_core(
        reaction, library_path=library_path, backend=_backend(solver), **options
    )


def enumerate_kernel_milp_mappings(lgp, *, library_path, allowed_domains, **options):
    """Enumerate only the explicitly supplied compatible assignment kernel."""
    if allowed_domains is None:
        raise ValueError("KMILP requires explicit allowed_domains")
    return enumerate_core_mappings(
        lgp,
        library_path=library_path,
        backend="kmilp",
        allowed_domains=allowed_domains,
        **options,
    )


def map_reaction_kernel_milp(reaction, *, library_path, allowed_domains, **options):
    """Map within explicit domains using native classification by default.

    Set native_classification=False to select reference Python classification.
    The proof and enumeration scope remains the supplied kernel.
    """
    if allowed_domains is None:
        raise ValueError("KMILP requires explicit allowed_domains")
    options.setdefault("native_classification", True)
    return map_reaction_core(
        reaction,
        library_path=library_path,
        backend="kmilp",
        allowed_domains=allowed_domains,
        **options,
    )
