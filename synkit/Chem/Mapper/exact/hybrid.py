"""Auditable exact-backend selection without changing shell semantics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from ..slap.lap import _adjacency_and_elements
from .distance import enumerate_distance_mappings, normalize_distance_target
from .distance_records import DistanceEnumerationResult
from .edit_support import (
    BinaryEditBudget,
    binary_edit_budget,
    enumerate_binary_edit_support_mappings,
)

HybridBackend = Literal["auto", "assignment", "edit_support"]


@dataclass(frozen=True)
class HybridBackendDecision:
    """Machine-readable explanation of one exact backend choice."""

    requested_backend: HybridBackend
    selected_backend: str
    selection_reason: str
    edit_support_pair_count: int | None
    edit_support_pair_limit: int | None
    output_scope: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _total_bijections(lgp, binary):
    reactant, reactant_elements = _adjacency_and_elements(lgp[0], binary)
    product, product_elements = _adjacency_and_elements(lgp[1], binary)
    if reactant.shape != product.shape:
        raise ValueError("reactant and product must have the same number of atoms")
    reactant_counts = Counter(reactant_elements)
    if reactant_counts != Counter(product_elements):
        raise ValueError("reactant and product atom-type multisets differ")
    total = math.prod(math.factorial(count) for count in reactant_counts.values())
    maximum = 0.5 * float(np.abs(reactant).sum() + np.abs(product).sum())
    return total, maximum


def _edit_ineligibility(target, binary, options):
    if target == "minimal":
        return "edit-support backend requires a numeric shell"
    if not binary:
        return "edit-support backend requires binary bond presence"
    if options.get("certify", False):
        return "edit-support backend does not emit prefix certificates"
    if options.get("symmetry_pruning", False):
        return "edit-support backend currently returns labeled mappings only"
    if options.get("expand_symmetry", False):
        return "edit-support backend does not use symmetry expansion"
    if options.get("fixed_mapping"):
        return "edit-support backend does not support fixed assignments"
    if options.get("compute_minimum_cost", False):
        return "edit-support backend does not run the optimization pass"
    return None


def enumerate_hybrid_distance_mappings(
    lgp,
    *,
    CD="minimal",
    binary=True,
    backend: HybridBackend = "auto",
    max_edit_support_pairs: int | None = 50_000,
    max_bijections: int | None = 1_000_000,
    tolerance: float = 1e-9,
    time_limit_seconds: float | None = None,
    max_mappings: int | None = None,
    collect_mappings: bool = True,
    mapping_callback=None,
    compute_minimum_cost: bool = False,
    **assignment_options,
) -> DistanceEnumerationResult:
    """Select an exact backend while preserving the requested output scope.

    The edit-support route is selected only for a numeric, binary, labeled
    shell without fixed assignments or certificate requests.  Otherwise the
    complete assignment branch-and-bound implementation is used.  Every result
    records the backend and selection evidence; automatic selection never
    substitutes a quotient shell for a labeled shell.
    """
    if backend not in {"auto", "assignment", "edit_support"}:
        raise ValueError("backend must be 'auto', 'assignment', or 'edit_support'")
    if max_edit_support_pairs is not None and (
        isinstance(max_edit_support_pairs, bool)
        or not isinstance(max_edit_support_pairs, Integral)
        or max_edit_support_pairs < 1
    ):
        raise ValueError("max_edit_support_pairs must be positive or None")

    target = normalize_distance_target(CD)
    options = dict(assignment_options)
    options["compute_minimum_cost"] = compute_minimum_cost
    reason = _edit_ineligibility(target, binary, options)
    budget: BinaryEditBudget | None = None
    if reason is None:
        try:
            budget = binary_edit_budget(lgp, target, tolerance=tolerance)
        except ValueError as error:
            reason = f"edit-support structural precondition failed: {error}"
    if (
        reason is None
        and budget is not None
        and max_edit_support_pairs is not None
        and budget.support_pair_count > max_edit_support_pairs
    ):
        reason = (
            f"estimated support pairs {budget.support_pair_count} exceed "
            f"selector limit {max_edit_support_pairs}"
        )

    select_edit = backend == "edit_support" or (backend == "auto" and reason is None)
    if backend == "edit_support" and reason is not None:
        raise ValueError(reason)
    if backend == "assignment":
        reason = "assignment backend explicitly requested"
        select_edit = False

    if not select_edit:
        decision = HybridBackendDecision(
            requested_backend=backend,
            selected_backend="assignment_branch_and_bound",
            selection_reason=reason or "assignment backend selected",
            edit_support_pair_count=(
                None if budget is None else budget.support_pair_count
            ),
            edit_support_pair_limit=max_edit_support_pairs,
            output_scope="complete_atom_compatible_assignment_space",
        )
        result = enumerate_distance_mappings(
            lgp,
            CD=target,
            binary=binary,
            max_bijections=max_bijections,
            tolerance=tolerance,
            time_limit_seconds=time_limit_seconds,
            max_mappings=max_mappings,
            collect_mappings=collect_mappings,
            mapping_callback=mapping_callback,
            compute_minimum_cost=compute_minimum_cost,
            **assignment_options,
        )
        result.backend = decision.selected_backend
        result.backend_statistics = {"selector": decision.as_dict()}
        return result

    decision = HybridBackendDecision(
        requested_backend=backend,
        selected_backend="binary_edit_support",
        selection_reason=(
            "numeric binary support estimate is within the selector limit"
        ),
        edit_support_pair_count=budget.support_pair_count,
        edit_support_pair_limit=max_edit_support_pairs,
        output_scope="complete_binary_atom_compatible_cd_shell",
    )
    edit_result = enumerate_binary_edit_support_mappings(
        lgp,
        target,
        tolerance=tolerance,
        max_support_pairs=max_edit_support_pairs,
        max_mappings=max_mappings,
        time_limit_seconds=time_limit_seconds,
        collect_mappings=collect_mappings,
        mapping_callback=mapping_callback,
    )
    total_bijections, maximum_cost = _total_bijections(lgp, True)
    selected_cost = float(target) if edit_result.selected_mapping_count else None
    status = (
        "timeout"
        if not edit_result.complete
        else ("complete" if edit_result.selected_mapping_count else "no_solutions")
    )
    return DistanceEnumerationResult(
        target=target,
        cost=selected_cost,
        minimum_cost=None,
        mappings=edit_result.mappings,
        distances=[float(target)] * len(edit_result.mappings),
        total_bijections=total_bijections,
        maximum_cost_upper_bound=maximum_cost,
        visited_leaves=edit_result.isomorphisms_visited,
        pruned_branches=(
            edit_result.support_pairs_visited - edit_result.invariant_compatible_pairs
        ),
        elapsed_seconds=edit_result.elapsed_seconds,
        status=status,
        complete=edit_result.complete,
        truncation_reason=edit_result.truncation_reason,
        scope=edit_result.scope,
        visited_nodes=edit_result.support_pairs_visited,
        selected_mapping_count=edit_result.selected_mapping_count,
        selected_labeled_mapping_count=edit_result.selected_mapping_count,
        backend=decision.selected_backend,
        backend_statistics={
            "selector": decision.as_dict(),
            "support_pairs_visited": edit_result.support_pairs_visited,
            "invariant_compatible_pairs": (edit_result.invariant_compatible_pairs),
            "isomorphisms_visited": edit_result.isomorphisms_visited,
        },
    )


__all__ = [
    "HybridBackend",
    "HybridBackendDecision",
    "enumerate_hybrid_distance_mappings",
]
