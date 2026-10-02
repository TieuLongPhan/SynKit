"""Shared exact cost-spectrum queries for several numeric PABS shells."""

from __future__ import annotations

import math
import time
from collections import Counter
from numbers import Integral, Real

import numpy as np

from ..slap.lap import _adjacency_and_elements
from .distance_records import (
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    normalize_distance_target,
)
from .propagation import (
    _eligible,
    _fixed_and_seed,
    enumerate_synister_cp_mappings,
)
from .propagation_limits import PropagationDeadline
from .suffix_spectrum import SuffixCostSpectrum


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _timeout_result(target, total, maximum, elapsed, fixed):
    return DistanceEnumerationResult(
        target=target,
        cost=None,
        minimum_cost=None,
        mappings=[],
        distances=[],
        total_bijections=total,
        maximum_cost_upper_bound=maximum,
        visited_leaves=0,
        pruned_branches=0,
        elapsed_seconds=elapsed,
        status="timeout",
        complete=False,
        truncation_reason="time_limit",
        scope=(
            "fixed_assignment_subspace"
            if fixed
            else "complete_atom_compatible_assignment_space"
        ),
        visited_nodes=0,
        selected_mapping_count=0,
        selected_labeled_mapping_count=0,
        backend="pabs_suffix_spectrum",
    )


def enumerate_pabs_shells(  # noqa: C901
    lgp,
    CDs,
    *,
    binary=False,
    fixed_mapping=None,
    initial_mapping=None,
    tolerance=1e-9,
    time_limit_seconds=None,
    max_bijections=1_000_000,
    max_mappings=100_000,
    max_residual_size=10,
    max_cost=4096,
    max_states=20_000,
    max_seconds_per_prepare=0.05,
):
    """Enumerate several exact numeric CD shells, sharing one suffix diagram.

    The shared decision diagram is used when the loop-free weighted graph pair
    is supported, the non-fixed residual has at most ``max_residual_size``
    atoms, and preparation fits both resource caps. Otherwise each requested
    shell falls back to the ordinary exact Python PABS solver. A single wall
    deadline covers preparation and every shell query. The C++ backend, minimum
    proof, symmetry quotienting and callbacks are intentionally outside this
    batch API; returned maps are complete labeled mappings unless a resource or
    output cap is reported.

    ``CDs`` must contain nonnegative numeric targets. Results are keyed by the
    normalized float targets. Duplicate requests are evaluated once. When the
    shared diagram prepares successfully, every shell uses the same exact
    reachable-cost support and stored suffix states.
    """
    if not isinstance(binary, bool):
        raise TypeError("binary must be Boolean")
    if isinstance(CDs, (str, bytes)):
        raise TypeError("CDs must be a nonempty iterable of numeric targets")
    try:
        targets = tuple(dict.fromkeys(normalize_distance_target(cd) for cd in CDs))
    except TypeError:
        raise TypeError("CDs must be a nonempty iterable of numeric targets") from None
    if not targets:
        raise ValueError("CDs must contain at least one numeric target")
    if any(target == "minimal" for target in targets):
        raise ValueError("enumerate_pabs_shells accepts numeric CDs only")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a finite nonnegative number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds <= 0
    ):
        raise ValueError("time_limit_seconds must be finite and positive")
    max_bijections = (
        None
        if max_bijections is None
        else _positive_integer(max_bijections, "max_bijections")
    )
    max_mappings = (
        None
        if max_mappings is None
        else _positive_integer(max_mappings, "max_mappings")
    )
    max_residual_size = _positive_integer(max_residual_size, "max_residual_size")
    max_cost = _nonnegative_integer(max_cost, "max_cost")
    max_states = _positive_integer(max_states, "max_states")
    if (
        isinstance(max_seconds_per_prepare, bool)
        or not isinstance(max_seconds_per_prepare, Real)
        or not math.isfinite(max_seconds_per_prepare)
        or max_seconds_per_prepare <= 0
    ):
        raise ValueError("max_seconds_per_prepare must be finite and positive")

    started = time.perf_counter()
    deadline = (
        None if time_limit_seconds is None else started + float(time_limit_seconds)
    )
    a_float, labels = _adjacency_and_elements(lgp[0], binary)
    b_float, product_labels = _adjacency_and_elements(lgp[1], binary)
    if a_float.shape != b_float.shape or Counter(labels) != Counter(product_labels):
        raise ValueError("Endpoints must have equal atom-type inventories")
    fixed, seed = _fixed_and_seed(
        labels, product_labels, fixed_mapping, initial_mapping
    )
    n = len(labels)
    residual_rows = tuple(i for i in range(n) if i not in fixed)
    used_fixed = set(fixed.values())
    residual_columns = tuple(j for j in range(n) if j not in used_fixed)
    counts = Counter(labels[i] for i in residual_rows)
    total = math.prod(math.factorial(count) for count in counts.values())
    if max_bijections is not None and total > max_bijections:
        raise ExactEnumerationLimitError(
            f"{total:,} atom-compatible bijections exceed "
            f"max_bijections={max_bijections:,}"
        )
    maximum = 0.5 * float(np.abs(a_float).sum() + np.abs(b_float).sum())

    def remaining_seconds():
        return None if deadline is None else max(0.0, deadline - time.perf_counter())

    def fallback(reason):
        results = {}
        for target in targets:
            remaining = remaining_seconds()
            if remaining is not None and remaining <= 0:
                results[target] = _timeout_result(
                    target, total, maximum, time.perf_counter() - started, fixed
                )
                continue
            result = enumerate_synister_cp_mappings(
                lgp,
                CD=target,
                binary=binary,
                fixed_mapping=fixed,
                initial_mapping=seed,
                tolerance=tolerance,
                time_limit_seconds=remaining,
                max_bijections=max_bijections,
                max_mappings=max_mappings,
                compute_minimum_cost=False,
            )
            statistics = dict(result.backend_statistics or {})
            statistics["multi_shell_fallback_reason"] = reason
            results[target] = DistanceEnumerationResult(
                **{
                    **result.__dict__,
                    "backend_statistics": statistics,
                }
            )
        return results

    if _eligible(a_float, b_float, False) is not None:
        return fallback("unsupported_graph_domain")
    if len(residual_rows) > max_residual_size:
        return fallback("residual_size_limit")

    a = np.rint(a_float * 4).astype(np.int64)
    b = np.rint(b_float * 4).astype(np.int64)
    fixed_rows = tuple(fixed)
    fixed_cost = sum(
        abs(int(a[left, right]) - int(b[fixed[left], fixed[right]]))
        for position, left in enumerate(fixed_rows)
        for right in fixed_rows[position + 1 :]
    )
    unary = np.zeros((len(residual_rows), len(residual_columns)), dtype=np.int64)
    allowed = np.zeros_like(unary, dtype=bool)
    for row_pos, row in enumerate(residual_rows):
        for col_pos, col in enumerate(residual_columns):
            allowed[row_pos, col_pos] = labels[row] == product_labels[col]
            unary[row_pos, col_pos] = sum(
                abs(int(a[row, fixed_row]) - int(b[col, fixed_image]))
                for fixed_row, fixed_image in fixed.items()
            )

    shell_bounds = {
        target: (
            math.ceil((target - tolerance) * 4) - fixed_cost,
            math.floor((target + tolerance) * 4) - fixed_cost,
        )
        for target in targets
    }
    highest = max(upper for _, upper in shell_bounds.values())
    if highest > max_cost:
        return fallback("cost_limit")
    if highest < 0:
        return {
            target: DistanceEnumerationResult(
                target=target,
                cost=None,
                minimum_cost=None,
                mappings=[],
                distances=[],
                total_bijections=total,
                maximum_cost_upper_bound=maximum,
                visited_leaves=0,
                pruned_branches=0,
                elapsed_seconds=time.perf_counter() - started,
                status="no_solutions",
                complete=True,
                scope=(
                    "fixed_assignment_subspace"
                    if fixed
                    else "complete_atom_compatible_assignment_space"
                ),
                selected_mapping_count=0,
                symmetry_group_order=1,
                selected_labeled_mapping_count=0,
                backend="pabs_suffix_spectrum",
                backend_statistics={"shared_spectrum_prepared": False},
            )
            for target in targets
        }

    remaining = remaining_seconds()
    if remaining is not None and remaining <= 0:
        return {
            target: _timeout_result(
                target, total, maximum, time.perf_counter() - started, fixed
            )
            for target in targets
        }
    spectrum = SuffixCostSpectrum(
        a,
        b,
        residual_rows,
        residual_columns,
        unary,
        allowed,
        max_cost=highest,
        max_states=max_states,
        max_seconds=(
            min(float(max_seconds_per_prepare), remaining)
            if remaining is not None
            else float(max_seconds_per_prepare)
        ),
        deadline=deadline,
    )
    try:
        prepared = spectrum.prepare()
    except PropagationDeadline:
        return {
            target: _timeout_result(
                target, total, maximum, time.perf_counter() - started, fixed
            )
            for target in targets
        }
    if not prepared:
        return fallback("spectrum_resource_cap")

    results = {}
    for target in targets:
        query_started = time.perf_counter()
        lower, upper = shell_bounds[target]
        mappings, distances = [], []
        complete = True
        reason = None
        try:
            for residual_mapping, residual_cost in spectrum.mappings_between(
                lower, upper
            ):
                if max_mappings is not None and len(mappings) >= max_mappings:
                    complete = False
                    reason = "mapping_limit"
                    break
                mapping = [-1] * n
                for row, image in fixed.items():
                    mapping[row] = image
                for row, image in zip(residual_rows, residual_mapping):
                    mapping[row] = image
                mappings.append(mapping)
                distances.append((fixed_cost + residual_cost) / 4)
        except PropagationDeadline:
            complete = False
            reason = "time_limit"
        count = len(mappings)
        results[target] = DistanceEnumerationResult(
            target=target,
            cost=float(target) if count else None,
            minimum_cost=None,
            mappings=mappings,
            distances=distances,
            total_bijections=total,
            maximum_cost_upper_bound=maximum,
            visited_leaves=count,
            pruned_branches=0,
            elapsed_seconds=time.perf_counter() - query_started,
            status=(
                "timeout"
                if reason == "time_limit"
                else (
                    "complete"
                    if complete and count
                    else "no_solutions" if complete else "timeout"
                )
            ),
            complete=complete,
            truncation_reason=reason,
            scope=(
                "fixed_assignment_subspace"
                if fixed
                else "complete_atom_compatible_assignment_space"
            ),
            visited_nodes=spectrum.states,
            selected_mapping_count=count,
            symmetry_group_order=1,
            selected_labeled_mapping_count=count,
            backend="pabs_suffix_spectrum",
            backend_statistics={
                "shared_spectrum_prepared": True,
                "spectrum_states": spectrum.states,
                "spectrum_transitions": spectrum.transitions,
                "reachable_cost_count": len(spectrum.reachable_costs()),
                "preparation_seconds": time.perf_counter() - started,
            },
        )
    return results


__all__ = ["enumerate_pabs_shells"]
