"""Full-space minimum and distance-range adapters for optional C++ search."""

import math
import time
from collections import Counter
from functools import partial
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...slap.lap import _adjacency_and_elements
from ..distance_search.bounds import atom_profile_costs
from ..distance_search.records import (
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    normalize_distance_target,
)
from ..pabs.cost_lattice import CostLattice
from .candidates import (
    enumerate_native_range,
    minimize_native_candidates,
    prepare_native_candidates,
)


def _profile_seed(a, b, labels, product_labels):
    """Choose a compatible feasible incumbent; its cost is verified in C++."""
    profiles = atom_profile_costs(a, b, labels, product_labels)
    mapping = [-1] * len(labels)
    for element in dict.fromkeys(labels):
        rows = [i for i, value in enumerate(labels) if value == element]
        columns = [i for i, value in enumerate(product_labels) if value == element]
        left, right = linear_sum_assignment(profiles[np.ix_(rows, columns)])
        for i, j in zip(left, right):
            mapping[rows[i]] = columns[j]
    return mapping


def enumerate_cpp_mappings(  # noqa: C901
    lgp,
    *,
    library_path,
    CD="minimal",
    CD_mode="exact",
    binary=True,
    initial_mapping=None,
    max_bijections=1_000_000,
    max_mappings=None,
    time_limit_seconds=None,
    collect_mappings=True,
    mapping_callback=None,
    compute_minimum_cost=True,
    tolerance=1e-9,
    symmetry=False,
    node_properties=(),
    separator_config=None,
):
    """Prove PMCD and enumerate its full shell, or enumerate a supplied CD.

    A numeric query defaults to CD_mode='exact'. 'at_most' instead selects
    every distance at or below CD in one traversal. Returned distances and
    callbacks carry each mapping's actual cost.

    The default preserves every compatible labeled mapping. symmetry=True uses
    verified side automorphisms preserving node_properties; partial subgroup
    discovery only weakens pruning. Chemical frontends still perform exact ITS
    classification and report representative counts separately from labeled
    counts. Requires 1..256 atoms, symmetric loop-free half-integer weights and
    at most 16 bond levels. Compilation remains explicit.

    Minimum proof and complete enumeration are independent passes sharing one
    deadline. An interrupted proof never publishes its incumbent as PMCD.
    ``separator_config`` explicitly enables an experimental bounded residual
    oracle, with ``residual_limit`` (0..12) and ``max_states`` per call. Unknown
    oracle results retain ordinary search. The default leaves it disabled.
    """
    started = time.perf_counter()
    target = normalize_distance_target(CD)
    minimum_function = minimize_native_candidates
    range_function = enumerate_native_range
    separator_options = separator_statistics = None
    if separator_config is not None:
        from .separator import (
            _normalize_separator_config,
            enumerate_native_range_separator,
            minimize_native_candidates_separator,
        )

        separator_options = _normalize_separator_config(separator_config)
        minimum_function = partial(minimize_native_candidates_separator, **separator_options)
        range_function = partial(enumerate_native_range_separator, **separator_options)
        separator_statistics = dict.fromkeys(("calls", "known", "pruned", "states"), 0)
    if CD_mode not in {"exact", "at_most"}:
        raise ValueError("CD_mode must be 'exact' or 'at_most'")
    if target == "minimal" and CD_mode != "exact":
        raise ValueError("CD_mode='at_most' requires numeric CD")
    for name, value in (
        ("binary", binary),
        ("collect_mappings", collect_mappings),
        ("compute_minimum_cost", compute_minimum_cost),
        ("symmetry", symmetry),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    for name, value in (
        ("max_bijections", max_bijections),
        ("max_mappings", max_mappings),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and non-negative")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 0.25
    ):
        raise ValueError("C++ tolerance must be finite, nonnegative and less than 0.25")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable")
    a, labels = _adjacency_and_elements(lgp[0], binary)
    b, product_labels = _adjacency_and_elements(lgp[1], binary)
    if a.shape != b.shape or Counter(labels) != Counter(product_labels):
        raise ValueError("Endpoints must have equal atom-type inventories")
    total = math.prod(math.factorial(count) for count in Counter(labels).values())
    if max_bijections is not None and total > max_bijections:
        raise ExactEnumerationLimitError(
            f"{total:,} bijections exceed max_bijections={max_bijections:,}"
        )
    maximum = float(np.abs(a).sum() + np.abs(b).sum()) / 2
    if maximum > 250000:
        raise ValueError("C++ total distance range exceeds 250000")
    seed = (
        _profile_seed(a, b, labels, product_labels)
        if initial_mapping is None
        else initial_mapping
    )
    prepared = prepare_native_candidates(
        lgp,
        0,
        library_path=library_path,
        initial_mapping=seed,
        binary=binary,
        symmetry=symmetry,
        node_properties=node_properties,
        allow_partial_symmetry=True,
    )
    lattice = CostLattice.from_matrices(np.rint(a * 4), np.rint(b * 4))
    deadline = None if time_limit_seconds is None else started + time_limit_seconds
    nodes = leaves = pruned = count = 0
    minimum = reason = proof = None
    mappings, distances = [], []
    first_cost = None
    mixed_costs = False

    def remaining():
        return 1e100 if deadline is None else max(0, deadline - time.perf_counter())

    def record_statistics(record):
        nonlocal nodes, leaves, pruned
        nodes += int(record["visited_nodes"])
        leaves += int(record["visited_leaves"])
        pruned += int(record["pruned"])
        if separator_statistics is not None:
            for name in separator_statistics:
                separator_statistics[name] += record["separator_statistics"][name]

    if target == "minimal" or compute_minimum_cost:
        proof = minimum_function(
            lgp,
            library_path=library_path,
            initial_mapping=seed,
            time_limit_seconds=remaining(),
            _prepared=prepared,
        )
        record_statistics(proof)
        minimum = proof["minimum_cost"]
        reason = proof["reason"]

    def receive(mapping, cost):
        nonlocal count, first_cost, mixed_costs
        count += 1
        if first_cost is None:
            first_cost = cost
        elif cost != first_cost:
            mixed_costs = True
        if collect_mappings:
            mappings.append(list(mapping))
            distances.append(cost)
        if mapping_callback is not None:
            mapping_callback(list(mapping), cost)

    selected = minimum if target == "minimal" else float(target)
    if reason is None and selected is not None:
        if CD_mode == "at_most":
            lower = 0.0
            upper = min(
                maximum, math.floor((min(selected, maximum) + tolerance) * 2) / 2
            )
            reachable = lattice.intersects(0, round(upper * 4))
        elif selected > maximum + tolerance:
            reachable = False
        else:
            lower = upper = round(selected * 2) / 2
            reachable = (
                abs(selected - upper) <= tolerance
                and upper <= maximum
                and lattice.intersects(round(upper * 4), round(upper * 4))
            )
        if reachable:
            record = range_function(
                lgp,
                lower,
                upper,
                library_path=library_path,
                time_limit_seconds=remaining(),
                max_mappings=None if max_mappings is None else int(max_mappings),
                callback=receive,
                _prepared=prepared,
            )
            record_statistics(record)
            reason = record["reason"]
    complete = reason is None
    reactant_order, product_order = prepared[6:8]
    group_order = (
        reactant_order * product_order
        if reactant_order is not None and product_order is not None
        else None
    )
    labeled_count = count if count == 0 or not symmetry or group_order == 1 else None
    symmetry_complete = prepared.symmetry_search_complete
    generator_count = len(prepared[4]) + len(prepared[5]) - 2
    return DistanceEnumerationResult(
        target=target,
        cost=first_cost if count and not mixed_costs else None,
        minimum_cost=minimum,
        mappings=mappings,
        distances=distances,
        total_bijections=total,
        maximum_cost_upper_bound=maximum,
        visited_leaves=leaves,
        pruned_branches=pruned,
        visited_nodes=nodes,
        elapsed_seconds=time.perf_counter() - started,
        status=("complete" if count else "no_solutions") if complete else "timeout",
        complete=complete,
        truncation_reason=reason,
        scope=(
            "atom_compatible_assignment_space_modulo_verified_side_symmetry"
            if symmetry
            else "complete_atom_compatible_assignment_space"
        ),
        selected_mapping_count=count,
        selected_labeled_mapping_count=labeled_count,
        symmetry_automorphism_count=generator_count + 1,
        symmetry_search_complete=symmetry_complete,
        symmetry_group_order=group_order,
        symmetry_quotient_complete=complete and symmetry_complete,
        backend="pabs_cpp",
        backend_statistics={
            "method": "PABS",
            "requested_backend": "cpp",
            "implementation": "native_assignment_branch_and_bound",
            "minimum_shell_probes": 0,
            "minimum_optimization": proof,
            "CD_mode": CD_mode,
            "mapping_count_scope": (
                "symmetry_candidates" if symmetry else "labeled_mappings"
            ),
            "symmetry_enabled": symmetry,
            "side_groups_complete": symmetry_complete,
            "symmetry_generator_count": generator_count,
            "library_path": str(library_path),
            "cost_lattice_modulus_quarters": lattice.modulus,
            "cost_lattice_residue_quarters": lattice.residue,
            **(
                {"separator_config": separator_options,
                 "separator_statistics": separator_statistics}
                if separator_options is not None else {}
            ),
        },
    )


__all__ = ["enumerate_cpp_mappings"]
