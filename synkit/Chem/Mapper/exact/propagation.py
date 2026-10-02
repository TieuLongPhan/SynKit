"""Synister-CP: an opt-in cost-propagating exact graph-assignment engine.

The existing ``enumerate_distance_mappings`` remains the legacy engine. This
implementation has its own traversal, reversible AllDifferent domains, adaptive
conditioned bounds, and repaired integer assignment certificates. Inputs outside
its explicit lattice domain use the legacy solver with a recorded fallback.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from ..graph.automorphism import bounded_automorphism_permutations
from ..slap.lap import _adjacency_and_elements
from .distance import enumerate_distance_mappings
from .distance_records import (
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    normalize_distance_target,
)
from .propagation_search import PropagatedSearch
from .propagation_symmetry import bounded_generated_subgroup
from .propagation_limits import PropagationDeadline, check_deadline


@dataclass(frozen=True)
class PropagationConfig:
    """Independently selectable Synister-CP mechanisms for controlled comparisons."""

    domain_propagation: bool = True
    adaptive_bounds: bool = True
    incremental_assignments: bool = True
    cache_propagation: bool = True
    block_assignments: bool = True
    typed_mass_bounds: bool = True
    typed_transport_bounds: bool = True
    batch_forced_assignments: bool = True
    cost_lattice_pruning: bool = True
    # These incumbent/star experiments are optional: the paired 100-case
    # histogram+batch run showed better end-to-end performance without them.
    seed_local_search: bool = False
    seed_improvement_seconds: float = 0.05
    star_assignment_bounds: bool = False
    star_bound_residual_limit: int = 16
    star_bound_max_anchors: int = 32
    star_bound_seconds: float = 0.25
    pairwise_edge_bounds: bool = False
    pairwise_bound_residual_limit: int = 12
    pairwise_bound_max_anchors: int = 16
    pairwise_bound_seconds: float = 0.1
    column_cost_bounds: bool = True
    cycle_cost_bounds: bool = True
    # Experimental SR-GED style exact residual bound. It is opt-in until the
    # cohort benchmark demonstrates an end-to-end win.
    separator_bounds: bool = False
    separator_spectrum: bool = False
    separator_residual_limit: int = 10
    separator_max_size: int = 2
    separator_max_states: int = 20_000
    separator_spectrum_max_calls: int = 16
    # Factor-wise bitset support is a cheaper but weaker residual CD test.
    factor_spectrum_bounds: bool = False
    factor_spectrum_residual_limit: int = 10
    factor_spectrum_max_range: int = 4096
    factor_spectrum_max_calls: int = 16
    # Exact fixed-order suffix-state merging for small selected-CD residuals.
    suffix_spectrum: bool = False
    suffix_spectrum_orbit_pruning: bool = False
    suffix_spectrum_representation: str = "cost_table"
    suffix_spectrum_max_reward: int = 4096
    suffix_spectrum_residual_limit: int = 10
    suffix_spectrum_max_states: int = 20_000
    suffix_spectrum_max_cost: int = 4096
    suffix_spectrum_max_calls: int = 2
    suffix_spectrum_max_seconds_per_call: float = 0.002
    branch_order: str = "default"
    small_residual: int = 48
    bound_interval: int = 4
    bound_slack: int = 16

    def __post_init__(self):
        for name in (
            "domain_propagation",
            "adaptive_bounds",
            "incremental_assignments",
            "cache_propagation",
            "block_assignments",
            "typed_mass_bounds",
            "typed_transport_bounds",
            "batch_forced_assignments",
            "cost_lattice_pruning",
            "seed_local_search",
            "star_assignment_bounds",
            "column_cost_bounds",
            "cycle_cost_bounds",
            "separator_bounds",
            "separator_spectrum",
            "factor_spectrum_bounds",
            "suffix_spectrum",
            "suffix_spectrum_orbit_pruning",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be Boolean")
        for name in (
            "small_residual",
            "bound_interval",
            "bound_slack",
            "star_bound_residual_limit",
            "star_bound_max_anchors",
            "pairwise_bound_residual_limit",
            "pairwise_bound_max_anchors",
            "separator_residual_limit",
            "separator_max_size",
            "separator_max_states",
            "separator_spectrum_max_calls",
            "factor_spectrum_residual_limit",
            "factor_spectrum_max_range",
            "factor_spectrum_max_calls",
            "suffix_spectrum_residual_limit",
            "suffix_spectrum_max_states",
            "suffix_spectrum_max_cost",
            "suffix_spectrum_max_reward",
            "suffix_spectrum_max_calls",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(
            self.suffix_spectrum_representation, str
        ) or self.suffix_spectrum_representation not in {
            "cost_table",
            "reward_frontier",
        }:
            raise ValueError(
                "suffix_spectrum_representation must be cost_table or reward_frontier"
            )
        suffix_seconds = self.suffix_spectrum_max_seconds_per_call
        if (
            isinstance(suffix_seconds, bool)
            or not isinstance(suffix_seconds, Real)
            or not math.isfinite(suffix_seconds)
            or suffix_seconds <= 0
            or suffix_seconds > 0.1
        ):
            raise ValueError(
                "suffix_spectrum_max_seconds_per_call must be finite and in (0, 0.1]"
            )
        star_seconds = self.star_bound_seconds
        if (
            isinstance(star_seconds, bool)
            or not isinstance(star_seconds, Real)
            or not math.isfinite(star_seconds)
            or star_seconds < 0
            or star_seconds > 0.5
        ):
            raise ValueError("star_bound_seconds must be finite and in [0, 0.5]")
        seconds = self.seed_improvement_seconds
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, Real)
            or not math.isfinite(seconds)
            or seconds < 0
            or seconds > 0.1
        ):
            raise ValueError("seed_improvement_seconds must be finite and in [0, 0.1]")
        if not isinstance(self.branch_order, str) or self.branch_order not in {
            "default",
            "pagerank",
            "impact",
            "contention",
        }:
            raise ValueError(
                "branch_order must be 'default', 'pagerank', 'impact', or 'contention'"
            )
        pairwise_seconds = self.pairwise_bound_seconds
        if (
            isinstance(pairwise_seconds, bool)
            or not isinstance(pairwise_seconds, Real)
            or not math.isfinite(pairwise_seconds)
            or pairwise_seconds < 0
            or pairwise_seconds > 0.5
        ):
            raise ValueError("pairwise_bound_seconds must be finite and in [0, 0.5]")


def _validate_options(tolerance, seconds, caps, flags, callback, config):
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and nonnegative")
    if seconds is not None and (
        isinstance(seconds, bool)
        or not isinstance(seconds, Real)
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    for name, value in caps.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    for name, value in flags.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be Boolean")
    if callback is not None and not callable(callback):
        raise TypeError("mapping_callback must be callable or None")
    if not isinstance(config, PropagationConfig):
        raise TypeError("config must be a PropagationConfig")


def _fixed_and_seed(labels, product_labels, fixed_mapping, initial_mapping):
    n = len(labels)
    if fixed_mapping is not None and not hasattr(fixed_mapping, "items"):
        raise TypeError("fixed_mapping must be a mapping or None")
    fixed = dict(fixed_mapping or {})
    if any(
        isinstance(x, bool) or not isinstance(x, Integral)
        for pair in fixed.items()
        for x in pair
    ):
        raise TypeError("fixed_mapping indices must be integers")
    if any(not 0 <= i < n or not 0 <= j < n for i, j in fixed.items()):
        raise ValueError("fixed_mapping index out of range")
    if len(set(fixed.values())) != len(fixed) or any(
        labels[i] != product_labels[j] for i, j in fixed.items()
    ):
        raise ValueError("fixed_mapping must be injective and atom-compatible")
    if initial_mapping is None:
        seed = [-1] * n
        available = set(range(n)) - set(fixed.values())
        for i, j in fixed.items():
            seed[i] = int(j)
        for i in range(n):
            if i in fixed:
                continue
            j = next(j for j in sorted(available) if labels[i] == product_labels[j])
            seed[i] = j
            available.remove(j)
    else:
        seed = list(initial_mapping)
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in seed):
            raise TypeError("initial_mapping indices must be integers")
        if sorted(seed) != list(range(n)) or any(
            labels[i] != product_labels[j] for i, j in enumerate(seed)
        ):
            raise ValueError("initial_mapping must be an atom-compatible permutation")
        if any(seed[i] != j for i, j in fixed.items()):
            raise ValueError("initial_mapping must agree with fixed_mapping")
    return fixed, list(map(int, seed))


def _eligible(a, b, certify):
    if certify:
        return "legacy_prefix_certificate_requested"
    if len(a) > 256:
        return "order_above_256"
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return "nonfinite_weights"
    if (
        not np.array_equal(a, a.T)
        or not np.array_equal(b, b.T)
        or np.any(np.diag(a))
        or np.any(np.diag(b))
    ):
        return "directed_or_looped_graph"
    if (
        not np.equal(a * 2, np.rint(a * 2)).all()
        or not np.equal(b * 2, np.rint(b * 2)).all()
    ):
        return "weights_outside_half_integer_lattice"
    if 2 * (np.abs(a).sum() + np.abs(b).sum()) > 2**30:
        return "weights_above_exact_integer_range"
    return None


def _result(
    target,
    total,
    maximum,
    started,
    search,
    mappings,
    distances,
    minimum,
    complete,
    group,
    scope,
):
    count = search.selected
    cost = minimum if target == "minimal" else float(target) if count else None
    return DistanceEnumerationResult(
        target=target,
        cost=cost,
        minimum_cost=minimum,
        mappings=mappings,
        distances=distances,
        total_bijections=total,
        maximum_cost_upper_bound=maximum,
        visited_leaves=search.statistics["visited_leaves"],
        pruned_branches=sum(
            search.statistics[k]
            for k in ("lower_bound_pruned", "upper_bound_pruned", "hall_pruned")
        ),
        elapsed_seconds=time.perf_counter() - started,
        status="timeout" if not complete else "complete" if count else "no_solutions",
        complete=complete,
        truncation_reason=search.stop_reason,
        scope=scope,
        visited_nodes=search.statistics["visited_nodes"],
        symmetry_pruned_branches=search.statistics["symmetry_pruned"],
        lower_bound_pruned_branches=search.statistics["lower_bound_pruned"],
        upper_bound_pruned_branches=search.statistics["upper_bound_pruned"],
        selected_mapping_count=count,
        symmetry_group_order=len(group),
        symmetry_automorphism_count=len(group),
        selected_labeled_mapping_count=(
            count
            if search.expand
            else None if search.reactant_symmetry else count * len(group)
        ),
        backend="synister_cp",
        backend_statistics={"search": dict(search.statistics)},
    )


def _initialize_search(
    a,
    b,
    labels,
    product_labels,
    lgp,
    seed,
    fixed,
    config,
    deadline,
    *,
    binary,
    symmetry_pruning,
    reactant_symmetry_pruning,
    expand_symmetry,
    symmetry_node_properties,
    max_symmetry_automorphisms,
    symmetry_timeout_seconds,
    symmetry_max_search_nodes,
):
    """Budget symmetry discovery, subgroup closure and certified root setup."""
    check_deadline(deadline)
    group = (tuple(range(len(a))),)
    reactant_group = (tuple(range(len(a))),)
    discovered = True
    if symmetry_pruning:
        remaining = None if deadline is None else max(0, deadline - time.perf_counter())
        seconds = (
            math.inf if symmetry_timeout_seconds is None else symmetry_timeout_seconds
        )
        if remaining is not None:
            seconds = min(seconds, remaining)
        symmetry_deadline = deadline
        automorphism_seconds = seconds
        if math.isfinite(seconds):
            local_deadline = time.perf_counter() + seconds
            if symmetry_deadline is None or local_deadline < symmetry_deadline:
                symmetry_deadline = local_deadline
            automorphism_seconds = (
                0.4 * seconds if reactant_symmetry_pruning else 0.8 * seconds
            )
        permutations, discovered = bounded_automorphism_permutations(
            lgp[1],
            binary,
            limit=max_symmetry_automorphisms,
            timeout_seconds=automorphism_seconds,
            max_search_nodes=symmetry_max_search_nodes,
            node_properties=symmetry_node_properties,
        )
        group = bounded_generated_subgroup(
            permutations,
            max_order=max_symmetry_automorphisms,
            deadline=symmetry_deadline,
        )
        group = tuple(g for g in group if all(g[j] == j for j in fixed.values()))
    if reactant_symmetry_pruning:
        if not symmetry_pruning:
            raise ValueError("reactant symmetry pruning requires symmetry_pruning=True")
        remaining = None if deadline is None else max(0, deadline - time.perf_counter())
        seconds = (
            math.inf if symmetry_timeout_seconds is None else symmetry_timeout_seconds
        )
        if remaining is not None:
            seconds = min(seconds, remaining)
        automorphism_seconds = seconds
        if math.isfinite(seconds):
            automorphism_seconds = 0.4 * seconds
        permutations, source_discovered = bounded_automorphism_permutations(
            lgp[0],
            binary,
            limit=max_symmetry_automorphisms,
            timeout_seconds=automorphism_seconds,
            max_search_nodes=symmetry_max_search_nodes,
            node_properties=symmetry_node_properties,
        )
        reactant_group = bounded_generated_subgroup(
            permutations,
            max_order=max_symmetry_automorphisms,
            deadline=symmetry_deadline if symmetry_pruning else deadline,
        )
        reactant_group = tuple(
            g for g in reactant_group if all(g[i] == i for i in fixed)
        )
        discovered = discovered and source_discovered
    check_deadline(deadline)
    search = PropagatedSearch(
        np.rint(a * 4).astype(np.int64),
        np.rint(b * 4).astype(np.int64),
        labels,
        product_labels,
        seed,
        fixed,
        group,
        reactant_group,
        expand_symmetry,
        config,
        deadline,
    )
    check_deadline(deadline)
    return search, group, reactant_group, discovered


def _setup_timeout(target, total, maximum, started, scope):
    """Report an interrupted preprocessing phase without claiming infeasibility."""
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
        elapsed_seconds=time.perf_counter() - started,
        status="timeout",
        complete=False,
        truncation_reason="time_limit",
        scope=scope,
        backend="synister_cp",
        symmetry_search_complete=False,
        backend_statistics={
            "search": {"interrupted_phase": "preprocessing", "visited_nodes": 0}
        },
    )


def enumerate_synister_cp_mappings(  # noqa: C901
    lgp,
    *,
    CD="minimal",
    binary=True,
    initial_mapping=None,
    fixed_mapping=None,
    max_bijections=1_000_000,
    max_mappings=None,
    tolerance=1e-9,
    time_limit_seconds=None,
    symmetry_pruning=False,
    reactant_symmetry_pruning=False,
    expand_symmetry=False,
    symmetry_node_properties=(),
    max_symmetry_automorphisms=256,
    symmetry_timeout_seconds=0.25,
    symmetry_max_search_nodes=10_000,
    collect_mappings=True,
    mapping_callback=None,
    compute_minimum_cost=True,
    certify=False,
    config=None,
):
    """Enumerate exact weighted shells using the separate Synister-CP engine.

    The new path supports loop-free undirected half-integer matrices of order
    at most 256 in a bounded integer range. Generic graphs and legacy tree-cover
    certificates explicitly fall back to the unchanged legacy solver. Verified
    cyclic product subgroups provide unique expansion, including fixed subspaces.

    :param lgp: Reactant/product labeled graphs.
    :param CD: ``'minimal'`` or a nonnegative selected distance.
    :param config: Optional propagation and adaptive-bound controls.
    :return: Standard distance-enumeration result with explicit backend metadata.
    """
    started = time.perf_counter()
    target = normalize_distance_target(CD)
    config = PropagationConfig() if config is None else config
    flags = dict(
        binary=binary,
        symmetry_pruning=symmetry_pruning,
        expand_symmetry=expand_symmetry,
        collect_mappings=collect_mappings,
        compute_minimum_cost=compute_minimum_cost,
        certify=certify,
    )
    if not isinstance(reactant_symmetry_pruning, bool):
        raise TypeError("reactant_symmetry_pruning must be Boolean")
    if reactant_symmetry_pruning and not symmetry_pruning:
        raise ValueError("reactant_symmetry_pruning requires symmetry_pruning=True")
    caps = dict(
        max_bijections=max_bijections,
        max_mappings=max_mappings,
        max_symmetry_automorphisms=max_symmetry_automorphisms,
        symmetry_max_search_nodes=symmetry_max_search_nodes,
    )
    _validate_options(
        tolerance, time_limit_seconds, caps, flags, mapping_callback, config
    )
    if max_symmetry_automorphisms is None or symmetry_max_search_nodes is None:
        raise ValueError("Symmetry discovery caps must be positive integers")
    _validate_options(tolerance, symmetry_timeout_seconds, {}, {}, None, config)
    if expand_symmetry and not symmetry_pruning:
        raise ValueError("expand_symmetry requires symmetry_pruning")
    if isinstance(symmetry_node_properties, str):
        raise TypeError("symmetry_node_properties must be a sequence")
    a, labels = _adjacency_and_elements(lgp[0], binary)
    b, product_labels = _adjacency_and_elements(lgp[1], binary)
    if a.shape != b.shape or Counter(labels) != Counter(product_labels):
        raise ValueError("Endpoints must have equal atom-type inventories")
    fixed, seed = _fixed_and_seed(
        labels, product_labels, fixed_mapping, initial_mapping
    )
    reason = _eligible(a, b, certify)
    if reason is not None:
        result = enumerate_distance_mappings(
            lgp,
            CD=CD,
            initial_mapping=initial_mapping,
            fixed_mapping=fixed_mapping,
            tolerance=tolerance,
            time_limit_seconds=time_limit_seconds,
            mapping_callback=mapping_callback,
            symmetry_node_properties=symmetry_node_properties,
            symmetry_timeout_seconds=symmetry_timeout_seconds,
            **flags,
            **caps,
        )
        statistics = dict(result.backend_statistics or {})
        statistics["synister_cp_fallback"] = reason
        return replace(
            result, backend="synister_cp_legacy_fallback", backend_statistics=statistics
        )
    counts = Counter(labels[i] for i in range(len(labels)) if i not in fixed)
    total = math.prod(math.factorial(count) for count in counts.values())
    if max_bijections is not None and total > max_bijections:
        raise ExactEnumerationLimitError(
            f"{total:,} atom-compatible bijections exceed max_bijections={max_bijections:,}"
        )
    maximum = 0.5 * float(np.abs(a).sum() + np.abs(b).sum())
    deadline = None if time_limit_seconds is None else started + time_limit_seconds
    scope = (
        "fixed_assignment_subspace"
        if fixed
        else "complete_atom_compatible_assignment_space"
    )
    if symmetry_pruning and not expand_symmetry:
        scope = (
            "verified_reactant_product_orbit_representatives"
            if reactant_symmetry_pruning
            else "verified_cyclic_product_orbit_representatives"
        ) + ("_within_fixed_subspace" if fixed else "")
    try:
        search, group, reactant_group, discovered = _initialize_search(
            a,
            b,
            labels,
            product_labels,
            lgp,
            seed,
            fixed,
            config,
            deadline,
            binary=binary,
            symmetry_pruning=symmetry_pruning,
            reactant_symmetry_pruning=reactant_symmetry_pruning,
            expand_symmetry=expand_symmetry,
            symmetry_node_properties=symmetry_node_properties,
            max_symmetry_automorphisms=max_symmetry_automorphisms,
            symmetry_timeout_seconds=symmetry_timeout_seconds,
            symmetry_max_search_nodes=symmetry_max_search_nodes,
        )
    except PropagationDeadline:
        return _setup_timeout(target, total, maximum, started, scope)
    search.statistics["preprocessing_seconds"] = time.perf_counter() - started
    search.selected, search.stop_reason = 0, None
    minimum, complete = None, True
    if target == "minimal" or compute_minimum_cost:
        before = time.perf_counter()
        complete = search.prove()
        search.statistics["minimum_proof_seconds"] = time.perf_counter() - before
        search.statistics["minimum_proof_nodes"] = search.statistics["visited_nodes"]
        search.statistics["minimum_proof_status"] = (
            "proved" if complete else "time_limit"
        )
        if complete:
            minimum = search.incumbent / 4
    else:
        search.statistics["minimum_proof_seconds"] = None
        search.statistics["minimum_proof_nodes"] = None
        search.statistics["minimum_proof_status"] = "not_requested"
    search.statistics["incumbent_quarters"] = int(search.incumbent)
    search.statistics["root_lower_bound_quarters"] = int(search.root_lower)
    search.statistics["root_gap_quarters"] = int(search.incumbent - search.root_lower)
    mappings, distances = [], []

    def emit(mapping, cost):
        if collect_mappings:
            mappings.append(mapping)
            distances.append(cost)
        if mapping_callback is not None:
            mapping_callback(mapping, cost)

    enumeration_seconds = 0.0
    enumeration_nodes = 0
    if target != "minimal" or complete:
        selected = minimum if target == "minimal" else target
        lower, upper = math.ceil((selected - tolerance) * 4), math.floor(
            (selected + tolerance) * 4
        )
        before_nodes = search.statistics["visited_nodes"]
        enumeration_started = time.perf_counter()
        complete = search.enumerate(lower, upper, emit, max_mappings)
        enumeration_seconds = time.perf_counter() - enumeration_started
        enumeration_nodes = search.statistics["visited_nodes"] - before_nodes
    search.statistics["enumeration_seconds"] = enumeration_seconds
    search.statistics["enumeration_nodes"] = enumeration_nodes
    search.statistics["enumeration_status"] = (
        "complete"
        if enumeration_seconds and complete
        else "incomplete" if enumeration_seconds else "not_started"
    )
    scope = (
        "fixed_assignment_subspace"
        if fixed
        else "complete_atom_compatible_assignment_space"
    )
    if symmetry_pruning and not expand_symmetry:
        scope = (
            "verified_reactant_product_orbit_representatives"
            if reactant_symmetry_pruning
            else "verified_cyclic_product_orbit_representatives"
        ) + ("_within_fixed_subspace" if fixed else "")
    result = _result(
        target,
        total,
        maximum,
        started,
        search,
        mappings,
        distances,
        minimum,
        complete,
        group,
        scope,
    )
    return replace(result, symmetry_search_complete=discovered)
