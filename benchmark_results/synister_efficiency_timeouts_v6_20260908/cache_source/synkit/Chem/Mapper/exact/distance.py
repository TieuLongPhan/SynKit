"""Complete atom-compatible enumeration at an exact chemical distance."""

from __future__ import annotations

import math
import time
from numbers import Integral, Real

import numpy as np

from ..graph.automorphism import bounded_automorphism_permutations
from ..slap.lap import _adjacency_and_elements
from .distance_bounds import (
    ResidualBondMass as _ResidualBondMass,
    atom_profile_costs as _atom_profile_costs,
    assignment_edge_lower_bounds as _assignment_edge_lower_bounds,
    blocked_assignment_extreme as _blocked_assignment_extreme,
    conditioned_profile_assignment_lower_bound as _conditioned_profile_bound,
    element_blocks as _element_blocks,
    reaction_center_order as _reaction_center_order,
)
from .distance_records import (
    CertificateVerificationError,
    DistanceEnumerationCertificate,
    DistanceEnumerationResult,
    DistanceTarget,
    EnumerationStatus,
    ExactEnumerationLimitError,
    _input_sha256,
    _make_certificate,
    _mappings_sha256,
    normalize_distance_target,
)
from .distance_records import _certificate_sha256 as _certificate_sha256  # noqa: F401
from .distance_verify import verify_distance_enumeration_certificate
from .symmetry import (
    _MAX_STABILIZER_GENERATORS,
    _MAX_SCHREIER_WORK,
    largest_cyclic_subgroup,
    orbital_candidate_witnesses,
    permutation_group_order,
    point_stabilizer_generators_checked,
)


_MAX_CROSS_COST_CACHE_BYTES = 32 * 1024**2


def _mapping_cost(reactant, product, mapping) -> float:
    images = np.asarray(mapping, dtype=int)
    mapped_product = product[images[:, None], images[None, :]]
    return 0.5 * float(np.abs(reactant - mapped_product).sum())


def enumerate_distance_mappings(  # noqa: C901
    lgp,
    *,
    CD: str | Real = "minimal",
    binary: bool = True,
    max_bijections: int | None = 1_000_000,
    tolerance: float = 1e-9,
    time_limit_seconds: float | None = None,
    certify: bool = False,
    symmetry_pruning: bool = False,
    max_symmetry_automorphisms: int = 256,
    symmetry_timeout_seconds: float = 0.25,
    symmetry_max_search_nodes: int = 10_000,
    expand_symmetry: bool = False,
    initial_mapping=None,
    max_mappings: int | None = None,
    assignment_lower_bound: bool = True,
    assignment_upper_bound: bool = True,
    atom_profile_pruning: bool = True,
    seed_center_order: bool = True,
    symmetry_node_properties=(),
    fixed_mapping=None,
    collect_mappings: bool = True,
    mapping_callback=None,
    compute_minimum_cost: bool = True,
    _optimization_only: bool = False,
) -> DistanceEnumerationResult:
    """Enumerate the complete atom-compatible space at an exact CD.

    ``CD='minimal'`` returns every globally minimal labeled mapping. A numeric
    ``CD`` returns every labeled mapping whose chemical distance equals that
    value within ``tolerance``. The pre-search factorial cap makes failure
    explicit instead of returning a silently incomplete enumeration. With
    ``symmetry_pruning=True``, verified product-graph automorphisms remove
    non-lex-leading mappings and their witnesses become part of the replayable
    tree cover. Automorphism discovery is bounded, so multiple representatives
    of a full product-automorphism orbit may remain. ``collect_mappings=False``
    keeps only the exact selected count and optionally streams each final
    mapping to ``mapping_callback``. Streaming a minimal shell uses a separate
    optimization pass so provisional incumbents are never emitted.
    ``compute_minimum_cost=False`` skips the otherwise separate minimization
    pass for a numeric shell; this is useful when only that exact shell is
    required and leaves ``result.minimum_cost`` unset.
    Numeric uncertified searches also remove atom pairs whose element-blocked
    incident-bond profiles alone exceed the target.
    ``fixed_mapping`` may pin an injective, atom-compatible subset of reactant
    atoms to product atoms. The returned shell is complete within that
    conditioned subspace, and product symmetry is restricted to the pointwise
    stabilizer of the fixed product images.
    ``expand_symmetry=True`` restricts pruning to one fully enumerated cyclic
    verified subgroup and streams every group image of each representative;
    this recovers the complete labeled shell exactly once.
    """
    target = normalize_distance_target(CD)
    started = time.perf_counter()
    if max_bijections is not None and (
        isinstance(max_bijections, bool)
        or not isinstance(max_bijections, Integral)
        or max_bijections < 1
    ):
        raise ValueError("max_bijections must be a positive integer or None")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(float(time_limit_seconds))
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and non-negative")
    if not isinstance(symmetry_pruning, bool):
        raise TypeError("symmetry_pruning must be boolean")
    if not isinstance(expand_symmetry, bool):
        raise TypeError("expand_symmetry must be boolean")
    if expand_symmetry and not symmetry_pruning:
        raise ValueError("expand_symmetry=True requires symmetry_pruning=True")
    if expand_symmetry and certify:
        raise ValueError("symmetry expansion is not supported by certificates")
    if (
        isinstance(max_symmetry_automorphisms, bool)
        or not isinstance(max_symmetry_automorphisms, Integral)
        or max_symmetry_automorphisms < 1
    ):
        raise ValueError("max_symmetry_automorphisms must be a positive integer")
    if (
        isinstance(symmetry_timeout_seconds, bool)
        or not isinstance(symmetry_timeout_seconds, Real)
        or not math.isfinite(float(symmetry_timeout_seconds))
        or symmetry_timeout_seconds < 0
    ):
        raise ValueError("symmetry_timeout_seconds must be finite and non-negative")
    if (
        isinstance(symmetry_max_search_nodes, bool)
        or not isinstance(symmetry_max_search_nodes, Integral)
        or symmetry_max_search_nodes < 1
    ):
        raise ValueError("symmetry_max_search_nodes must be a positive integer")
    if max_mappings is not None and (
        isinstance(max_mappings, bool)
        or not isinstance(max_mappings, Integral)
        or max_mappings < 1
    ):
        raise ValueError("max_mappings must be a positive integer or None")
    if not isinstance(assignment_lower_bound, bool):
        raise TypeError("assignment_lower_bound must be boolean")
    if not isinstance(assignment_upper_bound, bool):
        raise TypeError("assignment_upper_bound must be boolean")
    if not isinstance(atom_profile_pruning, bool):
        raise TypeError("atom_profile_pruning must be boolean")
    if not isinstance(seed_center_order, bool):
        raise TypeError("seed_center_order must be boolean")
    if isinstance(symmetry_node_properties, str):
        raise TypeError("symmetry_node_properties must be a sequence of names")
    symmetry_node_properties = tuple(str(name) for name in symmetry_node_properties)
    if not isinstance(collect_mappings, bool):
        raise TypeError("collect_mappings must be boolean")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable or None")
    if not isinstance(compute_minimum_cost, bool):
        raise TypeError("compute_minimum_cost must be boolean")
    if certify and not collect_mappings:
        raise ValueError("certify=True currently requires collect_mappings=True")
    if not isinstance(_optimization_only, bool):
        raise TypeError("_optimization_only must be boolean")

    reactant, reactant_elements = _adjacency_and_elements(lgp[0], binary)
    product, product_elements = _adjacency_and_elements(lgp[1], binary)
    if reactant.shape != product.shape:
        raise ValueError("reactant and product must have the same number of atoms")

    reactant_counts: dict[object, int] = {}
    product_by_element: dict[object, list[int]] = {}
    for element in reactant_elements:
        reactant_counts[element] = reactant_counts.get(element, 0) + 1
    for index, element in enumerate(product_elements):
        product_by_element.setdefault(element, []).append(index)
    if reactant_counts != {
        element: len(indices) for element, indices in product_by_element.items()
    }:
        raise ValueError("reactant and product atom-type multisets differ")

    atom_count = len(reactant_elements)
    if fixed_mapping is None:
        fixed = {}
    elif hasattr(fixed_mapping, "items"):
        fixed = {}
        for atom, image in fixed_mapping.items():
            if (
                isinstance(atom, bool)
                or not isinstance(atom, Integral)
                or isinstance(image, bool)
                or not isinstance(image, Integral)
            ):
                raise TypeError("fixed_mapping indices must be integers")
            fixed[int(atom)] = int(image)
    else:
        raise TypeError("fixed_mapping must be a mapping or None")
    if any(atom < 0 or atom >= atom_count for atom in fixed):
        raise ValueError("fixed_mapping reactant index is out of range")
    if any(image < 0 or image >= atom_count for image in fixed.values()):
        raise ValueError("fixed_mapping product index is out of range")
    if len(set(fixed.values())) != len(fixed):
        raise ValueError("fixed_mapping product images must be unique")
    if any(
        reactant_elements[atom] != product_elements[image]
        for atom, image in fixed.items()
    ):
        raise ValueError("fixed_mapping must preserve atom types")
    if fixed and certify:
        raise ValueError("certificates do not yet support fixed_mapping")
    if fixed and expand_symmetry:
        raise ValueError("symmetry expansion does not yet support fixed_mapping")

    fixed_images = set(fixed.values())
    remaining_reactant_counts: dict[object, int] = {}
    remaining_product_counts: dict[object, int] = {}
    for atom, element in enumerate(reactant_elements):
        if atom not in fixed:
            remaining_reactant_counts[element] = (
                remaining_reactant_counts.get(element, 0) + 1
            )
    for image, element in enumerate(product_elements):
        if image not in fixed_images:
            remaining_product_counts[element] = (
                remaining_product_counts.get(element, 0) + 1
            )
    if remaining_reactant_counts != remaining_product_counts:
        raise ValueError("fixed_mapping leaves incompatible atom-type multisets")

    total_bijections = math.prod(
        math.factorial(count) for count in remaining_reactant_counts.values()
    )
    maximum_cost_upper_bound = 0.5 * float(
        np.abs(reactant).sum() + np.abs(product).sum()
    )
    input_sha256 = _input_sha256(
        reactant,
        product,
        reactant_elements,
        product_elements,
        binary,
    )
    if target != "minimal" and target > maximum_cost_upper_bound + tolerance:
        certificate = None
        if certify:
            certificate = _make_certificate(
                schema_version=1,
                kind="maximum_cost_upper_bound",
                input_sha256=input_sha256,
                target=target,
                binary=bool(binary),
                tolerance=float(tolerance),
                reactant_order=(),
                terminal_prefixes=(),
                frontier_prefixes=(),
                total_bijections=total_bijections,
                maximum_cost_upper_bound=maximum_cost_upper_bound,
                selected_mapping_count=0,
                selected_mappings_sha256=_mappings_sha256([]),
                cost=None,
                pruning_limit=None,
                status="no_solutions",
            )
        return DistanceEnumerationResult(
            target=target,
            cost=None,
            minimum_cost=None,
            mappings=[],
            distances=[],
            total_bijections=total_bijections,
            maximum_cost_upper_bound=maximum_cost_upper_bound,
            visited_leaves=0,
            pruned_branches=1,
            elapsed_seconds=time.perf_counter() - started,
            status="no_solutions",
            complete=True,
            certificate=certificate,
        )
    if max_bijections is not None and total_bijections > max_bijections:
        raise ExactEnumerationLimitError(
            f"{total_bijections:,} atom-compatible bijections exceed "
            f"max_bijections={max_bijections:,}"
        )

    proven_minimum_cost = None
    fixed_minimum_cost = None
    optimization = None
    optimization_seconds = 0.0
    needs_optimization_pass = (target != "minimal" and compute_minimum_cost) or (
        target == "minimal" and (not collect_mappings or mapping_callback is not None)
    )
    if needs_optimization_pass and not _optimization_only:
        optimization_started = time.perf_counter()
        optimization = enumerate_distance_mappings(
            lgp,
            CD="minimal",
            binary=binary,
            max_bijections=max_bijections,
            tolerance=tolerance,
            time_limit_seconds=time_limit_seconds,
            certify=False,
            symmetry_pruning=symmetry_pruning,
            max_symmetry_automorphisms=max_symmetry_automorphisms,
            symmetry_timeout_seconds=symmetry_timeout_seconds,
            symmetry_max_search_nodes=symmetry_max_search_nodes,
            expand_symmetry=expand_symmetry,
            initial_mapping=initial_mapping,
            max_mappings=None,
            assignment_lower_bound=assignment_lower_bound,
            assignment_upper_bound=assignment_upper_bound,
            atom_profile_pruning=atom_profile_pruning,
            seed_center_order=seed_center_order,
            symmetry_node_properties=symmetry_node_properties,
            fixed_mapping=fixed,
            collect_mappings=True,
            mapping_callback=None,
            compute_minimum_cost=True,
            _optimization_only=True,
        )
        optimization_seconds = time.perf_counter() - optimization_started
        if optimization.complete:
            proven_minimum_cost = optimization.minimum_cost
            if target == "minimal":
                fixed_minimum_cost = proven_minimum_cost
            if optimization.mappings:
                initial_mapping = optimization.mappings[0]
        elif target == "minimal":
            # Streaming must never expose provisional incumbents. If the
            # prerequisite optimization did not prove the minimum, return its
            # explicit incomplete result without starting the callback pass.
            return optimization

    preferred_mapping = None
    if initial_mapping is not None:
        preferred_mapping = [int(image) for image in initial_mapping]
        if sorted(preferred_mapping) != list(range(atom_count)):
            raise ValueError("initial_mapping must be a complete permutation")
        if any(
            reactant_elements[atom] != product_elements[image]
            for atom, image in enumerate(preferred_mapping)
        ):
            raise ValueError("initial_mapping must preserve atom types")
        if any(preferred_mapping[atom] != image for atom, image in fixed.items()):
            raise ValueError("initial_mapping must agree with fixed_mapping")
    reactant_order = _reaction_center_order(
        reactant,
        product,
        reactant_elements,
        reactant_counts,
        preferred_mapping if seed_center_order else None,
    )
    reactant_order = [atom for atom in reactant_order if atom not in fixed]
    profile_pair_costs = None
    if atom_profile_pruning and not certify:
        profile_pair_costs = _atom_profile_costs(
            reactant, product, reactant_elements, product_elements
        )
    profile_edge_bounds = profile_pair_costs
    profile_domain_limit = float(target) if target != "minimal" else fixed_minimum_cost
    if profile_domain_limit is None and preferred_mapping is not None:
        profile_domain_limit = _mapping_cost(reactant, product, preferred_mapping)
    if profile_pair_costs is not None and profile_domain_limit is not None:
        finite_profiles = profile_pair_costs[np.isfinite(profile_pair_costs)]
        # Quarter-unit weights keep alternating-path arithmetic exact here.
        # General floats retain the existing tolerance-aware individual bound.
        if (
            maximum_cost_upper_bound < 2**40
            and np.equal(finite_profiles * 4, np.rint(finite_profiles * 4)).all()
        ):
            profile_edge_bounds = _assignment_edge_lower_bounds(
                profile_pair_costs, reactant_elements, product_elements
            )
    domains = {}
    profile_domain_pruned = 0
    for atom in reactant_order:
        images = [
            image
            for image in product_by_element[reactant_elements[atom]]
            if image not in fixed_images
        ]
        if profile_edge_bounds is not None and profile_domain_limit is not None:
            before = len(images)
            images = [
                image
                for image in images
                if profile_edge_bounds[atom, image] <= profile_domain_limit + tolerance
            ]
            profile_domain_pruned += before - len(images)
        domains[atom] = tuple(images)
    # At an attained global row-profile bound every row attains its own bound.
    # Zero-cost rows must then match exactly. Restrict this propagation to exact
    # quarter-unit costs with tolerance smaller than the objective quantum.
    tight_profile_rows = bool(
        profile_pair_costs is not None
        and profile_domain_limit is not None
        and maximum_cost_upper_bound < 2**40
        and tolerance < 0.25
        and np.equal(reactant * 2, np.rint(reactant * 2)).all()
        and np.equal(product * 2, np.rint(product * 2)).all()
        and _blocked_assignment_extreme(
            profile_pair_costs,
            range(atom_count),
            range(atom_count),
            reactant_elements,
            product_elements,
        )
        == profile_domain_limit
    )

    dynamic_row_order = profile_pair_costs is not None and not certify
    exact_weight_arithmetic = (
        maximum_cost_upper_bound < 2**40
        and np.equal(reactant * 2, np.rint(reactant * 2)).all()
        and np.equal(product * 2, np.rint(product * 2)).all()
    )

    mapping_scope = (
        "fixed_assignment_subspace"
        if fixed
        else "complete_atom_compatible_assignment_space"
    )
    symmetry_permutations = (tuple(range(atom_count)),)
    symmetry_search_complete = True
    symmetry_group_order = 1
    symmetry_quotient_complete = True
    if symmetry_pruning:
        symmetry_permutations, symmetry_search_complete = (
            bounded_automorphism_permutations(
                lgp[1],
                binary,
                limit=int(max_symmetry_automorphisms),
                timeout_seconds=float(symmetry_timeout_seconds),
                max_search_nodes=int(symmetry_max_search_nodes),
                node_properties=symmetry_node_properties,
            )
        )
        root_generators = tuple(symmetry_permutations[1:])
        for image in sorted(fixed_images):
            root_generators, stabilizer_complete = point_stabilizer_generators_checked(
                root_generators, image
            )
            symmetry_quotient_complete &= stabilizer_complete
        if expand_symmetry:
            symmetry_permutations = largest_cyclic_subgroup(
                symmetry_permutations,
                max_order=max_symmetry_automorphisms,
            )
            root_generators = tuple(symmetry_permutations[1:])
            symmetry_group_order = len(symmetry_permutations)
        else:
            mapping_scope = (
                "verified_product_automorphism_lex_leaders_within_"
                "fixed_assignment_subspace"
                if fixed
                else "verified_product_automorphism_lex_leaders"
            )
            symmetry_group_order = permutation_group_order(root_generators)
            if symmetry_group_order is None:
                symmetry_quotient_complete = False
    else:
        root_generators = ()
    mapping = [-1] * atom_count
    used_products = [False] * atom_count
    for atom, image in fixed.items():
        mapping[atom] = image
        used_products[image] = True
    mappings: list[list[int]] = []
    distances: list[float] = []
    selected_mapping_count = 0
    if preferred_mapping is None:
        preferred_mapping = [-1] * atom_count
        for atom, image in fixed.items():
            preferred_mapping[atom] = image
        reactant_blocks = _element_blocks(reactant_order, reactant_elements)
        for element, atoms in reactant_blocks.items():
            available_images = [
                image
                for image in product_by_element[element]
                if image not in fixed_images
            ]
            for atom, image in zip(atoms, available_images):
                preferred_mapping[atom] = image
    best_cost = (
        fixed_minimum_cost
        if fixed_minimum_cost is not None
        else (
            _mapping_cost(reactant, product, preferred_mapping)
            if target == "minimal"
            else math.inf
        )
    )
    if _optimization_only and target == "minimal":
        mappings = [list(preferred_mapping)]
        distances = [float(best_cost)]
        selected_mapping_count = 1
    profile_compatible = None
    tight_profile_pruned = 0
    if tight_profile_rows:
        profile_compatible = np.zeros((atom_count, atom_count), dtype=bool)
        for atom, images in domains.items():
            profile_compatible[atom, list(images)] = True

    def tighten_profiles(atom, image):
        allowed = (profile_pair_costs != 0) | (
            reactant[:, atom, None] == product[:, image][None, :]
        )
        if profile_pair_costs[atom, image] == 0:
            allowed &= reactant[atom, :, None] == product[image, :][None, :]
        # Each entry is removed at most once on the live DFS path, so retained
        # flat indices use O(n^2) space rather than one dense mask per level.
        removed = np.flatnonzero(profile_compatible & ~allowed)
        profile_compatible.flat[removed] = False
        return removed

    if tight_profile_rows:
        for atom, image in fixed.items():
            tighten_profiles(atom, image)
    cross_costs = np.zeros((atom_count, atom_count), dtype=float)
    for atom, image in fixed.items():
        delta = 0.5 * np.abs(reactant[:, atom, None] - product[:, image][None, :])
        delta += 0.5 * np.abs(reactant[atom, :, None] - product[image, :][None, :])
        cross_costs += delta
    remaining_product_atoms = [
        image for image in range(atom_count) if image not in fixed_images
    ]
    residual_bond_mass = _ResidualBondMass(
        reactant,
        product,
        reactant_order,
        remaining_product_atoms,
    )
    # A dynamic row order needs reversible reactant mass, rather than suffix
    # sums tied to the original order. Reuse the matrix-side removal tracker.
    reactant_bond_mass = (
        _ResidualBondMass(reactant, reactant, reactant_order, reactant_order)
        if dynamic_row_order
        else None
    )
    initial_order_rank = {
        atom: position for position, atom in enumerate(reactant_order)
    }
    reactant_neighbors = (reactant != 0) | (reactant.T != 0)
    assigned_neighbors = np.zeros(atom_count, dtype=int)
    for atom in fixed:
        assigned_neighbors += reactant_neighbors[:, atom]
    fixed_atoms = sorted(fixed)
    fixed_product_atoms = [fixed[atom] for atom in fixed_atoms]
    initial_committed_cost = (
        0.5
        * float(
            np.abs(
                reactant[np.ix_(fixed_atoms, fixed_atoms)]
                - product[np.ix_(fixed_product_atoms, fixed_product_atoms)]
            ).sum()
        )
        if fixed_atoms
        else 0.0
    )
    search_count = len(reactant_order)
    visited_leaves = 0
    pruned_branches = 0
    deadline = (
        None if time_limit_seconds is None else started + float(time_limit_seconds)
    )
    timed_out = False
    truncation_reason = None
    terminal_prefixes: list[tuple[int, ...]] = []
    frontier_prefixes: list[tuple[int, ...]] = []
    symmetry_prefixes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    lower_bound_prefixes: list[tuple[int, ...]] = []
    upper_bound_prefixes: list[tuple[int, ...]] = []
    visited_nodes = 0
    symmetry_pruned_branches = 0
    lower_bound_pruned_branches = 0
    upper_bound_pruned_branches = 0
    stopped_with_unexplored_work = False
    # Half-integer bond weights give exact quarter-unit DFS costs and bounds
    # in this conservative range. Only the proof pass can discard ties.
    # Arbitrary floats retain tolerance-aware pruning; the shell stays inclusive.
    strict_improvement = bool(
        _optimization_only
        and target == "minimal"
        and not certify
        and maximum_cost_upper_bound < 2**40
        and np.isfinite(reactant).all()
        and np.isfinite(product).all()
        and np.equal(reactant * 2, np.rint(reactant * 2)).all()
        and np.equal(product * 2, np.rint(product * 2)).all()
    )

    def current_prefix(depth: int) -> tuple[int, ...]:
        return tuple(mapping[atom] for atom in reactant_order[:depth])

    profile_bound_calls = 0
    profile_bound_pruned = 0
    conditioned_bound_calls = 0
    conditioned_bound_pruned = 0

    def remaining_assignment_interval(
        depth: int, committed_cost: float, profile_cost: float
    ):
        nonlocal profile_bound_calls, profile_bound_pruned
        nonlocal conditioned_bound_calls, conditioned_bound_pruned
        remaining_atoms = reactant_order[depth:]
        branch_costs = None
        if not remaining_atoms:
            return 0.0, 0.0, None
        remaining_products = [
            product_atom
            for product_atom in range(atom_count)
            if not used_products[product_atom]
        ]
        cross_lower = _blocked_assignment_extreme(
            cross_costs,
            remaining_atoms,
            remaining_products,
            reactant_elements,
            product_elements,
        )
        if reactant_bond_mass is None:
            internal_lower, internal_upper = residual_bond_mass.interval(depth)
        else:
            internal_lower = 0.5 * abs(
                reactant_bond_mass.product_sum - residual_bond_mass.product_sum
            )
            internal_upper = 0.5 * (
                reactant_bond_mass.product_abs_sum + residual_bond_mass.product_abs_sum
            )
        lower = cross_lower + internal_lower
        if tight_profile_rows:
            # The attained root bound remains valid for every prefix. Edge
            # filtering and zero-row propagation already constrain this face;
            # avoid repeatedly recomputing its uninformative profile relaxations.
            lower = max(lower, profile_domain_limit - committed_cost)
        limit = float(best_cost if target == "minimal" else target)
        base_pruned = committed_cost + lower > limit + tolerance or (
            strict_improvement and committed_cost + lower >= limit
        )
        if (
            profile_pair_costs is not None
            and not base_pruned
            and not tight_profile_rows
        ):
            # The prefix's half-row profile sum plus a remaining LAP bounds
            # total CD. Combine by max, avoiding overlapping edge costs.
            profile_bound_calls += 1
            profile_lower = profile_cost + _blocked_assignment_extreme(
                profile_pair_costs,
                remaining_atoms,
                remaining_products,
                reactant_elements,
                product_elements,
            )
            lower = max(lower, profile_lower - committed_cost)
            if profile_lower > limit + tolerance or (
                strict_improvement and profile_lower >= limit
            ):
                profile_bound_pruned += 1
        # Condition the residual profiles only after cheaper bounds survive.
        # Limit the cubic profile work to modest remaining subproblems.
        already_pruned = committed_cost + lower > limit + tolerance or (
            strict_improvement and committed_cost + lower >= limit
        )
        if (
            profile_pair_costs is not None
            and not tight_profile_rows
            and not already_pruned
            and depth > 0
            and len(remaining_atoms) <= 256
        ):
            conditioned_bound_calls += 1
            conditioned_lower, branch_costs = _conditioned_profile_bound(
                reactant,
                product,
                cross_costs,
                remaining_atoms,
                remaining_products,
                reactant_elements,
                product_elements,
                return_costs=True,
            )
            lower = max(lower, conditioned_lower)
            if committed_cost + lower > limit + tolerance or (
                strict_improvement and committed_cost + lower >= limit
            ):
                conditioned_bound_pruned += 1
        if target == "minimal" or not assignment_upper_bound:
            return lower, None, branch_costs
        cross_upper = _blocked_assignment_extreme(
            cross_costs,
            remaining_atoms,
            remaining_products,
            reactant_elements,
            product_elements,
            maximize=True,
        )
        return lower, cross_upper + internal_upper, branch_costs

    cached_delta_bytes = 0
    peak_cached_delta_bytes = 0
    stabilizer_calls = 0

    def cross_cost_delta(atom: int, image: int):
        """Build a prefix-cost delta; retain it only within the cache budget."""
        delta = 0.5 * np.abs(reactant[:, atom, None] - product[:, image][None, :])
        delta += 0.5 * np.abs(reactant[atom, :, None] - product[image, :][None, :])
        return delta

    def visit(
        depth: int,
        committed_cost: float,
        stabilizer_generators=(),
        pending_point=None,
        profile_cost=0.0,
    ) -> None:
        nonlocal best_cost, visited_leaves, pruned_branches, mappings, distances
        nonlocal selected_mapping_count
        nonlocal timed_out, truncation_reason, visited_nodes
        nonlocal symmetry_pruned_branches, lower_bound_pruned_branches
        nonlocal upper_bound_pruned_branches, stopped_with_unexplored_work
        nonlocal symmetry_quotient_complete, stabilizer_calls
        nonlocal cached_delta_bytes, peak_cached_delta_bytes
        nonlocal tight_profile_pruned

        visited_nodes += 1

        if timed_out:
            stopped_with_unexplored_work = True
            if certify:
                frontier_prefixes.append(current_prefix(depth))
            return
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            truncation_reason = "time_limit"
            if certify:
                frontier_prefixes.append(current_prefix(depth))
            return

        limit = best_cost if target == "minimal" else target
        if committed_cost > float(limit) + tolerance or (
            strict_improvement and committed_cost >= float(limit)
        ):
            pruned_branches += 1
            if certify:
                terminal_prefixes.append(current_prefix(depth))
            return
        branch_costs = None
        if assignment_lower_bound and depth < search_count:
            lower_bound, upper_bound, branch_costs = remaining_assignment_interval(
                depth, committed_cost, profile_cost
            )
            if committed_cost + lower_bound > float(limit) + tolerance or (
                strict_improvement and committed_cost + lower_bound >= float(limit)
            ):
                lower_bound_pruned_branches += 1
                if certify:
                    lower_bound_prefixes.append(current_prefix(depth))
                return
            if (
                upper_bound is not None
                and target != "minimal"
                and committed_cost + upper_bound < float(target) - tolerance
            ):
                upper_bound_pruned_branches += 1
                if certify:
                    upper_bound_prefixes.append(current_prefix(depth))
                return
        if depth == search_count:
            visited_leaves += 1
            if certify:
                terminal_prefixes.append(current_prefix(depth))
            cost = float(committed_cost)
            if cost < best_cost:
                best_cost = cost
                if target == "minimal":
                    if _optimization_only:
                        mappings = [list(mapping)]
                        distances = [cost]
                    else:
                        retained = [
                            (candidate, distance)
                            for candidate, distance in zip(mappings, distances)
                            if distance <= best_cost + tolerance
                        ]
                        mappings = [candidate for candidate, _ in retained]
                        distances = [distance for _, distance in retained]
                        selected_mapping_count = len(mappings)
            selected = (
                cost <= best_cost + tolerance
                if target == "minimal"
                else math.isclose(cost, target, abs_tol=tolerance, rel_tol=0.0)
            )
            if selected and not _optimization_only:
                expanded = (
                    (
                        [permutation[image] for image in mapping]
                        for permutation in symmetry_permutations
                    )
                    if expand_symmetry
                    else (list(mapping) for _ in range(1))
                )
                for selected_mapping in expanded:
                    selected_mapping_count += 1
                    if collect_mappings:
                        mappings.append(selected_mapping)
                        distances.append(cost)
                    if mapping_callback is not None:
                        mapping_callback(selected_mapping, cost)
                    if (
                        max_mappings is not None
                        and selected_mapping_count >= max_mappings
                    ):
                        timed_out = True
                        truncation_reason = "mapping_limit"
                        break
            return

        # Bounds and leaves need no child stabilizer. Construct it only for
        # a surviving internal node, where candidate orbit pruning uses it.
        if symmetry_pruning and pending_point is not None:
            # These generators come from the verified, deduplicated root set
            # or a completed stabilizer construction. Preserve the same tuple
            # when the subgroup already fixes this point, within helper budgets.
            if not (
                len(stabilizer_generators)
                <= min(_MAX_STABILIZER_GENERATORS, _MAX_SCHREIER_WORK)
                and all(
                    g[pending_point] == pending_point for g in stabilizer_generators
                )
            ):
                stabilizer_calls += 1
                stabilizer_generators, stabilizer_complete = (
                    point_stabilizer_generators_checked(
                        stabilizer_generators, pending_point
                    )
                )
                symmetry_quotient_complete &= stabilizer_complete

        selected_position = depth
        branch_allowed = None
        if dynamic_row_order:
            remaining_rows = reactant_order[depth:]
            remaining_images = np.flatnonzero(~np.asarray(used_products))
            if tight_profile_rows:
                counts = (
                    profile_compatible[remaining_rows] & ~np.asarray(used_products)
                ).sum(axis=1)
            elif branch_costs is not None:
                finite_costs = branch_costs[np.isfinite(branch_costs)]
                if (
                    maximum_cost_upper_bound < 2**40
                    and np.equal(finite_costs * 4, np.rint(finite_costs * 4)).all()
                ):
                    forced_bounds = committed_cost + _assignment_edge_lower_bounds(
                        branch_costs,
                        [reactant_elements[atom] for atom in remaining_rows],
                        [product_elements[image] for image in remaining_images],
                    )
                else:
                    row_minima = branch_costs.min(axis=1)
                    forced_bounds = (
                        committed_cost + branch_costs
                        + (row_minima.sum() - row_minima)[:, None]
                    )
                branch_allowed = forced_bounds <= float(limit) + tolerance
                if (
                    tolerance < 0.25
                    and exact_weight_arithmetic
                    and np.equal(finite_costs * 4, np.rint(finite_costs * 4)).all()
                ):
                    internal_profiles = branch_costs - cross_costs[
                        np.ix_(remaining_rows, remaining_images)
                    ]
                    exact_rows = (internal_profiles == 0) & (
                        forced_bounds == float(limit)
                    )
                    if exact_rows.any():
                        a_block = reactant[np.ix_(remaining_rows, remaining_rows)]
                        b_block = product[np.ix_(remaining_images, remaining_images)]
                        left, right = np.nonzero(a_block)
                        values = np.unique(a_block[left, right])
                        adjacency = {value: (b_block == value) for value in values}
                        for _ in range(8):
                            previous = branch_allowed.copy()
                            supports = {
                                value: (previous.astype(np.int32) @ edges.T) > 0
                                for value, edges in adjacency.items()
                            }
                            for row, neighbor in zip(left, right):
                                branch_allowed[row] &= (
                                    ~exact_rows[row]
                                    | supports[a_block[row, neighbor]][neighbor]
                                )
                            if np.array_equal(previous, branch_allowed):
                                break
                counts = branch_allowed.sum(axis=1)
            else:
                counts = np.asarray([
                    sum(
                        not used_products[image]
                        and committed_cost + cross_costs[atom, image]
                        <= float(limit) + tolerance
                        for image in domains[atom]
                    )
                    for atom in remaining_rows
                ])
            selected_position = depth + min(
                range(len(remaining_rows)),
                key=lambda position: (
                    int(counts[position]),
                    -int(assigned_neighbors[remaining_rows[position]]),
                    initial_order_rank[remaining_rows[position]],
                ),
            )
            reactant_order[depth], reactant_order[selected_position] = (
                reactant_order[selected_position],
                reactant_order[depth],
            )
        reactant_atom = reactant_order[depth]
        allowed_images = (
            set(remaining_images[branch_allowed[selected_position - depth]])
            if branch_allowed is not None else None
        )
        candidate_images = sorted(
            (
                image
                for image in domains[reactant_atom]
                if not used_products[image]
                and (allowed_images is None or image in allowed_images)
                and (
                    profile_compatible is None
                    or profile_compatible[reactant_atom, image]
                )
            ),
            key=lambda product_atom: (
                product_atom != preferred_mapping[reactant_atom],
                cross_costs[reactant_atom, product_atom],
                product_atom,
            ),
        )
        symmetry_witnesses = (
            orbital_candidate_witnesses(candidate_images, stabilizer_generators)
            if symmetry_pruning
            else {image: None for image in candidate_images}
        )
        for product_atom in candidate_images:
            # Certificate searches still need to record frontier prefixes.
            if timed_out and truncation_reason == "time_limit" and not certify:
                stopped_with_unexplored_work = True
                break
            symmetry_witness = symmetry_witnesses[product_atom]
            if symmetry_witness is not None:
                symmetry_pruned_branches += 1
                if certify:
                    prefix = current_prefix(depth) + (product_atom,)
                    symmetry_prefixes.append((prefix, symmetry_witness))
                continue
            incremental = float(cross_costs[reactant_atom, product_atom])
            mapping[reactant_atom] = product_atom
            used_products[product_atom] = True
            removed_profiles = None
            if tight_profile_rows:
                removed_profiles = tighten_profiles(reactant_atom, product_atom)
                remaining_rows = reactant_order[depth + 1 :]
                available = profile_compatible[remaining_rows] & ~np.asarray(
                    used_products
                )
                if not available.any(axis=1).all():
                    tight_profile_pruned += 1
                    profile_compatible.flat[removed_profiles] = True
                    used_products[product_atom] = False
                    mapping[reactant_atom] = -1
                    continue
            removed_bond_mass = residual_bond_mass.remove(product_atom)
            removed_reactant_mass = None
            if dynamic_row_order:
                removed_reactant_mass = reactant_bond_mass.remove(reactant_atom)
                assigned_neighbors[:] += reactant_neighbors[:, reactant_atom]
            delta = cross_cost_delta(reactant_atom, product_atom)
            cross_costs[:] += delta
            retained = cached_delta_bytes + delta.nbytes <= _MAX_CROSS_COST_CACHE_BYTES
            if retained:
                cached_delta_bytes += delta.nbytes
                peak_cached_delta_bytes = max(
                    peak_cached_delta_bytes, cached_delta_bytes
                )
            else:
                delta = None
            visit(
                depth + 1,
                committed_cost + incremental,
                stabilizer_generators,
                product_atom,
                profile_cost
                + (
                    float(profile_pair_costs[reactant_atom, product_atom])
                    if profile_pair_costs is not None
                    else 0.0
                ),
            )
            if retained:
                cached_delta_bytes -= delta.nbytes
            else:
                delta = cross_cost_delta(reactant_atom, product_atom)
            cross_costs[:] -= delta
            del delta
            residual_bond_mass.restore(product_atom, removed_bond_mass)
            if removed_reactant_mass is not None:
                reactant_bond_mass.restore(reactant_atom, removed_reactant_mass)
                assigned_neighbors[:] -= reactant_neighbors[:, reactant_atom]
            if removed_profiles is not None:
                profile_compatible.flat[removed_profiles] = True
            used_products[product_atom] = False
            mapping[reactant_atom] = -1
        if dynamic_row_order:
            reactant_order[depth], reactant_order[selected_position] = (
                reactant_order[selected_position],
                reactant_order[depth],
            )

    traversal_started = time.perf_counter()
    initial_profile_cost = (
        sum(float(profile_pair_costs[atom, image]) for atom, image in fixed.items())
        if profile_pair_costs is not None
        else 0.0
    )
    visit(0, initial_committed_cost, root_generators, profile_cost=initial_profile_cost)
    traversal_seconds = time.perf_counter() - traversal_started
    if (
        timed_out
        and truncation_reason == "mapping_limit"
        and not stopped_with_unexplored_work
    ):
        timed_out = False
        truncation_reason = None
    elapsed_seconds = time.perf_counter() - started
    if timed_out and target == "minimal" and fixed_minimum_cost is None:
        # Never expose a best-so-far mapping as a proven exact minimum.
        mappings = []
        distances = []
        selected_mapping_count = 0
        minimum_cost = None
        selected_cost = None
    else:
        minimum_cost = (
            proven_minimum_cost
            if target != "minimal"
            else None
            if math.isinf(best_cost)
            else float(best_cost)
        )
        selected_cost = (
            minimum_cost
            if target == "minimal"
            else (float(target) if selected_mapping_count else None)
        )
    complete = not timed_out
    selected_labeled_mapping_count = selected_mapping_count
    if symmetry_pruning and not expand_symmetry:
        selected_labeled_mapping_count = (
            selected_mapping_count * symmetry_group_order
            if symmetry_group_order is not None and symmetry_quotient_complete
            else None
        )
    status: EnumerationStatus = (
        "timeout"
        if timed_out
        else "complete"
        if selected_mapping_count
        else "no_solutions"
    )
    certificate = None
    if certify:
        certificate = _make_certificate(
            schema_version=(
                4
                if upper_bound_prefixes
                else 3
                if assignment_lower_bound
                else 2
                if symmetry_pruning
                else 1
            ),
            kind="prefix_cover",
            input_sha256=input_sha256,
            target=target,
            binary=bool(binary),
            tolerance=float(tolerance),
            reactant_order=tuple(reactant_order),
            terminal_prefixes=tuple(terminal_prefixes),
            frontier_prefixes=tuple(frontier_prefixes),
            total_bijections=total_bijections,
            maximum_cost_upper_bound=maximum_cost_upper_bound,
            selected_mapping_count=selected_mapping_count,
            selected_mappings_sha256=_mappings_sha256(mappings),
            cost=selected_cost,
            pruning_limit=(
                None
                if target == "minimal" and math.isinf(best_cost)
                else float(best_cost)
                if target == "minimal"
                else float(target)
            ),
            status=status,
            mapping_scope=mapping_scope,
            symmetry_prefixes=tuple(symmetry_prefixes),
            lower_bound_prefixes=tuple(lower_bound_prefixes),
            upper_bound_prefixes=tuple(upper_bound_prefixes),
        )
    return DistanceEnumerationResult(
        target=target,
        cost=selected_cost,
        minimum_cost=minimum_cost,
        mappings=mappings,
        distances=distances,
        total_bijections=total_bijections,
        maximum_cost_upper_bound=maximum_cost_upper_bound,
        visited_leaves=visited_leaves,
        pruned_branches=pruned_branches,
        elapsed_seconds=elapsed_seconds,
        status=status,
        complete=complete,
        truncation_reason=truncation_reason,
        scope=mapping_scope,
        certificate=certificate,
        visited_nodes=visited_nodes,
        symmetry_pruned_branches=symmetry_pruned_branches,
        symmetry_automorphism_count=len(symmetry_permutations),
        symmetry_search_complete=symmetry_search_complete,
        lower_bound_pruned_branches=lower_bound_pruned_branches,
        upper_bound_pruned_branches=upper_bound_pruned_branches,
        selected_mapping_count=selected_mapping_count,
        symmetry_group_order=symmetry_group_order,
        symmetry_quotient_complete=symmetry_quotient_complete,
        selected_labeled_mapping_count=selected_labeled_mapping_count,
        backend_statistics={
            "search": {
                "phase": "minimum_proof" if _optimization_only else "shell_enumeration",
                "strict_improvement": strict_improvement,
                "stabilizer_calls": stabilizer_calls,
                "profile_bound_calls": profile_bound_calls,
                "profile_domain_pruned": profile_domain_pruned,
                "tight_profile_rows": tight_profile_rows,
                "dynamic_profile_order": dynamic_row_order,
                "tight_profile_pruned": tight_profile_pruned,
                "profile_bound_pruned": profile_bound_pruned,
                "conditioned_bound_calls": conditioned_bound_calls,
                "conditioned_bound_pruned": conditioned_bound_pruned,
                "peak_cached_delta_bytes": peak_cached_delta_bytes,
                "incumbent_cost": (
                    float(best_cost)
                    if target == "minimal" and math.isfinite(best_cost)
                    else None
                ),
                "preprocessing_seconds": traversal_started
                - started
                - optimization_seconds,
                "traversal_seconds": traversal_seconds,
                "minimum_proof_seconds": optimization_seconds,
                "minimum_proof_nodes": optimization.visited_nodes
                if optimization is not None
                else None,
            }
        },
    )


__all__ = [
    "CertificateVerificationError",
    "DistanceEnumerationCertificate",
    "DistanceEnumerationResult",
    "DistanceTarget",
    "EnumerationStatus",
    "ExactEnumerationLimitError",
    "enumerate_distance_mappings",
    "normalize_distance_target",
    "verify_distance_enumeration_certificate",
]
