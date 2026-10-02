"""Validate exact-distance options and normalize conditioned mapping inputs."""

import math
from numbers import Integral, Real

import numpy as np

from ..slap.lap import _adjacency_and_elements
from .distance_records import _input_sha256


def validate_distance_options(  # noqa: C901
    *,
    max_bijections,
    tolerance,
    time_limit_seconds,
    symmetry_pruning,
    expand_symmetry,
    certify,
    max_symmetry_automorphisms,
    symmetry_timeout_seconds,
    symmetry_max_search_nodes,
    max_mappings,
    assignment_lower_bound,
    assignment_upper_bound,
    atom_profile_pruning,
    seed_center_order,
    symmetry_node_properties,
    collect_mappings,
    mapping_callback,
    compute_minimum_cost,
    _optimization_only,
):
    """Reject invalid budgets and unsupported enumeration option combinations."""
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


def prepare_distance_problem(lgp, binary, fixed_mapping, certify, expand_symmetry):
    """Normalize graph matrices, fixed assignments, and input proof metadata."""
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
    fixed = normalize_fixed_mapping(
        fixed_mapping,
        atom_count,
        reactant_elements,
        product_elements,
        certify,
        expand_symmetry,
    )

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
    return (
        reactant,
        product,
        reactant_elements,
        product_elements,
        reactant_counts,
        product_by_element,
        atom_count,
        fixed,
        fixed_images,
        total_bijections,
        maximum_cost_upper_bound,
        input_sha256,
    )


def normalize_initial_mapping(
    initial_mapping, atom_count, reactant_elements, product_elements, fixed
):
    """Validate an optional complete atom-compatible incumbent permutation."""
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
    return preferred_mapping


def initial_distance_domains(
    reactant_order,
    reactant_elements,
    product_by_element,
    fixed_images,
    profile_edge_bounds,
    profile_domain_limit,
    tolerance,
):
    """Build the initial atom-compatible domains and count profile exclusions."""
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
    return domains, profile_domain_pruned


def supports_strict_improvement(
    _optimization_only, target, certify, maximum_cost_upper_bound, reactant, product
):
    """Allow tie rejection only in a proof pass with exact lattice costs."""
    return bool(
        _optimization_only
        and target == "minimal"
        and not certify
        and maximum_cost_upper_bound < 2**40
        and np.isfinite(reactant).all()
        and np.isfinite(product).all()
        and np.equal(reactant * 2, np.rint(reactant * 2)).all()
        and np.equal(product * 2, np.rint(product * 2)).all()
    )


def normalize_fixed_mapping(
    fixed_mapping,
    atom_count,
    reactant_elements,
    product_elements,
    certify,
    expand_symmetry,
):
    """Validate an injective, compatible set of pinned atom assignments."""
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

    return fixed
