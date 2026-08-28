"""Independent replay for exact chemical-distance certificates."""

from __future__ import annotations

import math

import numpy as np

from ..slap.lap import _adjacency_and_elements
from .distance_bounds import (
    blocked_assignment_extreme as _blocked_assignment_extreme,
    internal_cost_interval as _internal_cost_interval,
)
from .distance_records import (
    CertificateVerificationError,
    DistanceEnumerationCertificate,
    _certificate_sha256,
    _input_sha256,
    _mappings_sha256,
    normalize_distance_target,
)


def verify_distance_enumeration_certificate(  # noqa: C901
    lgp,
    certificate: DistanceEnumerationCertificate | dict[str, object],
    *,
    mappings=None,
) -> bool:
    """Independently replay a complete exact-CD certificate.

    The verifier reconstructs the atom-compatible decision tree, checks that
    terminal prefixes are prefix-free and cover every branch, and recomputes
    every terminal pruning bound without trusting search counters.
    """
    if isinstance(certificate, dict):
        certificate = DistanceEnumerationCertificate.from_dict(certificate)
    if not isinstance(certificate, DistanceEnumerationCertificate):
        raise TypeError("certificate must be a certificate object or dictionary")
    if certificate.schema_version not in {1, 2, 3, 4}:
        raise CertificateVerificationError("unsupported certificate schema")
    if _certificate_sha256(certificate.as_dict()) != certificate.certificate_sha256:
        raise CertificateVerificationError("certificate digest mismatch")

    reactant, reactant_elements = _adjacency_and_elements(lgp[0], certificate.binary)
    product, product_elements = _adjacency_and_elements(lgp[1], certificate.binary)
    if reactant.shape != product.shape:
        raise CertificateVerificationError("endpoint sizes differ")
    if (
        _input_sha256(
            reactant,
            product,
            reactant_elements,
            product_elements,
            certificate.binary,
        )
        != certificate.input_sha256
    ):
        raise CertificateVerificationError("certificate input binding mismatch")

    product_by_element: dict[object, tuple[int, ...]] = {}
    for index, element in enumerate(product_elements):
        product_by_element[element] = product_by_element.get(element, ()) + (index,)
    reactant_counts: dict[object, int] = {}
    for element in reactant_elements:
        reactant_counts[element] = reactant_counts.get(element, 0) + 1
    product_counts = {
        element: len(images) for element, images in product_by_element.items()
    }
    if reactant_counts != product_counts:
        raise CertificateVerificationError("atom-type multisets differ")
    expected_total = math.prod(
        math.factorial(count) for count in reactant_counts.values()
    )
    maximum_cost_upper_bound = 0.5 * float(
        np.abs(reactant).sum() + np.abs(product).sum()
    )
    if expected_total != certificate.total_bijections:
        raise CertificateVerificationError("bijection count mismatch")
    if not math.isclose(
        maximum_cost_upper_bound,
        certificate.maximum_cost_upper_bound,
        abs_tol=certificate.tolerance,
        rel_tol=0.0,
    ):
        raise CertificateVerificationError("maximum-CD bound mismatch")

    target = normalize_distance_target(certificate.target)
    if certificate.kind == "maximum_cost_upper_bound":
        if target == "minimal" or not (
            target > maximum_cost_upper_bound + certificate.tolerance
        ):
            raise CertificateVerificationError("invalid maximum-CD proof")
        if (
            certificate.terminal_prefixes
            or certificate.frontier_prefixes
            or certificate.symmetry_prefixes
            or certificate.lower_bound_prefixes
            or certificate.upper_bound_prefixes
            or certificate.selected_mapping_count
        ):
            raise CertificateVerificationError("upper-bound proof contains mappings")
        if certificate.cost is not None:
            raise CertificateVerificationError("upper-bound proof has a cost")
        if certificate.pruning_limit is not None:
            raise CertificateVerificationError("upper-bound proof has a pruning limit")
        if certificate.selected_mappings_sha256 != _mappings_sha256([]):
            raise CertificateVerificationError("empty mapping digest mismatch")
        if certificate.status != "no_solutions":
            raise CertificateVerificationError("upper-bound proof status mismatch")
        if mappings is not None and list(mappings):
            raise CertificateVerificationError("claimed empty shell has mappings")
        return True
    if certificate.kind != "prefix_cover":
        raise CertificateVerificationError("unknown certificate kind")
    valid_scopes = {
        "complete_atom_compatible_assignment_space",
        "verified_product_automorphism_lex_leaders",
        # Accepted for certificates emitted before the bounded-symmetry scope
        # was named precisely.
        "product_automorphism_orbit_representatives",
    }
    if certificate.mapping_scope not in valid_scopes:
        raise CertificateVerificationError("certificate mapping scope mismatch")
    symmetry_scopes = {
        "verified_product_automorphism_lex_leaders",
        "product_automorphism_orbit_representatives",
    }
    if certificate.symmetry_prefixes and certificate.mapping_scope not in (
        symmetry_scopes
    ):
        raise CertificateVerificationError("symmetry pruning requires quotient scope")
    if certificate.schema_version == 1 and certificate.mapping_scope != (
        "complete_atom_compatible_assignment_space"
    ):
        raise CertificateVerificationError("schema-1 scope mismatch")
    if target == "minimal":
        if not certificate.frontier_prefixes and (
            certificate.cost is None
            or certificate.pruning_limit is None
            or not math.isclose(
                certificate.cost,
                certificate.pruning_limit,
                abs_tol=certificate.tolerance,
                rel_tol=0.0,
            )
        ):
            raise CertificateVerificationError("invalid minimal pruning limit")
    elif certificate.pruning_limit is None or not math.isclose(
        certificate.pruning_limit,
        target,
        abs_tol=certificate.tolerance,
        rel_tol=0.0,
    ):
        raise CertificateVerificationError("numeric pruning limit mismatch")

    atom_count = len(reactant_elements)
    order = tuple(certificate.reactant_order)
    if sorted(order) != list(range(atom_count)):
        raise CertificateVerificationError("reactant order is not a permutation")
    trie: dict[object, object] = {}
    terminal = object()
    frontier = object()
    symmetry = object()
    lower_bound = object()
    upper_bound = object()

    def add_prefix(prefix, marker) -> None:
        node = trie
        if len(prefix) > atom_count:
            raise CertificateVerificationError("certificate prefix is too long")
        for image in prefix:
            if (
                terminal in node
                or frontier in node
                or symmetry in node
                or lower_bound in node
                or upper_bound in node
            ):
                raise CertificateVerificationError("certificate prefixes overlap")
            node = node.setdefault(int(image), {})
        if node:
            raise CertificateVerificationError("certificate prefixes overlap")
        node[marker] = True

    for prefix in certificate.terminal_prefixes:
        add_prefix(prefix, terminal)
    for prefix in certificate.frontier_prefixes:
        add_prefix(prefix, frontier)
    for prefix, witness in certificate.symmetry_prefixes:
        add_prefix(prefix, symmetry)
        node = trie
        for image in prefix:
            node = node[int(image)]
        node[symmetry] = tuple(witness)
    for prefix in certificate.lower_bound_prefixes:
        add_prefix(prefix, lower_bound)
    for prefix in certificate.upper_bound_prefixes:
        add_prefix(prefix, upper_bound)

    selected_mappings: list[list[int]] = []
    verified_symmetries: set[tuple[int, ...]] = set()

    def verify_symmetry(witness: tuple[int, ...]) -> None:
        if witness in verified_symmetries:
            return
        if sorted(witness) != list(range(atom_count)):
            raise CertificateVerificationError("invalid symmetry permutation")
        for image, transformed in enumerate(witness):
            if product_elements[image] != product_elements[transformed]:
                raise CertificateVerificationError("symmetry changes atom type")
        transformed_product = product[
            np.asarray(witness)[:, None], np.asarray(witness)[None, :]
        ]
        if not np.array_equal(product, transformed_product):
            raise CertificateVerificationError("symmetry changes product graph")
        verified_symmetries.add(witness)

    def prefix_cost(prefix: tuple[int, ...]) -> float:
        cost = 0.0
        for right_position, right_image in enumerate(prefix):
            right_atom = order[right_position]
            for left_position in range(right_position):
                left_atom = order[left_position]
                left_image = prefix[left_position]
                cost += 0.5 * abs(
                    float(reactant[right_atom, left_atom])
                    - float(product[right_image, left_image])
                )
                cost += 0.5 * abs(
                    float(reactant[left_atom, right_atom])
                    - float(product[left_image, right_image])
                )
        return cost

    def remaining_assignment_interval(prefix: tuple[int, ...]) -> tuple[float, float]:
        depth = len(prefix)
        remaining_atoms = list(order[depth:])
        used = set(prefix)
        remaining_products = [image for image in range(atom_count) if image not in used]
        costs = np.zeros((atom_count, atom_count), dtype=float)
        for reactant_atom in remaining_atoms:
            for product_atom in remaining_products:
                interaction = 0.0
                for position, assigned_product in enumerate(prefix):
                    assigned_reactant = order[position]
                    interaction += 0.5 * abs(
                        float(reactant[reactant_atom, assigned_reactant])
                        - float(product[product_atom, assigned_product])
                    )
                    interaction += 0.5 * abs(
                        float(reactant[assigned_reactant, reactant_atom])
                        - float(product[assigned_product, product_atom])
                    )
                costs[reactant_atom, product_atom] = interaction
        cross_lower = _blocked_assignment_extreme(
            costs,
            remaining_atoms,
            remaining_products,
            reactant_elements,
            product_elements,
        )
        cross_upper = _blocked_assignment_extreme(
            costs,
            remaining_atoms,
            remaining_products,
            reactant_elements,
            product_elements,
            maximize=True,
        )
        internal_lower, internal_upper = _internal_cost_interval(
            reactant,
            product,
            remaining_atoms,
            remaining_products,
        )
        return cross_lower + internal_lower, cross_upper + internal_upper

    def replay(node, prefix: tuple[int, ...], used: frozenset[int]) -> None:
        depth = len(prefix)
        if frontier in node:
            if len(node) != 1:
                raise CertificateVerificationError("frontier has descendants")
            return
        if symmetry in node:
            if len(node) != 1:
                raise CertificateVerificationError("symmetry prefix has descendants")
            witness = tuple(node[symmetry])
            verify_symmetry(witness)
            transformed = tuple(witness[image] for image in prefix)
            if not transformed < prefix:
                raise CertificateVerificationError("invalid symmetry lex leader")
            return
        if lower_bound in node:
            if len(node) != 1:
                raise CertificateVerificationError("lower-bound prefix has descendants")
            limit = certificate.pruning_limit
            bound, _ = remaining_assignment_interval(prefix)
            bound += prefix_cost(prefix)
            if limit is None or not bound > float(limit) + certificate.tolerance:
                raise CertificateVerificationError("invalid assignment lower bound")
            return
        if upper_bound in node:
            if len(node) != 1:
                raise CertificateVerificationError("upper-bound prefix has descendants")
            if target == "minimal":
                raise CertificateVerificationError(
                    "upper bound cannot prune minimization"
                )
            _, bound = remaining_assignment_interval(prefix)
            bound += prefix_cost(prefix)
            if not bound < float(target) - certificate.tolerance:
                raise CertificateVerificationError("invalid assignment upper bound")
            return
        if terminal in node:
            if len(node) != 1:
                raise CertificateVerificationError("terminal has descendants")
            cost = prefix_cost(prefix)
            if depth < atom_count:
                limit = certificate.pruning_limit
                if limit is None or not cost > float(limit) + certificate.tolerance:
                    raise CertificateVerificationError("invalid pruned-prefix bound")
                return
            mapping = [-1] * atom_count
            for position, image in enumerate(prefix):
                mapping[order[position]] = image
            selected = (
                certificate.cost is not None
                and math.isclose(
                    cost,
                    certificate.cost,
                    abs_tol=certificate.tolerance,
                    rel_tol=0.0,
                )
                if target == "minimal"
                else math.isclose(
                    cost,
                    target,
                    abs_tol=certificate.tolerance,
                    rel_tol=0.0,
                )
            )
            if selected:
                selected_mappings.append(mapping)
            return
        if depth == atom_count:
            raise CertificateVerificationError("uncovered complete assignment")
        atom = order[depth]
        expected_images = {
            image
            for image in product_by_element.get(reactant_elements[atom], ())
            if image not in used
        }
        actual_images = set(node)
        if actual_images != expected_images:
            raise CertificateVerificationError("terminal prefixes do not cover tree")
        for image in sorted(expected_images):
            replay(node[image], prefix + (image,), used | {image})

    replay(trie, (), frozenset())
    selected_digest = _mappings_sha256(selected_mappings)
    if len(selected_mappings) != certificate.selected_mapping_count:
        raise CertificateVerificationError("selected mapping count mismatch")
    if selected_digest != certificate.selected_mappings_sha256:
        raise CertificateVerificationError("selected mapping digest mismatch")
    expected_status = (
        "timeout"
        if certificate.frontier_prefixes
        else "complete" if selected_mappings else "no_solutions"
    )
    if certificate.status != expected_status:
        raise CertificateVerificationError("certificate status mismatch")
    if mappings is not None and _mappings_sha256(mappings) != selected_digest:
        raise CertificateVerificationError("provided mappings differ from certificate")
    return True


__all__ = ["verify_distance_enumeration_certificate"]
