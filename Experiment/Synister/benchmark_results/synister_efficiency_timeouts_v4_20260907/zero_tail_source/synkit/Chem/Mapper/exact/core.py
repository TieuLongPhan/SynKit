"""Reference-ITS component reduction for staged exact AAM enumeration.

The reduction is intentionally explicit about scope.  Given a *known* mapped
reaction (or any trusted seed mapping), it keeps every connected component of
the union ITS that contains a changed heavy-atom bond, H-count change, or charge
change.  Completely unchanged disconnected components are fixed to the seed
mapping and can subsequently be represented by endpoint automorphism orbits.

This is exact for the seed-conditioned active-component space and is useful for
reference-AAM reproduction.  It is not an unrestricted proof over mappings that
exchange atoms between active and inactive components; callers must not label it
as the complete atom-compatible CD shell.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..graph.labeled_graph import LabeledGraph
from ..slap.lap import _adjacency_and_elements


def _node_property(lg, name, default=0):
    values = lg.props.get(name)
    if values is None:
        return [default] * len(lg.labels)
    if len(values) != len(lg.labels):
        raise ValueError(f"{name} must contain one value per graph atom")
    return list(values)


def _induced_labeled_graph(lg, nodes):
    nodes = tuple(nodes)
    position = {node: index for index, node in enumerate(nodes)}
    graph = {index: {} for index in range(len(nodes))}
    for source in nodes:
        for target, weight in lg.graph.get(source, {}).items():
            if target in position:
                graph[position[source]][position[target]] = weight
    reduced = LabeledGraph(graph, [lg.labels[node] for node in nodes])
    for name in ("atomic numbers", "hcounts", "charges"):
        values = lg.props.get(name)
        if values is not None:
            reduced.set_prop(name, [values[node] for node in nodes])
    return reduced


def _union_components(reactant, mapped_product):
    size = reactant.shape[0]
    neighbours = [set() for _ in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            if reactant[left, right] != 0 or mapped_product[left, right] != 0:
                neighbours[left].add(right)
                neighbours[right].add(left)
    components = []
    unseen = set(range(size))
    while unseen:
        root = min(unseen)
        component = set()
        pending = [root]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(neighbours[node] - component)
        unseen.difference_update(component)
        components.append(frozenset(component))
    return tuple(components)


@dataclass(frozen=True)
class ITSComponentReduction:
    """A seed-conditioned active ITS problem and its exact lifting map."""

    lgp: tuple[LabeledGraph, LabeledGraph]
    reference_mapping: tuple[int, ...]
    reactant_atoms: tuple[int, ...]
    product_atoms: tuple[int, ...]
    inactive_atoms: tuple[int, ...]
    fixed_mapping: tuple[int, ...]
    changed_atoms: tuple[int, ...]
    heavy_distance: float
    hydrogen_distance: int
    scope: str = "reference_active_its_component_space"

    def lift(self, core_mapping) -> list[int]:
        """Lift one reduced mapping while retaining the seed outside the ITS core."""
        core_mapping = tuple(int(image) for image in core_mapping)
        size = len(self.reactant_atoms)
        if sorted(core_mapping) != list(range(size)):
            raise ValueError("core_mapping must be a complete reduced permutation")
        lifted = list(self.fixed_mapping)
        for position, reactant_atom in enumerate(self.reactant_atoms):
            lifted[reactant_atom] = self.product_atoms[core_mapping[position]]
        return lifted


def reduce_to_reference_its_components(
    lgp,
    reference_mapping,
    *,
    binary=False,
    tolerance=1e-9,
) -> ITSComponentReduction:
    """Keep union-ITS components changed by a trusted reference mapping.

    Hydrogens remain implicit.  Per-heavy-atom H counts are carried in the
    reduced graph metadata and H-count changes participate in reaction-center
    selection.  ``hydrogen_distance`` is the L1 H-count discrepancy induced by
    the seed heavy-atom mapping; it is the minimum explicit H--parent bond-edit
    contribution when the two hydrogen inventories are balanced.
    """
    reactant, reactant_elements = _adjacency_and_elements(lgp[0], binary)
    product, product_elements = _adjacency_and_elements(lgp[1], binary)
    if reactant.shape != product.shape:
        raise ValueError("reactant and product must contain the same atom count")
    size = reactant.shape[0]
    mapping = tuple(int(image) for image in reference_mapping)
    if sorted(mapping) != list(range(size)):
        raise ValueError("reference_mapping must be a complete permutation")
    if any(
        reactant_elements[atom] != product_elements[image]
        for atom, image in enumerate(mapping)
    ):
        raise ValueError("reference_mapping must preserve atom types")

    images = np.asarray(mapping, dtype=int)
    mapped_product = product[images[:, None], images[None, :]]
    changed = set()
    for left in range(size):
        for right in range(left + 1, size):
            if not math.isclose(
                float(reactant[left, right]),
                float(mapped_product[left, right]),
                abs_tol=tolerance,
                rel_tol=0.0,
            ):
                changed.update((left, right))

    h_r = _node_property(lgp[0], "hcounts")
    h_p = _node_property(lgp[1], "hcounts")
    charge_r = _node_property(lgp[0], "charges")
    charge_p = _node_property(lgp[1], "charges")
    hydrogen_distance = 0
    for atom, image in enumerate(mapping):
        h_delta = abs(int(h_r[atom]) - int(h_p[image]))
        hydrogen_distance += h_delta
        if h_delta or charge_r[atom] != charge_p[image]:
            changed.add(atom)

    components = _union_components(reactant, mapped_product)
    active = (
        set().union(*(component for component in components if component & changed))
        if changed
        else set()
    )
    reactant_atoms = tuple(sorted(active))
    product_atoms = tuple(mapping[atom] for atom in reactant_atoms)
    inactive_atoms = tuple(atom for atom in range(size) if atom not in active)
    reduced_pair = (
        _induced_labeled_graph(lgp[0], reactant_atoms),
        _induced_labeled_graph(lgp[1], product_atoms),
    )
    heavy_distance = 0.5 * float(np.abs(reactant - mapped_product).sum())
    return ITSComponentReduction(
        lgp=reduced_pair,
        reference_mapping=tuple(range(len(reactant_atoms))),
        reactant_atoms=reactant_atoms,
        product_atoms=product_atoms,
        inactive_atoms=inactive_atoms,
        fixed_mapping=mapping,
        changed_atoms=tuple(sorted(changed)),
        heavy_distance=heavy_distance,
        hydrogen_distance=hydrogen_distance,
    )


__all__ = ["ITSComponentReduction", "reduce_to_reference_its_components"]
