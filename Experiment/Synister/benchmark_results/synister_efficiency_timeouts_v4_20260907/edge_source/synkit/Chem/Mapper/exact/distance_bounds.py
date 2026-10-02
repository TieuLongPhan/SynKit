"""Polynomial admissible bounds for exact chemical-distance search."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


class ResidualBondMass:
    """Maintain exact residual matrix sum/absolute-sum bounds under DFS."""

    def __init__(self, reactant, product, reactant_order, product_atoms=None):
        atom_count = len(reactant_order)
        self.reactant_sums = [0.0] * (atom_count + 1)
        self.reactant_abs_sums = [0.0] * (atom_count + 1)
        for depth in range(atom_count - 1, -1, -1):
            atom = reactant_order[depth]
            later = reactant_order[depth + 1 :]
            self.reactant_sums[depth] = self.reactant_sums[depth + 1] + float(
                reactant[atom, atom]
                + reactant[atom, later].sum()
                + reactant[later, atom].sum()
            )
            self.reactant_abs_sums[depth] = (
                self.reactant_abs_sums[depth + 1]
                + float(abs(reactant[atom, atom]))
                + float(np.abs(reactant[atom, later]).sum())
                + float(np.abs(reactant[later, atom]).sum())
            )
        self.product = product
        if product_atoms is None:
            product_atoms = range(product.shape[0])
        product_atoms = tuple(product_atoms)
        self.active = np.zeros(product.shape[0], dtype=bool)
        self.active[list(product_atoms)] = True
        product_block = product[np.ix_(product_atoms, product_atoms)]
        self.product_sum = float(product_block.sum())
        self.product_abs_sum = float(np.abs(product_block).sum())

    def interval(self, depth):
        """Return residual bond-mass lower and upper bounds."""
        lower = 0.5 * abs(self.reactant_sums[depth] - self.product_sum)
        upper = 0.5 * (self.reactant_abs_sums[depth] + self.product_abs_sum)
        return lower, upper

    def remove(self, image):
        """Remove one product image and return exact restoration deltas."""
        diagonal = float(self.product[image, image])
        removed = float(
            self.product[image, self.active].sum()
            + self.product[self.active, image].sum()
            - diagonal
        )
        removed_abs = float(
            np.abs(self.product[image, self.active]).sum()
            + np.abs(self.product[self.active, image]).sum()
            - abs(diagonal)
        )
        self.active[image] = False
        self.product_sum -= removed
        self.product_abs_sum -= removed_abs
        return removed, removed_abs

    def restore(self, image, deltas):
        """Undo :meth:`remove` during DFS backtracking."""
        removed, removed_abs = deltas
        self.product_sum += removed
        self.product_abs_sum += removed_abs
        self.active[image] = True


def element_blocks(indices, elements):
    """Group atom indices by their exact element label."""
    blocks: dict[object, list[int]] = {}
    for index in indices:
        blocks.setdefault(elements[index], []).append(index)
    return blocks


def blocked_assignment_extreme(
    costs,
    reactant_atoms,
    product_atoms,
    reactant_elements,
    product_elements,
    *,
    maximize=False,
) -> float:
    """Solve independent element-compatible LAP blocks."""
    reactant_blocks = element_blocks(reactant_atoms, reactant_elements)
    product_blocks = element_blocks(product_atoms, product_elements)
    if set(reactant_blocks) != set(product_blocks):
        return -math.inf if maximize else math.inf

    total = 0.0
    for element, row_atoms in reactant_blocks.items():
        column_atoms = product_blocks[element]
        if len(row_atoms) != len(column_atoms):
            return -math.inf if maximize else math.inf
        block = costs[np.ix_(row_atoms, column_atoms)]
        if not np.isfinite(block).all():
            return -math.inf if maximize else math.inf
        rows, columns = linear_sum_assignment(-block if maximize else block)
        total += float(block[rows, columns].sum())
    return total


def internal_cost_interval(reactant, product, reactant_atoms, product_atoms):
    """Admissible min/max bounds for still-unassigned internal pairs."""
    if not reactant_atoms:
        return 0.0, 0.0
    reactant_block = reactant[np.ix_(reactant_atoms, reactant_atoms)]
    product_block = product[np.ix_(product_atoms, product_atoms)]
    lower = 0.5 * abs(float(reactant_block.sum() - product_block.sum()))
    upper = 0.5 * float(np.abs(reactant_block).sum() + np.abs(product_block).sum())
    return lower, upper


def atom_profile_costs(reactant, product, reactant_elements, product_elements):
    """Lower-bound total CD forced by each individual atom assignment.

    For a fixed ``i -> p``, every element-preserving completion pairs the
    entries in reactant row ``i`` with entries in product row ``p`` inside
    each neighbor-element block. Sorted one-dimensional matching minimizes
    their L1 difference. Half of that row mismatch cannot exceed the complete
    chemical distance, making values above a numeric target safe to remove
    from that atom's domain. Summing the half-row costs over a compatible
    bijection also bounds its total CD. A fixed-prefix sum plus a LAP over
    the remaining profile matrix is therefore an admissible global bound;
    combine it with other global bounds using max to avoid double-counting.
    """
    atom_count = len(reactant_elements)
    costs = np.zeros((atom_count, atom_count), dtype=float)
    reactant_blocks = element_blocks(range(atom_count), reactant_elements)
    product_blocks = element_blocks(range(atom_count), product_elements)
    for element, reactant_atoms in reactant_blocks.items():
        product_atoms = product_blocks[element]
        sorted_reactant = np.sort(reactant[:, reactant_atoms], axis=1)
        sorted_product = np.sort(product[:, product_atoms], axis=1)
        costs += np.abs(sorted_reactant[:, None, :] - sorted_product[None, :, :]).sum(
            axis=2
        )
    costs *= 0.5
    for reactant_atom, element in enumerate(reactant_elements):
        for product_atom, product_element in enumerate(product_elements):
            if element != product_element:
                costs[reactant_atom, product_atom] = math.inf
    return costs


def assignment_edge_lower_bounds(costs, reactant_elements, product_elements):
    """Minimum typed LAP cost forced by every individual assignment edge.

    Relative to an optimal matching, forcing an edge closes an alternating
    cycle. All-pairs shortest paths give its cheapest completion, without
    solving a separate LAP for every candidate edge. This is a bound on the
    complete profile assignment, not an additional edge contribution.
    """
    n = len(reactant_elements)
    result = np.full((n, n), math.inf)
    row_blocks = element_blocks(range(n), reactant_elements)
    column_blocks = element_blocks(range(n), product_elements)
    total = 0.0
    for element, rows in row_blocks.items():
        columns = column_blocks[element]
        block = costs[np.ix_(rows, columns)]
        row_indices, permutation = linear_sum_assignment(block)
        matched_columns = [columns[index] for index in permutation]
        matched_costs = block[row_indices, permutation]
        total += float(matched_costs.sum())
        delta = block[:, permutation] - matched_costs[:, None]
        distances = delta.copy()
        for pivot in range(len(rows)):
            distances = np.minimum(
                distances, distances[:, pivot, None] + distances[pivot, None, :]
            )
        result[np.ix_(rows, matched_columns)] = delta + distances.T
    result += total
    return result


def conditioned_profile_assignment_lower_bound(
    reactant,
    product,
    cross_costs,
    reactant_atoms,
    product_atoms,
    reactant_elements,
    product_elements,
):
    """Bound remaining CD using profiles conditioned on the assigned prefix.

    Cross costs account for edges to assigned atoms exactly. Sorted incident
    profiles within the remaining subgraphs bound internal outgoing-row costs.
    One LAP on their sum enforces a shared remaining atom assignment for both
    contributions. Add committed prefix CD to obtain a full-objective bound.
    """
    if not reactant_atoms:
        return 0.0
    rows = [reactant_elements[atom] for atom in reactant_atoms]
    columns = [product_elements[atom] for atom in product_atoms]
    internal = atom_profile_costs(
        reactant[np.ix_(reactant_atoms, reactant_atoms)],
        product[np.ix_(product_atoms, product_atoms)],
        rows,
        columns,
    )
    costs = internal + cross_costs[np.ix_(reactant_atoms, product_atoms)]
    return blocked_assignment_extreme(
        costs,
        range(len(rows)),
        range(len(columns)),
        rows,
        columns,
    )


def reaction_center_order(
    reactant,
    product,
    reactant_elements,
    element_counts,
    seed_mapping,
):
    """Order a seeded exact search from changed atoms out through the graph."""
    atom_count = len(reactant_elements)
    default = sorted(
        range(atom_count),
        key=lambda atom: (
            element_counts[reactant_elements[atom]],
            -int(np.count_nonzero(reactant[atom])),
            atom,
        ),
    )
    if seed_mapping is None:
        return default
    images = np.asarray(seed_mapping, dtype=int)
    mapped_product = product[images[:, None], images[None, :]]
    local_change = 0.5 * np.abs(reactant - mapped_product).sum(axis=1)
    seeds = [atom for atom in range(atom_count) if local_change[atom] > 0]
    if not seeds:
        return default

    distances = [atom_count + 1] * atom_count
    pending = list(seeds)
    for atom in seeds:
        distances[atom] = 0
    for atom in pending:
        neighbors = np.flatnonzero(
            np.logical_or(reactant[atom] != 0, reactant[:, atom] != 0)
        )
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if distances[neighbor] <= distances[atom] + 1:
                continue
            distances[neighbor] = distances[atom] + 1
            pending.append(neighbor)
    return sorted(
        range(atom_count),
        key=lambda atom: (
            distances[atom],
            -float(local_change[atom]),
            element_counts[reactant_elements[atom]],
            -int(np.count_nonzero(reactant[atom])),
            atom,
        ),
    )


__all__ = [
    "ResidualBondMass",
    "atom_profile_costs",
    "blocked_assignment_extreme",
    "element_blocks",
    "internal_cost_interval",
    "reaction_center_order",
]
