"""Synister-CP integer DFS with cost-aware domains and adaptive matching bounds.

The objective is undirected bond disagreement on the half-integer lattice.
Bounds and search costs are represented in quarter units. Only a proved
minimum is passed to the enumeration phase, so callbacks never see incumbents.
"""

from __future__ import annotations

import time
from collections import Counter
from operator import itemgetter

import numpy as np

from .assignment_domains import AssignmentDomains, bits
from .distance_bounds import atom_profile_costs, reaction_center_order
from .incremental_assignment import (
    solve_assignment,
    solve_partitioned_assignment,
    cycle_edge_lower_bounds,
)
from .propagation_limits import PropagationDeadline, check_deadline
from .residual_typed_bonds import ResidualTypedBonds
from .cost_lattice import CostLattice
from .separator_spectrum import SeparatorCostSpectrum
from .suffix_spectrum import SuffixCostSpectrum
from .reward_frontier import RewardFrontierSpectrum
from .propagation_bounds import PropagationBounds


class PropagatedSearch(PropagationBounds):
    """Own a reversible search state shared by proof and enumeration passes."""

    def __init__(
        self,
        a,
        b,
        labels,
        product_labels,
        seed,
        fixed,
        group,
        reactant_group,
        expand,
        config,
        deadline,
    ):
        self.a, self.b = a, b
        self.cost_lattice = CostLattice.from_matrices(a, b)
        self.neighbors = a != 0
        self.labels, self.product_labels = labels, product_labels
        self.n, self.seed, self.fixed = len(a), list(seed), dict(fixed)
        self.branch_rank = (
            self._pagerank(a, deadline) if config.branch_order == "pagerank" else None
        )
        self.impact_rank = (
            self._impact_scores(a, b, labels, product_labels, deadline)
            if config.branch_order == "impact"
            else None
        )
        self.group = group
        self.reactant_group = reactant_group
        self.reactant_symmetry = len(reactant_group) > 1
        self.config, self.deadline = config, deadline
        check_deadline(deadline)
        self.expand = expand
        self.emitted_maps = set()
        self.statistics = Counter()
        self.statistics.update(
            assignment_calls=0,
            inherited_edges=0,
            augmentations=0,
            hall_pruned=0,
            domain_edges_removed=0,
            conditioned_calls=0,
            conditioned_skipped=0,
            symmetry_pruned=0,
            reactant_symmetry_pruned=0,
            reactant_symmetry_group_order=len(reactant_group),
            visited_nodes=0,
            visited_leaves=0,
            lower_bound_pruned=0,
            upper_bound_pruned=0,
            forced_assignments=0,
            transport_bound_calls=0,
            seed_swap_candidates=0,
            seed_swaps=0,
            seed_improvement_quarters=0,
            seed_improvement_seconds=0.0,
            star_bound_calls=0,
            star_inner_assignments=0,
            star_infeasible_anchors=0,
            star_improved_entries=0,
            star_bound_pruned=0,
            star_bound_seconds=0.0,
            pairwise_bound_calls=0,
            pairwise_bound_anchors=0,
            pairwise_bound_pruned_edges=0,
            pairwise_bound_seconds=0.0,
            lattice_shell_rejections=0,
            lattice_bound_roundings=0,
            separator_bound_calls=0,
            separator_bound_pruned=0,
            separator_bound_skipped=0,
            separator_bound_seconds=0.0,
            separator_spectrum_calls=0,
            separator_spectrum_skipped=0,
            separator_spectrum_mappings=0,
            separator_spectrum_orbits_pruned=0,
            separator_spectrum_states=0,
            separator_spectrum_seconds=0.0,
            factor_spectrum_calls=0,
            factor_spectrum_pruned=0,
            factor_spectrum_skipped=0,
            factor_spectrum_seconds=0.0,
            suffix_spectrum_calls=0,
            suffix_spectrum_skipped=0,
            suffix_spectrum_mappings=0,
            suffix_spectrum_orbits_pruned=0,
            suffix_spectrum_orbit_branches_pruned=0,
            suffix_spectrum_states=0,
            suffix_spectrum_seconds=0.0,
            suffix_spectrum_prepare_seconds=0.0,
            suffix_spectrum_reconstruction_seconds=0.0,
            suffix_spectrum_reward_frontier_calls=0,
            suffix_spectrum_max_frontier=0,
        )
        self.star_seconds_used = 0.0
        self.pairwise_seconds_used = 0.0
        if config.seed_local_search and config.seed_improvement_seconds:
            self.seed = self.improve_seed(self.seed, config.seed_improvement_seconds)
        floats = atom_profile_costs(a / 4, b / 4, labels, product_labels)
        self.profile = np.rint(np.where(np.isfinite(floats), floats * 4, 0)).astype(
            np.int64
        )
        check_deadline(deadline)
        order = reaction_center_order(a / 4, b / 4, labels, Counter(labels), seed)
        self.rank = {row: i for i, row in enumerate(order)}
        self.base_masks = [
            sum(1 << j for j, element in enumerate(product_labels) if element == label)
            for label in labels
        ]
        self.root_rows = tuple(i for i in range(self.n) if i not in fixed)
        self.root_columns = tuple(
            j for j in range(self.n) if j not in set(fixed.values())
        )
        allowed = np.asarray(
            [
                [labels[i] == product_labels[j] for j in self.root_columns]
                for i in self.root_rows
            ],
            dtype=bool,
        ).reshape(len(self.root_rows), len(self.root_columns))
        self.root_assignment = self.assignment(
            self.profile[np.ix_(self.root_rows, self.root_columns)],
            allowed,
            self.root_rows,
            self.root_columns,
            None,
        )
        self.fixed_profile = sum(int(self.profile[i, j]) for i, j in fixed.items())
        self.root_profile_lower = self.fixed_profile + self.root_assignment.lower_bound
        self.root_lower = self.round_cost_bound(self.root_profile_lower)
        self.root_u = dict(
            zip(self.root_rows, self.root_assignment.certificate.row_potentials)
        )
        self.root_v = dict(
            zip(self.root_columns, self.root_assignment.certificate.column_potentials)
        )
        self.incumbent = self.cost(self.seed)
        self.witness = list(self.seed)
        self.root_forced = None
        if self.config.cycle_cost_bounds and self.incumbent > self.root_lower:
            self.root_forced = np.zeros((self.n, self.n), dtype=np.int64)
            self.root_forced[np.ix_(self.root_rows, self.root_columns)] = (
                self.fixed_profile
                + cycle_edge_lower_bounds(
                    self.profile[np.ix_(self.root_rows, self.root_columns)],
                    allowed,
                    self.root_assignment,
                    deadline=self.deadline,
                )
            )
            self.statistics["cycle_bound_calls"] += 1

    def round_cost_bound(self, lower):
        """Round a valid lower bound to the necessary global cost lattice."""
        if not self.config.cost_lattice_pruning or not self.cost_lattice.modulus:
            return lower
        rounded = self.cost_lattice.first_at_least(lower)
        self.statistics["lattice_bound_roundings"] += int(rounded != lower)
        return rounded

    def cost(self, mapping):
        """Rescore a complete mapping in exact quarter units."""
        delta = np.abs(self.a - self.b[np.ix_(mapping, mapping)])
        return int(delta.sum() // 2)

    def improve_seed(self, mapping, seconds):
        """Try exact same-type swaps under a small incumbent-only time slice."""
        started = time.perf_counter()
        stop = started + seconds
        if self.deadline is not None:
            stop = min(stop, self.deadline)
        current = list(mapping)
        current_cost = self.cost(current)
        fixed_rows = set(self.fixed)
        n = self.n
        evaluated = 0
        changed = True
        while changed and evaluated < 512 and time.perf_counter() < stop:
            changed = False
            for i in range(n):
                if i in fixed_rows:
                    continue
                for j in range(i + 1, n):
                    if (
                        j in fixed_rows
                        or self.labels[i] != self.labels[j]
                        or evaluated >= 512
                        or time.perf_counter() >= stop
                    ):
                        continue
                    evaluated += 1
                    keep = np.fromiter(
                        (k for k in range(n) if k != i and k != j), dtype=int
                    )
                    images = np.asarray(current, dtype=int)[keep]
                    old = int(
                        np.abs(self.a[i, keep] - self.b[current[i], images]).sum()
                        + np.abs(self.a[j, keep] - self.b[current[j], images]).sum()
                    )
                    new = int(
                        np.abs(self.a[i, keep] - self.b[current[j], images]).sum()
                        + np.abs(self.a[j, keep] - self.b[current[i], images]).sum()
                    )
                    self.statistics["seed_swap_candidates"] += 1
                    if new < old:
                        current[i], current[j] = current[j], current[i]
                        current_cost -= old - new
                        self.statistics["seed_swaps"] += 1
                        changed = True
                if evaluated >= 512 or time.perf_counter() >= stop:
                    break
        rescored = self.cost(current)
        if rescored != current_cost:
            raise ArithmeticError(
                "Incremental seed-swap cost disagrees with literal score"
            )
        self.statistics["seed_improvement_quarters"] = self.cost(mapping) - rescored
        self.statistics["seed_improvement_seconds"] = time.perf_counter() - started
        return current

    @staticmethod
    def _pagerank(adjacency, deadline=None):
        """Return a bounded unweighted PageRank tie-break score per reactant."""
        transition = (adjacency != 0).astype(float)
        degrees = transition.sum(axis=1)
        transition = np.divide(
            transition,
            degrees[:, None],
            out=np.zeros_like(transition),
            where=degrees[:, None] != 0,
        )
        score = np.full(len(adjacency), 1.0 / len(adjacency))
        for _ in range(100):
            check_deadline(deadline)
            dangling = score[degrees == 0].sum() / len(adjacency)
            updated = 0.85 * (transition.T @ score + dangling) + 0.15 / len(adjacency)
            if np.max(np.abs(updated - score)) < 1e-12:
                return updated
            score = updated
        return score

    @staticmethod
    def _impact_scores(a, b, labels, product_labels, deadline=None):
        """Upper-bound the bond-disagreement mass incident to each reactant."""
        max_product_abs = {}
        for left, left_label in enumerate(product_labels):
            check_deadline(deadline)
            for right, right_label in enumerate(product_labels):
                if left != right:
                    key = (left_label, right_label)
                    max_product_abs[key] = max(
                        max_product_abs.get(key, 0), abs(int(b[left, right]))
                    )
        scores = np.zeros(len(a), dtype=np.int64)
        for i in range(len(a)):
            check_deadline(deadline)
            for k in range(len(a)):
                if i != k:
                    product_cap = max_product_abs.get((labels[i], labels[k]), 0)
                    # |a-b| <= |a|+|b| for signed and nonnegative weights.
                    scores[i] += abs(int(a[i, k])) + product_cap
        return scores

    def expired(self):
        """Record an honest wall-time stop, including preprocessing time."""
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.stop_reason = "time_limit"
        return self.stop_reason is not None

    def assignment(self, costs, allowed, rows, columns, parent, deadline=None):
        """Solve and independently verify an exact residual assignment."""
        self.statistics["assignment_calls"] += 1
        inherited = parent if self.config.incremental_assignments else None
        deadline = self.deadline if deadline is None else deadline
        if self.config.block_assignments:
            state = solve_partitioned_assignment(
                costs,
                allowed,
                rows,
                columns,
                [self.labels[i] for i in rows],
                [self.product_labels[j] for j in columns],
                inherited,
                deadline=deadline,
            )
        else:
            state = solve_assignment(
                costs, allowed, rows, columns, inherited, deadline=deadline
            )
        if state is not None:
            self.statistics["inherited_edges"] += state.inherited_edges
            self.statistics["augmentations"] += state.augmentations
        return state

    def budget(self):
        """Proof rejects ties; selected shells retain every tolerance tie."""
        return self.incumbent - 1 if self.proof else self.upper

    def reset(self, *, proof, lower=0, upper=0, emit=None, max_mappings=None):
        self.proof, self.lower, self.upper = proof, lower, upper
        self.emit, self.max_mappings = emit, max_mappings
        self.stop_reason, self.proven_by_root = None, False
        self.selected = 0
        self.mapping = [-1] * self.n
        self.available = (1 << self.n) - 1
        self.cross = np.zeros((self.n, self.n), dtype=np.int64)
        self.domains = AssignmentDomains(self.base_masks)
        self.typed_bonds = ResidualTypedBonds(
            self.a,
            self.b,
            self.labels,
            self.product_labels,
            self.root_rows,
            self.root_columns,
        )
        assigned = sorted(self.fixed)
        self.assigned_neighbors = np.count_nonzero(self.neighbors[:, assigned], axis=1)
        images = [self.fixed[i] for i in assigned]
        committed = int(
            np.abs(
                self.a[np.ix_(assigned, assigned)] - self.b[np.ix_(images, images)]
            ).sum()
            // 2
        )
        for row, image in self.fixed.items():
            self.mapping[row] = image
            self.available &= ~(1 << image)
            self.cross += np.abs(self.a[:, row, None] - self.b[:, image][None, :])
        for row in self.root_rows:
            self.domains.intersect(row, self.available)
        if self.budget() == self.root_profile_lower:
            for row, image in self.fixed.items():
                self.tighten_zero_rows(row, image, self.root_rows)
        return committed

    def prove(self):
        """Prove the exact optimum without exposing intermediate witnesses."""
        committed = self.reset(proof=True)
        if self.expired():
            return False
        if self.root_lower >= self.incumbent:
            self.statistics["root_attainment_proofs"] += 1
            return True
        try:
            self.visit(
                self.root_rows,
                committed,
                self.fixed_profile,
                self.group,
                self.root_assignment,
                0,
            )
        except PropagationDeadline:
            self.stop_reason = "time_limit"
        return self.stop_reason is None

    def enumerate(self, lower, upper, emit, max_mappings):
        """Enumerate the selected lattice shell with unique subgroup expansion."""
        committed = self.reset(
            proof=False, lower=lower, upper=upper, emit=emit, max_mappings=max_mappings
        )
        self.emitted_maps.clear()
        if self.config.cost_lattice_pruning and not self.cost_lattice.intersects(
            lower, upper
        ):
            self.statistics["lattice_shell_rejections"] += 1
            return True
        try:
            self.visit(
                self.root_rows,
                committed,
                self.fixed_profile,
                self.group,
                self.root_assignment,
                0,
            )
        except PropagationDeadline:
            self.stop_reason = "time_limit"
        return self.stop_reason is None

    def restrict(self, rows, columns, allowed):
        """Apply a sound edge mask through the reversible trail."""
        removed = self.domains.restrict_matrix(rows, columns, allowed)
        self.statistics["domain_edges_removed"] += removed

    def propagate(self, rows, *, matching=None, full=None):
        check_deadline(self.deadline)
        ok, removed = self.domains.propagate(
            rows,
            self.available,
            full=self.config.domain_propagation if full is None else full,
            matching=matching,
            cache=self.config.cache_propagation,
        )
        self.statistics["domain_edges_removed"] += removed
        if not ok:
            self.statistics["hall_pruned"] += 1
        return ok

    @staticmethod
    def matching_witness(state, rows, columns):
        """Project a certified parent matching onto a residual subproblem."""
        if state is None:
            return None
        certificate = state.certificate
        previous = {
            row: state.columns[certificate.permutation[index]]
            for index, row in enumerate(state.rows)
        }
        column_position = {image: index for index, image in enumerate(columns)}
        try:
            witness = tuple(column_position[previous[row]] for row in rows)
        except KeyError:
            return None
        if len(set(witness)) != len(columns) or len(witness) != len(rows):
            return None
        return witness

    def tighten_zero_rows(self, row, image, remaining):
        """On an attained profile face, zero-disagreement rows preserve edges."""
        assigned_zero = self.profile[row, image] == 0
        indices = np.asarray(remaining, dtype=int)
        allowed = self.a[indices, row, None] == self.b[None, :, image]
        if not assigned_zero:
            allowed |= self.profile[indices] != 0
        self.domains.restrict_matrix(remaining, tuple(range(self.n)), allowed)

    def select_row(self, rows):
        """Choose a constrained atom while keeping all admissible candidates."""
        branch_rank = self.branch_rank
        impact_rank = self.impact_rank
        if self.reactant_symmetry:
            assigned = tuple(i for i, image in enumerate(self.mapping) if image >= 0)
            stabilizer = tuple(
                g for g in self.reactant_group if all(g[i] == i for i in assigned)
            )
            remaining = set(rows)
            representatives = []
            while remaining:
                representative = min(remaining, key=lambda i: self.rank[i])
                orbit = {g[representative] for g in stabilizer} & remaining
                if not orbit:
                    orbit = {representative}
                representatives.append(representative)
                remaining.difference_update(orbit)
                self.statistics["reactant_symmetry_pruned"] += len(orbit) - 1
            rows = tuple(representatives)
        if self.config.branch_order == "contention":
            return min(
                rows,
                key=lambda i: (
                    self.domains.masks[i].bit_count(),
                    -int(self.assigned_neighbors[i]),
                    -sum(
                        (self.domains.masks[i] & self.domains.masks[k]).bit_count()
                        for k in rows
                        if k != i
                    ),
                    self.rank[i],
                ),
            )
        return min(
            rows,
            key=lambda i: (
                self.domains.masks[i].bit_count(),
                -int(self.assigned_neighbors[i]),
                -float(branch_rank[i]) if branch_rank is not None else 0.0,
                -int(impact_rank[i]) if impact_rank is not None else 0,
                self.rank[i],
            ),
        )

    def accept(self, committed):  # noqa: C901
        if self.proof:
            if committed < self.incumbent:
                self.incumbent, self.witness = committed, list(self.mapping)
                self.proven_by_root = self.incumbent == self.root_lower
            return
        if not self.lower <= committed <= self.upper:
            return
        if not self.reactant_symmetry:
            # Product action on full bijections is free: g o f = h o f
            # implies g = h because f is surjective. The verified subgroup
            # contains distinct permutations, so no per-leaf dedup is needed.
            cost = committed / 4
            if not self.expand or len(self.group) == 1:
                if self.expired():
                    return
                if self.max_mappings is not None and self.selected >= self.max_mappings:
                    self.stop_reason = "mapping_limit"
                    return
                self.selected += 1
                self.emit(tuple(self.mapping), cost)
                return
            # A nontrivial group needs at least two images, so itemgetter
            # always returns a tuple. Construct it once for this full map.
            permute = itemgetter(*self.mapping)
            for permutation in self.group:
                if self.expired():
                    return
                if self.max_mappings is not None and self.selected >= self.max_mappings:
                    self.stop_reason = "mapping_limit"
                    return
                self.selected += 1
                self.emit(permute(permutation), cost)
            return
        product_expansion = self.group if self.expand else (tuple(range(self.n)),)
        reactant_expansion = (
            self.reactant_group if self.expand else (tuple(range(self.n)),)
        )
        emitted = set()
        for source_permutation in reactant_expansion:
            inverse_source = [0] * self.n
            for i, j in enumerate(source_permutation):
                inverse_source[j] = i
            source_mapped = [self.mapping[inverse_source[i]] for i in range(self.n)]
            for product_permutation in product_expansion:
                if self.expired():
                    return
                mapping = tuple(product_permutation[j] for j in source_mapped)
                if mapping in emitted or (
                    self.reactant_symmetry and mapping in self.emitted_maps
                ):
                    continue
                emitted.add(mapping)
                if self.reactant_symmetry:
                    self.emitted_maps.add(mapping)
                if self.max_mappings is not None and self.selected >= self.max_mappings:
                    # Stop only when another valid map exists. Exactly hitting
                    # the cap on the final map permits honest completion.
                    self.stop_reason = "mapping_limit"
                    return
                self.selected += 1
                self.emit(mapping, committed / 4)

    def reconstruct_suffix(self, spectrum, rows, committed, stabilizers):
        """Emit exact suffix mappings and record time even on interruption."""
        started = time.perf_counter()
        try:
            for suffix, residual_cost in spectrum.mappings_between(
                self.lower - committed,
                self.upper - committed,
                symmetry_group=(
                    stabilizers if self.config.suffix_spectrum_orbit_pruning else ()
                ),
            ):
                if self.stop_reason or self.expired():
                    break
                if (
                    not self.config.suffix_spectrum_orbit_pruning
                    and len(stabilizers) > 1
                ):
                    orbit_minimum = min(
                        tuple(group[image] for image in suffix) for group in stabilizers
                    )
                    if suffix != orbit_minimum:
                        self.statistics["suffix_spectrum_orbits_pruned"] += 1
                        continue
                try:
                    for row, image in zip(rows, suffix):
                        self.mapping[row] = image
                    self.statistics["visited_leaves"] += 1
                    self.statistics["suffix_spectrum_mappings"] += 1
                    self.accept(committed + residual_cost)
                finally:
                    for row in rows:
                        self.mapping[row] = -1
        finally:
            if self.config.suffix_spectrum_orbit_pruning:
                self.statistics[
                    "suffix_spectrum_orbit_branches_pruned"
                ] += spectrum.orbits_pruned
            elapsed = time.perf_counter() - started
            self.statistics["suffix_spectrum_reconstruction_seconds"] += elapsed
            self.statistics["suffix_spectrum_seconds"] += elapsed

    def visit(self, rows, committed, profile_sum, stabilizers, parent, depth):  # noqa: C901
        """Explore one reversible partial-bijection state."""
        self.statistics["visited_nodes"] += 1
        if self.expired() or self.proven_by_root:
            return
        if committed > self.budget():
            self.statistics["lower_bound_pruned"] += 1
            return
        if not rows:
            self.statistics["visited_leaves"] += 1
            self.accept(committed)
            return
        token = self.domains.checkpoint()
        forced = []
        try:
            if not self.propagate(rows, full=False):
                return
            if self.config.batch_forced_assignments:
                rows, committed, profile_sum, stabilizers, feasible = self.batch_forced(
                    rows, committed, profile_sum, stabilizers, forced
                )
                if not feasible or self.stop_reason or committed > self.budget():
                    return
                if not rows:
                    self.statistics["visited_leaves"] += 1
                    self.accept(committed)
                    return
            columns = tuple(bits(self.available))
            cheap = self.cheap_bounds(rows, columns, committed, profile_sum)
            witness = self.matching_witness(parent, rows, columns)
            if cheap is None or not self.propagate(rows, matching=witness):
                return
            ok, state = self.conditioned(rows, columns, committed, cheap, parent, depth)
            if not ok or self.expired():
                return
            if (
                self.config.suffix_spectrum
                and not self.proof
                and len(rows) <= self.config.suffix_spectrum_residual_limit
                and self.statistics["suffix_spectrum_calls"]
                < self.config.suffix_spectrum_max_calls
                and not self.reactant_symmetry
                and self.upper - committed <= self.config.suffix_spectrum_max_cost
            ):
                allowed = self.domains.matrix(rows, columns)
                spectrum_type = SuffixCostSpectrum
                spectrum_options = {}
                if self.config.suffix_spectrum_representation == "reward_frontier":
                    spectrum_type = RewardFrontierSpectrum
                    spectrum_options["max_reward"] = (
                        self.config.suffix_spectrum_max_reward
                    )
                    self.statistics["suffix_spectrum_reward_frontier_calls"] += 1
                started = time.perf_counter()
                spectrum = spectrum_type(
                    self.a,
                    self.b,
                    rows,
                    columns,
                    self.cross[np.ix_(rows, columns)],
                    allowed,
                    deadline=self.deadline,
                    max_cost=self.upper - committed,
                    max_states=self.config.suffix_spectrum_max_states,
                    max_seconds=self.config.suffix_spectrum_max_seconds_per_call,
                    **spectrum_options,
                )
                self.statistics["suffix_spectrum_max_frontier"] = max(
                    self.statistics["suffix_spectrum_max_frontier"],
                    getattr(spectrum, "max_frontier", 0),
                )
                self.statistics["suffix_spectrum_calls"] += 1
                try:
                    prepared = spectrum.prepare()
                finally:
                    elapsed = time.perf_counter() - started
                    self.statistics["suffix_spectrum_prepare_seconds"] += elapsed
                    self.statistics["suffix_spectrum_seconds"] += elapsed
                    self.statistics["suffix_spectrum_states"] += spectrum.states
                if prepared:
                    self.reconstruct_suffix(spectrum, rows, committed, stabilizers)
                    return
                self.statistics["suffix_spectrum_skipped"] += 1
            if (
                self.config.separator_spectrum
                and not self.proof
                and len(rows) <= self.config.separator_residual_limit
                and self.statistics["separator_spectrum_calls"]
                < self.config.separator_spectrum_max_calls
                and not self.reactant_symmetry
            ):
                allowed = self.domains.matrix(rows, columns)
                spectrum = SeparatorCostSpectrum(
                    self.a,
                    self.b,
                    rows,
                    columns,
                    self.cross[np.ix_(rows, columns)],
                    allowed,
                    deadline=self.deadline,
                    max_separator_size=self.config.separator_max_size,
                    max_states=self.config.separator_max_states,
                    max_cost=self.upper - committed,
                )
                if spectrum.has_separator():
                    self.statistics["separator_spectrum_calls"] += 1
                    started = time.perf_counter()
                    if spectrum.prepare():
                        self.statistics["separator_spectrum_states"] += spectrum.states
                        for suffix, residual_cost in spectrum.mappings_between(
                            self.lower - committed, self.upper - committed
                        ):
                            if self.stop_reason or self.expired():
                                break
                            if len(stabilizers) > 1:
                                orbit_minimum = min(
                                    tuple(group[image] for image in suffix)
                                    for group in stabilizers
                                )
                                if suffix != orbit_minimum:
                                    self.statistics[
                                        "separator_spectrum_orbits_pruned"
                                    ] += 1
                                    continue
                            for row, image in zip(rows, suffix):
                                self.mapping[row] = image
                            self.statistics["visited_leaves"] += 1
                            self.statistics["separator_spectrum_mappings"] += 1
                            self.accept(committed + residual_cost)
                            for row in rows:
                                self.mapping[row] = -1
                        self.statistics["separator_spectrum_seconds"] += (
                            time.perf_counter() - started
                        )
                        return
                    self.statistics["separator_spectrum_skipped"] += 1
                    self.statistics["separator_spectrum_states"] += spectrum.states
                    self.statistics["separator_spectrum_seconds"] += (
                        time.perf_counter() - started
                    )
            row = self.select_row(rows)
            remaining = tuple(i for i in rows if i != row)
            candidates = sorted(
                bits(self.domains.masks[row] & self.available),
                key=lambda j: (j != self.seed[row], int(self.cross[row, j]), j),
            )
            seen = set()
            for image in candidates:
                if self.stop_reason or self.proven_by_root:
                    break
                if image in seen:
                    self.statistics["symmetry_pruned"] += 1
                    continue
                seen.update(g[image] for g in stabilizers)
                self.child(
                    row,
                    image,
                    remaining,
                    committed,
                    profile_sum,
                    stabilizers,
                    state,
                    depth,
                )
        finally:
            for row, image, delta, bond_token in reversed(forced):
                self.typed_bonds.restore(bond_token)
                self.cross -= delta
                self.available |= 1 << image
                self.mapping[row] = -1
                self.assigned_neighbors -= self.neighbors[:, row]
            self.domains.restore(token)

    def batch_forced(self, rows, committed, profile_sum, stabilizers, undo):
        """Commit forced singleton rows iteratively before another full bound.

        The caller owns the outer domain checkpoint. Assigned images and every
        numeric cache are changed with the same reversible operations as child().
        """
        remaining = tuple(rows)
        while remaining:
            check_deadline(self.deadline)
            forced_rows = [
                row for row in remaining if self.domains.masks[row].bit_count() == 1
            ]
            if not forced_rows:
                break
            row = forced_rows[0]
            image = next(bits(self.domains.masks[row]))
            if not (self.available & (1 << image)):
                self.statistics["hall_pruned"] += 1
                return remaining, committed, profile_sum, stabilizers, False
            next_rows = tuple(i for i in remaining if i != row)
            increment = int(self.cross[row, image])
            self.mapping[row] = image
            self.assigned_neighbors += self.neighbors[:, row]
            self.available &= ~(1 << image)
            delta = np.abs(self.a[:, row, None] - self.b[image][None, :])
            self.cross += delta
            bond_token = self.typed_bonds.remove(
                row, image, next_rows, tuple(bits(self.available))
            )
            undo.append((row, image, delta, bond_token))
            if self.budget() == self.root_profile_lower:
                self.tighten_zero_rows(row, image, next_rows)
            stabilizers = tuple(g for g in stabilizers if g[image] == image)
            committed += increment
            profile_sum += int(self.profile[row, image])
            self.statistics["forced_assignments"] += 1
            remaining = next_rows
            if not self.propagate(remaining, full=False):
                return remaining, committed, profile_sum, stabilizers, False
        return remaining, committed, profile_sum, stabilizers, True

    def child(
        self, row, image, remaining, committed, profile_sum, stabilizers, parent, depth
    ):
        token = self.domains.checkpoint()
        increment = int(self.cross[row, image])
        self.mapping[row] = image
        self.assigned_neighbors += self.neighbors[:, row]
        self.available &= ~(1 << image)
        delta = np.abs(self.a[:, row, None] - self.b[:, image][None, :])
        self.cross += delta
        bond_token = self.typed_bonds.remove(
            row, image, remaining, tuple(bits(self.available))
        )
        try:
            if self.budget() == self.root_profile_lower:
                self.tighten_zero_rows(row, image, remaining)
            group = tuple(g for g in stabilizers if g[image] == image)
            self.visit(
                remaining,
                committed + increment,
                profile_sum + int(self.profile[row, image]),
                group,
                parent,
                depth + 1,
            )
        finally:
            self.typed_bonds.restore(bond_token)
            self.cross -= delta
            self.available |= 1 << image
            self.mapping[row] = -1
            self.assigned_neighbors -= self.neighbors[:, row]
            self.domains.restore(token)
