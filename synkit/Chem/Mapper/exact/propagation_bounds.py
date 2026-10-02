"""Admissible residual bounds for the propagated exact-distance search."""

import time

import numpy as np

from .distance_bounds import atom_profile_costs
from .incremental_assignment import reduced_cost_filter, cycle_edge_lower_bounds
from .propagation_limits import PropagationDeadline, check_deadline
from .separator_bound import minimum_separator_residual_cost
from .factor_spectrum import factor_cost_support_intersects


class PropagationBounds:
    """Strengthen a search state's domains with independent residual bounds."""

    def cheap_bounds(self, rows, columns, committed, profile_sum):  # noqa: C901
        """Filter candidates using separate, non-overlapping admissible bounds."""
        allowed = self.domains.matrix(rows, columns)
        if not allowed.any(axis=0).all():
            self.statistics["hall_pruned"] += 1
            return None
        costs = self.cross[np.ix_(rows, columns)]
        row_min = np.where(allowed, costs, 2**60).min(axis=1)
        column_min = np.where(allowed, costs, 2**60).min(axis=0)
        if self.config.typed_transport_bounds:
            transport_lower = self.typed_bonds.transport_lower_bound()
            self.statistics["transport_bound_calls"] += int(transport_lower is not None)
        else:
            transport_lower = None
        if transport_lower is not None:
            mass_lower = transport_lower
        elif self.config.typed_mass_bounds:
            mass_lower = self.typed_bonds.lower_bound()
        else:
            mass_lower = abs(
                int(self.typed_bonds.reactant.sum())
                - int(self.typed_bonds.product.sum())
            )
        lower = committed + max(int(row_min.sum()), int(column_min.sum())) + mass_lower
        dual = (
            profile_sum
            + sum(self.root_u[i] for i in rows)
            + sum(self.root_v[j] for j in columns)
        )
        lower = self.round_cost_bound(lower)
        rounded_dual = self.round_cost_bound(dual)
        if max(lower, rounded_dual) > self.budget():
            self.statistics["lower_bound_pruned"] += 1
            return None
        if (
            self.config.factor_spectrum_bounds
            and len(rows) <= self.config.factor_spectrum_residual_limit
            and self.statistics["factor_spectrum_calls"]
            < self.config.factor_spectrum_max_calls
        ):
            started = time.perf_counter()
            self.statistics["factor_spectrum_calls"] += 1
            residual_lower = 0 if self.proof else self.lower - committed
            residual_upper = self.budget() - committed
            support = factor_cost_support_intersects(
                self.a,
                self.b,
                rows,
                columns,
                costs,
                allowed,
                residual_lower,
                residual_upper,
                max_range=self.config.factor_spectrum_max_range,
                deadline=self.deadline,
            )
            self.statistics["factor_spectrum_seconds"] += time.perf_counter() - started
            if support is False:
                self.statistics["factor_spectrum_pruned"] += 1
                self.statistics["lower_bound_pruned"] += 1
                return None
            if support is None:
                self.statistics["factor_spectrum_skipped"] += 1
        if (
            self.config.separator_bounds
            and len(rows) <= self.config.separator_residual_limit
        ):
            started = time.perf_counter()
            self.statistics["separator_bound_calls"] += 1
            residual_minimum = minimum_separator_residual_cost(
                self.a,
                self.b,
                rows,
                columns,
                costs,
                allowed,
                deadline=self.deadline,
                max_separator_size=self.config.separator_max_size,
                max_states=self.config.separator_max_states,
            )
            self.statistics["separator_bound_seconds"] += time.perf_counter() - started
            if residual_minimum is None:
                self.statistics["separator_bound_skipped"] += 1
            elif committed + residual_minimum > self.budget():
                self.statistics["separator_bound_pruned"] += 1
                self.statistics["lower_bound_pruned"] += 1
                return None
        if not self.proof:
            cross_upper = int(np.where(allowed, costs, 0).max(axis=1).sum())
            internal_upper = self.typed_bonds.upper_bound()
            if committed + cross_upper + internal_upper < self.lower:
                self.statistics["upper_bound_pruned"] += 1
                return None
        global_reduced = (
            self.profile[np.ix_(rows, columns)]
            - np.asarray([self.root_u[i] for i in rows])[:, None]
            - np.asarray([self.root_v[j] for j in columns])[None, :]
        )
        forced = committed + mass_lower + costs + int(row_min.sum()) - row_min[:, None]
        if self.config.column_cost_bounds:
            column_forced = (
                committed
                + mass_lower
                + costs
                + int(column_min.sum())
                - column_min[None, :]
            )
            forced = np.maximum(forced, column_forced)
        permitted = (
            allowed
            & (forced <= self.budget())
            & (dual + global_reduced <= self.budget())
        )
        if self.root_forced is not None:
            permitted &= self.root_forced[np.ix_(rows, columns)] <= self.budget()
        self.restrict(rows, columns, permitted)
        return max(lower, rounded_dual), costs

    def conditioned(self, rows, columns, committed, cheap, parent, depth):
        """Spend on a stronger bound only where the declared policy selects it."""
        lower, cross = cheap
        size = len(rows)
        density = sum(self.domains.masks[i].bit_count() for i in rows) / (size * size)
        selected = (
            not self.config.adaptive_bounds
            or size <= self.config.small_residual
            or depth % self.config.bound_interval == 0
            or self.budget() - lower <= self.config.bound_slack
            or density <= 0.3
        )
        if not selected or self.budget() == self.root_profile_lower:
            self.statistics["conditioned_skipped"] += 1
            return True, parent
        self.statistics["conditioned_calls"] += 1
        if (
            depth == 0
            and not self.fixed
            and tuple(rows) == self.root_rows
            and tuple(columns) == self.root_columns
        ):
            costs = self.profile[np.ix_(rows, columns)]
            allowed = self.domains.matrix(rows, columns)
            state = self.root_assignment
            if allowed[np.arange(size), state.certificate.permutation].all():
                self.statistics["root_assignment_reused"] += 1
                filtered = reduced_cost_filter(
                    costs, allowed, state, self.budget() - committed
                )
                self.restrict(rows, columns, filtered)
                return (
                    self.propagate(rows, matching=state.certificate.permutation),
                    state,
                )
        internal = atom_profile_costs(
            self.a[np.ix_(rows, rows)] / 4,
            self.b[np.ix_(columns, columns)] / 4,
            [self.labels[i] for i in rows],
            [self.product_labels[j] for j in columns],
        )
        costs = cross + np.rint(
            np.where(np.isfinite(internal), internal * 4, 0)
        ).astype(np.int64)
        allowed = self.domains.matrix(rows, columns)
        state = self.assignment(costs, allowed, rows, columns, parent)
        if state is None or committed + state.lower_bound > self.budget():
            self.statistics["lower_bound_pruned"] += 1
            return False, state
        filtered = reduced_cost_filter(costs, allowed, state, self.budget() - committed)
        if self.config.cycle_cost_bounds:
            forced = cycle_edge_lower_bounds(
                costs, allowed, state, deadline=self.deadline
            )
            self.statistics["cycle_bound_calls"] += 1
            filtered &= forced <= self.budget() - committed
        self.restrict(rows, columns, filtered)
        if not self.propagate(rows, matching=state.certificate.permutation):
            return False, state
        if self.config.pairwise_edge_bounds:
            if not self.pairwise_edge_bound(rows, columns, cross, committed):
                return False, state
            if not self.propagate(rows, matching=state.certificate.permutation):
                return False, state
        if self.config.star_assignment_bounds:
            if not self.star_assignment_bound(
                rows, columns, cross, internal, committed
            ):
                return False, state
            if not self.propagate(rows, matching=state.certificate.permutation):
                return False, state
        return True, state

    def pairwise_edge_bound(self, rows, columns, cross, committed):  # noqa: C901
        """Prune an anchor only by independent minima of disjoint cost terms.

        For fixed i->j, the bound adds: its already-assigned cross cost; an
        independent minimum cross cost for each other residual row; an
        independent minimum for each residual pair incident to i; and an
        independent minimum for each remaining residual pair. These terms
        partition every cost not already in ``committed``. Relaxing the joint
        bijection constraints between terms can only lower the sum.
        """
        started = time.perf_counter()
        local_deadline = started + self.config.pairwise_bound_seconds
        if self.deadline is not None:
            local_deadline = min(local_deadline, self.deadline)
        try:
            if (
                len(rows) > self.config.pairwise_bound_residual_limit
                or self.pairwise_seconds_used >= self.config.pairwise_bound_seconds
                or not rows
            ):
                return True
            self.statistics["pairwise_bound_calls"] += 1
            allowed = self.domains.matrix(rows, columns)
            anchors = [
                (int(allowed[x].sum()), int(cross[x, y]), x, y)
                for x in range(len(rows))
                for y in range(len(columns))
                if allowed[x, y]
            ]
            anchors.sort()
            evaluated = 0
            for _, _, anchor_row, anchor_col in anchors:
                check_deadline(self.deadline)
                if (
                    evaluated >= self.config.pairwise_bound_max_anchors
                    or time.perf_counter() >= local_deadline
                ):
                    break
                evaluated += 1
                self.statistics["pairwise_bound_anchors"] += 1
                anchor_image = columns[anchor_col]
                residual = [p for p in range(len(rows)) if p != anchor_row]
                domains = {}
                lower = int(committed) + int(cross[anchor_row, anchor_col])
                feasible = True
                for p in residual:
                    choices = [
                        columns[q]
                        for q in range(len(columns))
                        if q != anchor_col and allowed[p, q]
                    ]
                    if not choices:
                        feasible = False
                        break
                    domains[p] = choices
                    lower += min(
                        int(cross[p, q])
                        for q in range(len(columns))
                        if q != anchor_col and allowed[p, q]
                    )
                    source_row = rows[anchor_row]
                    other_row = rows[p]
                    lower += min(
                        abs(
                            int(self.a[source_row, other_row])
                            - int(self.b[anchor_image, image])
                        )
                        for image in choices
                    )
                if not feasible:
                    allowed[anchor_row, anchor_col] = False
                    self.statistics["pairwise_bound_pruned_edges"] += 1
                    continue
                for offset, left in enumerate(residual):
                    if time.perf_counter() >= local_deadline:
                        break
                    for right in residual[offset + 1 :]:
                        left_row, right_row = rows[left], rows[right]
                        minimum = None
                        for left_image in domains[left]:
                            for right_image in domains[right]:
                                if left_image == right_image:
                                    continue
                                value = abs(
                                    int(self.a[left_row, right_row])
                                    - int(self.b[left_image, right_image])
                                )
                                if minimum is None or value < minimum:
                                    minimum = value
                        if minimum is None:
                            feasible = False
                            break
                        lower += minimum
                    if not feasible:
                        break
                if not feasible:
                    allowed[anchor_row, anchor_col] = False
                    self.statistics["pairwise_bound_pruned_edges"] += 1
                elif lower > self.budget():
                    allowed[anchor_row, anchor_col] = False
                    self.statistics["pairwise_bound_pruned_edges"] += 1
            if not allowed.any(axis=1).all():
                self.statistics["hall_pruned"] += 1
                return False
            self.restrict(rows, columns, allowed)
            return True
        finally:
            elapsed = time.perf_counter() - started
            self.pairwise_seconds_used += elapsed
            self.statistics["pairwise_bound_seconds"] += elapsed

    def star_assignment_bound(self, rows, columns, cross, internal, committed):  # noqa: C901
        """Strengthen row-profile stars with current-domain inner assignments."""
        if (
            len(rows) > self.config.star_bound_residual_limit
            or self.star_seconds_used >= self.config.star_bound_seconds
        ):
            return True
        started = time.perf_counter()
        remaining_time = self.config.star_bound_seconds - self.star_seconds_used
        local_deadline = started + remaining_time
        if self.deadline is not None:
            local_deadline = min(local_deadline, self.deadline)
        self.statistics["star_bound_calls"] += 1
        try:
            profile = np.rint(np.where(np.isfinite(internal), internal * 4, 0)).astype(
                np.int64
            )
            pair_stars = 2 * profile
            allowed = self.domains.matrix(rows, columns)
            anchors = [
                (int(self.domains.masks[i].bit_count()), int(cross[x, y]), i, j, x, y)
                for x, i in enumerate(rows)
                for y, j in enumerate(columns)
                if allowed[x, y]
            ]
            anchors.sort()
            evaluated = 0
            changed = False
            for _, _, row, image, row_pos, image_pos in anchors:
                check_deadline(self.deadline)
                if (
                    evaluated >= self.config.star_bound_max_anchors
                    or time.perf_counter() >= local_deadline
                ):
                    break
                evaluated += 1
                inner_rows = tuple(i for i in rows if i != row)
                inner_columns = tuple(j for j in columns if j != image)
                if not inner_rows:
                    inner_cost = 0
                else:
                    inner_allowed = self.domains.matrix(inner_rows, inner_columns)
                    inner_costs = np.abs(
                        self.a[row, np.asarray(inner_rows), None]
                        - self.b[image, np.asarray(inner_columns)][None, :]
                    )
                    self.statistics["star_inner_assignments"] += 1
                    try:
                        inner = self.assignment(
                            inner_costs,
                            inner_allowed,
                            inner_rows,
                            inner_columns,
                            None,
                            deadline=local_deadline,
                        )
                    except PropagationDeadline:
                        if (
                            self.deadline is not None
                            and time.perf_counter() >= self.deadline
                        ):
                            raise
                        break
                    if inner is None:
                        allowed[row_pos, image_pos] = False
                        self.statistics["star_infeasible_anchors"] += 1
                        changed = True
                        continue
                    inner_cost = int(inner.lower_bound)
                base = int(pair_stars[row_pos, image_pos])
                if inner_cost < base:
                    raise ArithmeticError(
                        "Domain-aware star LAP fell below its profile relaxation"
                    )
                if inner_cost > base:
                    pair_stars[row_pos, image_pos] = inner_cost
                    self.statistics["star_improved_entries"] += 1
                    changed = True
            if not changed:
                return True
            self.restrict(rows, columns, allowed)
            if not allowed.any(axis=0).all():
                self.statistics["hall_pruned"] += 1
                return False
            doubled_costs = 2 * cross + pair_stars
            supported = self.domains.matrix(rows, columns)
            try:
                outer = self.assignment(
                    doubled_costs,
                    supported,
                    rows,
                    columns,
                    None,
                    deadline=local_deadline,
                )
            except PropagationDeadline:
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    raise
                return True
            if outer is None:
                self.statistics["hall_pruned"] += 1
                return False
            if outer.lower_bound > 2 * (self.budget() - committed):
                self.statistics["star_bound_pruned"] += 1
                self.statistics["lower_bound_pruned"] += 1
                return False
            filtered = reduced_cost_filter(
                doubled_costs,
                supported,
                outer,
                2 * (self.budget() - committed),
            )
            self.restrict(rows, columns, filtered)
            return True
        finally:
            elapsed = time.perf_counter() - started
            self.star_seconds_used += elapsed
            self.statistics["star_bound_seconds"] += elapsed
