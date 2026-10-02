"""Refinement helpers for the native exact graph canonizer."""

from __future__ import annotations

from typing import Any, Hashable

_TRACE_NODE_LIMIT = 64
_TRACE_PREVIEW_WORK_LIMIT = 512
_CERTIFICATE_UNRESOLVED_LIMIT = 16
_COMPONENT_CELL_LIMIT = 64


class ExactRefinementMixin:
    """Incremental refinement and component helpers for exact search."""

    def _affected_cell_indices(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        changed_members: frozenset[Hashable],
    ) -> frozenset[int]:
        affected = set(changed_members)
        for node in changed_members:
            affected.update(neighbour for neighbour, _token in self._outgoing[node])
            if self._directed:
                affected.update(neighbour for neighbour, _token in self._incoming[node])
        cell_index = {
            node: index for index, cell in enumerate(partition) for node in cell
        }
        return frozenset(cell_index[node] for node in affected)

    def _refine_incremental(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        changed_members: frozenset[Hashable],
    ) -> tuple[tuple[Hashable, ...], ...]:
        """Resume equitable refinement after one stable-cell split."""
        current = partition
        candidates = self._affected_cell_indices(current, changed_members)
        while True:
            changed = False
            refined: list[tuple[Hashable, ...]] = []
            split_members: set[Hashable] = set()
            cell_index = {
                node: index for index, cell in enumerate(current) for node in cell
            }
            for index, cell in enumerate(current):
                if index not in candidates or len(cell) == 1:
                    refined.append(cell)
                    continue
                buckets: dict[tuple[Any, ...], list[Hashable]] = {}
                for node in cell:
                    signature = self._cell_signature(
                        node,
                        current,
                        cell_index,
                    )
                    buckets.setdefault(signature, []).append(node)
                if len(buckets) == 1:
                    refined.append(cell)
                    continue
                changed = True
                split_members.update(cell)
                refined.extend(
                    tuple(buckets[signature]) for signature in sorted(buckets)
                )
            self._refinement_rounds += 1
            current = tuple(refined)
            if not changed:
                return current
            candidates = self._affected_cell_indices(
                current,
                frozenset(split_members),
            )

    def _partition_lower_bound(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
    ) -> tuple[Any, ...]:
        """Return a componentwise optimistic certificate for a subtree."""
        position_cells = tuple(
            index for index, cell in enumerate(partition) for _node in cell
        )
        node_part = tuple(
            self._node_colors[cell[0]] for cell in partition for _node in cell
        )
        minimum_tokens: dict[tuple[int, int, bool], tuple[Any, ...]] = {}

        def minimum_token(
            left_index: int,
            right_index: int,
            same_position: bool,
        ) -> tuple[Any, ...]:
            cache_key = (left_index, right_index, same_position)
            cached = minimum_tokens.get(cache_key)
            if cached is not None:
                return cached
            left_cell = partition[left_index]
            right_cell = partition[right_index]
            candidates = []
            for left in left_cell:
                for right in right_cell:
                    if same_position and left != right:
                        continue
                    if (
                        not same_position
                        and left_index == right_index
                        and left == right
                    ):
                        continue
                    token = self._edge_token(left, right)
                    candidates.append(("absent",) if token is None else ("edge", token))
            if not candidates:
                raise RuntimeError("Canonical lower bound has no candidate edge.")
            result = min(candidates)
            minimum_tokens[cache_key] = result
            return result

        adjacency = []
        if self._directed:
            pairs = (
                (left, right)
                for left in range(len(position_cells))
                for right in range(len(position_cells))
            )
        else:
            pairs = (
                (left, right)
                for left in range(len(position_cells))
                for right in range(left, len(position_cells))
            )
        for left, right in pairs:
            adjacency.append(
                minimum_token(
                    position_cells[left],
                    position_cells[right],
                    left == right,
                )
            )
        return (
            ("directed", int(self._directed)),
            ("nodes", node_part),
            ("adjacency", tuple(adjacency)),
        )

    def _refinement_trace(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
    ) -> tuple[Any, ...]:
        """Return an invariant trace used only for branch ordering."""
        cell_index = {
            node: index for index, cell in enumerate(partition) for node in cell
        }
        return tuple(
            (
                len(cell),
                self._node_colors[cell[0]],
                self._cell_signature(cell[0], partition, cell_index),
            )
            for cell in partition
        )

    def _cells_are_nonuniformly_joined(
        self,
        left: tuple[Hashable, ...],
        right: tuple[Hashable, ...],
        *,
        same_cell: bool,
    ) -> bool:
        """Return whether a cell pair contains more than one edge pattern.

        Refinement components call this for distinct ambiguous cells.  Walking
        the sparse adjacency lists avoids probing every possible node pair,
        while retaining the exact absent-edge versus coloured-edge test.
        """
        right_members = frozenset(right)
        edge_count = 0
        edge_tokens = set()
        for source in left:
            for target, token in self._outgoing[source]:
                if target not in right_members:
                    continue
                if same_cell and source == target:
                    continue
                edge_count += 1
                edge_tokens.add(token)
                if len(edge_tokens) > 1:
                    return True
        possible = len(left) * len(right)
        if same_cell:
            possible -= len(set(left) & right_members)
        return bool(edge_count) and edge_count < possible

    def _nonuniform_component(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        ambiguous: tuple[tuple[int, tuple[Hashable, ...]], ...],
        seed_index: int,
    ) -> frozenset[Hashable] | None:
        """Return the unresolved nonuniform component containing a cell."""
        cells = {index: cell for index, cell in ambiguous}
        # Component discovery is a branch-ordering heuristic, not part of the
        # certificate proof.  On large already-refined stereographs, scanning
        # every ambiguous-cell pair can cost more than the shallow exact
        # search it is intended to help.
        if len(cells) > _COMPONENT_CELL_LIMIT:
            return None
        adjacency = {index: set() for index in cells}
        indices = tuple(cells)
        for offset, left_index in enumerate(indices):
            for right_index in indices[offset:]:
                if left_index == right_index:
                    continue
                left, right = cells[left_index], cells[right_index]
                nonuniform = self._cells_are_nonuniformly_joined(
                    left,
                    right,
                    same_cell=False,
                )
                if self._directed and not nonuniform:
                    nonuniform = self._cells_are_nonuniformly_joined(
                        right,
                        left,
                        same_cell=False,
                    )
                if nonuniform:
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)

        component = {seed_index}
        pending = [seed_index]
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour not in component:
                    component.add(neighbour)
                    pending.append(neighbour)
        if len(component) == len(cells):
            return None
        return frozenset(node for index in component for node in partition[index])

    def _refine_search_partition(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        changed_members: frozenset[Hashable] | None,
        already_refined: bool,
    ) -> tuple[tuple[Hashable, ...], ...]:
        if already_refined:
            return partition
        if changed_members is None:
            return self._refine(partition)
        return self._refine_incremental(partition, changed_members)

    def _certificate_prunes(
        self,
        refined: tuple[tuple[Hashable, ...], ...],
        ambiguous: tuple[tuple[int, tuple[Hashable, ...]], ...],
        best_key: tuple[Any, ...] | None,
    ) -> bool:
        unresolved_nodes = sum(len(cell) for _index, cell in ambiguous)
        return (
            best_key is not None
            and len(self._nodes) <= _TRACE_NODE_LIMIT
            and unresolved_nodes <= _CERTIFICATE_UNRESOLVED_LIMIT
            and self._partition_lower_bound(refined) > best_key
        )

    def _select_component_target(
        self,
        refined: tuple[tuple[Hashable, ...], ...],
        ambiguous: tuple[tuple[int, tuple[Hashable, ...]], ...],
        active_component: frozenset[Hashable] | None,
    ) -> tuple[int, tuple[Hashable, ...], frozenset[Hashable] | None]:
        component_ambiguous = tuple(
            item
            for item in ambiguous
            if active_component is not None and set(item[1]).issubset(active_component)
        )
        cell_index, target = min(
            component_ambiguous or ambiguous,
            key=lambda item: (len(item[1]), item[0]),
        )
        if not component_ambiguous:
            active_component = self._nonuniform_component(
                refined,
                ambiguous,
                cell_index,
            )
        return cell_index, target, active_component

    def _individualized_child(
        self,
        refined: tuple[tuple[Hashable, ...], ...],
        cell_index: int,
        target: tuple[Hashable, ...],
        chosen: Hashable,
    ) -> tuple[tuple[Hashable, ...], ...]:
        remainder = tuple(node for node in target if node != chosen)
        child = list(refined)
        child[cell_index : cell_index + 1] = (
            [(chosen,), remainder] if remainder else [(chosen,)]
        )
        return self._refine_incremental(
            tuple(child),
            frozenset(target),
        )

    def _ordered_search_children(
        self,
        refined: tuple[tuple[Hashable, ...], ...],
        cell_index: int,
        target: tuple[Hashable, ...],
    ) -> list[
        tuple[
            tuple[Any, ...],
            int,
            Hashable,
            tuple[tuple[Hashable, ...], ...] | None,
        ]
    ]:
        if (
            len(self._nodes) > _TRACE_NODE_LIMIT
            or len(target) * len(self._nodes) > _TRACE_PREVIEW_WORK_LIMIT
        ):
            return [
                ((), position, chosen, None) for position, chosen in enumerate(target)
            ]
        prepared = []
        for position, chosen in enumerate(target):
            child = self._individualized_child(
                refined,
                cell_index,
                target,
                chosen,
            )
            prepared.append(
                (
                    self._refinement_trace(child),
                    position,
                    chosen,
                    child,
                )
            )
        return sorted(prepared, key=lambda item: (item[0], item[1]))

    def _orbit_already_explored(
        self,
        chosen: Hashable,
        explored: list[tuple[Hashable, tuple[Any, ...], tuple[Hashable, ...]]],
        stabilizers: tuple[dict[Hashable, Hashable], ...],
    ) -> bool:
        return any(
            chosen in self._generator_orbit(previous, stabilizers)
            for previous, _key, _order in explored
        )

    def _register_equal_child_generators(
        self,
        explored: list[tuple[Hashable, tuple[Any, ...], tuple[Hashable, ...]]],
        key: tuple[Any, ...],
        order: tuple[Hashable, ...],
        state: Any,
    ) -> None:
        for _previous, previous_key, previous_order in explored:
            if previous_key == key:
                self._register_generator(previous_order, order, state)


__all__ = ["ExactRefinementMixin"]
