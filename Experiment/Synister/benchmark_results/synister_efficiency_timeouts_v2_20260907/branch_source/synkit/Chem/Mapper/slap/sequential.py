from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..graph.labeled_graph import LabeledGraph
from ..graph.synkit_adapter import canonical_graph_hash, stable_int_token


class GraphMatcher:
    """WL-partitioned sequential LAP matcher."""

    _INF_INT = 100000
    _AUT_DEDUP_MAX_NODES = 20
    _AUT_DEDUP_MAX_AUTOMORPHISMS = 256

    def __init__(
        self,
        binary=False,
        max_lap_fingerprints=10000,
        cache_label_blocks=False,
        deterministic_labels=False,
    ):
        self.binary = binary
        self.max_lap_fingerprints = max_lap_fingerprints
        self.cache_label_blocks = cache_label_blocks
        self.deterministic_labels = deterministic_labels
        self._stack = []
        self._visited = set()
        self._label_block_cache = {}
        self.results = []
        self.minval = self._INF_INT
        self._keep_lap_sols = True

    def reset(self):
        """Clear state."""
        self._stack.clear()
        self._visited.clear()
        self._label_block_cache.clear()
        self.results.clear()
        self.minval = self._INF_INT

    def get_maps(self, lgp, break_sym_targets=None, interactive=False, base=None):
        """Find minimum-cost mapping states."""
        self.reset()
        interactive = bool(interactive and break_sym_targets is not None)
        self._keep_lap_sols = break_sym_targets is not None

        seed = [lgp[0].copy(), lgp[1].copy()]
        pending = [[lgp[0].copy(), lgp[1].copy()]]
        completed = []

        if self.binary:
            for pair in pending:
                for graph in pair:
                    graph.binarize_graph()

        chosen = [[], []]
        if interactive:
            base = self._read_index_base(base)

        while pending:
            self._stack.extend(pending)
            self.results.clear()
            self.minval = self._INF_INT
            self._drain_stack()
            self._keep_minimum_results()

            if break_sym_targets is None:
                break
            if interactive:
                pending, chosen = self._interactive_symmetry_split(
                    seed,
                    base,
                    chosen,
                    break_sym_targets,
                )
                self._visited.clear()
            else:
                pending, completed = self._symmetry_split(completed, break_sym_targets)

        if break_sym_targets is not None and not interactive:
            self.results = completed

        self._keep_minimum_results()
        if break_sym_targets is not None:
            self._drop_isomorphic_symmetry_choices(seed, break_sym_targets)

        for result in self.results:
            _compress_pair_labels(result["lgp"])
            if interactive:
                result["base"] = base
                result["choices"] = ";".join(
                    f"{i + base}>>{j + base}" for i, j in zip(chosen[0], chosen[1])
                )

    def _read_index_base(self, base):
        while base not in [0, 1]:
            try:
                base = int(input("select 0 or 1-based indexing [0/1]:"))
            except ValueError:
                pass
        return base

    def _drain_stack(self):
        while self._stack:
            self._solve_partition_path(self._stack.pop())

    def _keep_minimum_results(self):
        if not self.results:
            return
        best = min(result["val"] for result in self.results)
        self.results = [result for result in self.results if result["val"] == best]

    def _drop_isomorphic_symmetry_choices(self, initial_pair, break_sym_targets):
        if len(self.results) < 2:
            return

        if self._dedup_with_automorphism_keys(initial_pair, break_sym_targets):
            return

        merged_initial = _merge_pair(initial_pair)
        kept = []
        seen = set()
        for result in self.results:
            merged = merged_initial.copy()
            shift = len(result["lgp"][0].labels)
            for react_idx in break_sym_targets:
                label = result["lgp"][0].labels[react_idx]
                product_idx = result["lgp"][1].label2idxs[label][0]
                merged.graph[react_idx][product_idx + shift] = 1
                merged.graph[product_idx + shift][react_idx] = 1
            key = canonical_graph_hash(merged, binary=True)
            if key in seen:
                continue
            seen.add(key)
            kept.append(result)
        self.results = kept

    def _dedup_with_automorphism_keys(self, initial_pair, break_sym_targets):
        try:
            if len(initial_pair[0].labels) > self._AUT_DEDUP_MAX_NODES:
                return False

            from ..exact.enumerate import (
                _automorphism_permutations,
                canonical_partial_mapping_key,
            )

            for graph in initial_pair:
                perms = _automorphism_permutations(
                    graph,
                    self.binary,
                    self._AUT_DEDUP_MAX_AUTOMORPHISMS + 1,
                )
                if len(perms) > self._AUT_DEDUP_MAX_AUTOMORPHISMS:
                    return False

            seen = set()
            unique = []
            for result in self.results:
                pairs = []
                for react_idx in break_sym_targets:
                    label = result["lgp"][0].labels[react_idx]
                    product_idxs = result["lgp"][1].label2idxs.get(label, [])
                    if product_idxs:
                        pairs.append((react_idx, product_idxs[0]))
                key = canonical_partial_mapping_key(initial_pair, pairs, self.binary)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(result)
            self.results = unique
            return True
        except Exception:
            return False

    def _symmetry_split(self, completed, targets):
        next_pairs = []
        while self.results:
            result = self.results.pop()
            pair = result["lgp"]
            label = self._next_symmetry_label(pair, targets)
            if label is None:
                completed.append(result)
                continue

            new_label = _fresh_label(pair)
            sols, class_pair = result["lap_sols"][label]
            for sol in sols:
                if len(set(sol["groups_pair"][0])) > 1:
                    continue

                react_class = list(class_pair[0].values())[0]
                react_idx = min(react_class["idxs"])
                product_group = (
                    0
                    if sol["fingerprint"] is None
                    else int(np.where(sol["fingerprint"][0] > 0)[0][0])
                )
                product_class = list(class_pair[1].values())[product_group]

                wl_labels = _initial_wl_labels(pair[1])
                seen_wl = set()
                for product_idx in product_class["idxs"]:
                    wl_label = wl_labels[product_idx]
                    if wl_label in seen_wl:
                        continue
                    seen_wl.add(wl_label)
                    branched = [pair[0].copy(), pair[1].copy()]
                    _assign_singleton_label(branched[0], label, react_idx, new_label)
                    _assign_singleton_label(branched[1], label, product_idx, new_label)

                    key = _label_state_key(branched)
                    if key in self._visited:
                        continue
                    self._clear_cached_labels(branched, [label, new_label])
                    self._visited.add(key)
                    next_pairs.append(branched)
        return next_pairs, completed

    def _next_symmetry_label(self, pair, targets):
        for label in reversed(self._labels_by_size(pair)):
            idxs = pair[0].label2idxs[label]
            if len(idxs) > 1 and idxs[0] in targets:
                return label
        return None

    def _interactive_symmetry_split(self, initial_pair, base, chosen, targets):
        choices = self._collect_interactive_choices(targets)
        selected = None
        min_count = self._INF_INT
        for react_idx in targets:
            candidates = choices[react_idx]
            if 1 < len(candidates) < min_count:
                selected = (react_idx, candidates)
                min_count = len(candidates)

        if selected is None:
            return [], chosen

        react_idx, candidates = selected
        options = ", ".join(str(i + base) for i in sorted(candidates))
        while True:
            try:
                product_idx = (
                    int(input(f"{react_idx + base} >> ? (probably in {options}):"))
                    - base
                )
                break
            except ValueError:
                pass

        chosen[0].append(react_idx)
        chosen[1].append(product_idx)
        constrained = [initial_pair[0].copy(), initial_pair[1].copy()]
        next_label = 1000
        for r_idx, p_idx in zip(chosen[0], chosen[1]):
            while next_label in constrained[0].label2idxs:
                next_label += 1
            constrained[0].labels[r_idx] = next_label
            constrained[1].labels[p_idx] = next_label
            next_label += 1
        constrained[0].build_label2idxs()
        constrained[1].build_label2idxs()
        return [constrained], chosen

    def _collect_interactive_choices(self, targets):
        choices = defaultdict(set)
        for result in self.results:
            pair = result["lgp"]
            for label, react_idxs in pair[0].label2idxs.items():
                if react_idxs[0] not in targets:
                    continue
                sols, class_pair = result["lap_sols"][label]
                left_classes = list(class_pair[0].values())
                right_classes = list(class_pair[1].values())
                n_left = len(left_classes)
                n_right = len(right_classes)
                if n_left == 1 or n_right == 1:
                    fingerprint = np.ones((n_left, n_right), dtype=bool)
                else:
                    fingerprint = np.zeros((n_left, n_right), dtype=bool)
                    for sol in sols:
                        if len(set(sol["groups_pair"][0])) == 1:
                            fingerprint |= sol["fingerprint"] > 0
                for i, left in enumerate(left_classes):
                    for j, right in enumerate(right_classes):
                        if not fingerprint[i, j]:
                            continue
                        for react_idx in left["idxs"]:
                            choices[react_idx].update(right["idxs"])
        return choices

    def _solve_partition_path(self, pair, avoid_multi_sols=None):
        avoid_multi_sols = [1] if avoid_multi_sols is None else avoid_multi_sols
        total = 0
        lap_sols = {}

        for label in self._labels_by_size(pair):
            cached = label in pair[0]._irred_labels
            if cached:
                sols, class_pair = pair[0]._irred_labels[label]
            else:
                sols, class_pair = self._solve_label_block(pair, label)

            total += sols[0]["val"]
            if total > self.minval:
                return

            irreducible = cached or self._should_stop_on_multiple(
                pair, label, sols, avoid_multi_sols
            )
            if not irreducible:
                for sol in sols:
                    if not sol["proper"]:
                        continue
                    if len(set(sol["groups_pair"][0])) == 1:
                        irreducible = True
                        break
                    branched = self._refine_by_solution(pair, label, sol, class_pair)
                    key = _label_state_key(branched)
                    if key not in self._visited:
                        self._visited.add(key)
                        self._stack.append(branched)

            if self._keep_lap_sols:
                lap_sols[label] = (sols, _classes_as_legacy_pair(class_pair))
            if irreducible:
                pair[0]._irred_labels.setdefault(label, (sols, class_pair))
                continue
            return

        self.results.append({"lgp": pair, "val": total, "lap_sols": lap_sols})
        self.minval = min(self.minval, total)

    def _should_stop_on_multiple(self, pair, label, sols, avoid_multi_sols):
        initial_label = pair[0]._ini_labels[pair[0].label2idxs[label][0]]
        if initial_label not in avoid_multi_sols:
            return False
        return sum(1 for sol in sols if sol["proper"]) > 1

    def _labels_by_size(self, pair):
        return sorted(
            pair[0].label2idxs, key=lambda label: -len(pair[0].label2idxs[label])
        )

    def _solve_label_block(self, pair, label):
        key = self._label_block_key(pair, label)
        if key is not None and key in self._label_block_cache:
            cost, masks, n_classes, class_pair = self._label_block_cache[key]
            return self._enumerate_lap_fingerprints(cost, masks, n_classes), class_pair

        cost, masks, n_classes, class_pair = self._cost_matrix(pair, label)
        if key is not None:
            self._label_block_cache[key] = (cost, masks, n_classes, class_pair)
        return self._enumerate_lap_fingerprints(cost, masks, n_classes), class_pair

    def _label_block_key(self, pair, label):
        if not self.cache_label_blocks:
            return None
        return (label, tuple(pair[0].labels), tuple(pair[1].labels))

    def _cost_matrix(self, pair, label):
        blocks = [self._partition_block(graph, label) for graph in pair]
        left_classes, left_offsets = blocks[0]
        right_classes, right_offsets = blocks[1]
        cost = np.empty((left_offsets[-1], right_offsets[-1]), dtype=int)
        for left_pos, left in enumerate(left_classes):
            a0, a1 = left_offsets[left_pos], left_offsets[left_pos + 1]
            left_nbrs = left[2]
            for right_pos, right in enumerate(right_classes):
                b0, b1 = right_offsets[right_pos], right_offsets[right_pos + 1]
                cost[a0:a1, b0:b1] = self._neighborhood_cost(left_nbrs, right[2])

        n_classes = [len(left_classes), len(right_classes)]
        masks = (
            [
                _block_mask(n_classes[0], left_offsets),
                _block_mask(n_classes[1], right_offsets),
            ]
            if n_classes[0] > 1 and n_classes[1] > 1
            else [None, None]
        )
        return cost, masks, n_classes, [left_classes, right_classes]

    def _partition_block(self, graph, label):
        classes = self._next_classes(graph, label)
        offsets = [0]
        for _, idxs, _ in classes:
            offsets.append(offsets[-1] + len(idxs))

        return classes, offsets

    def _next_classes(self, graph, label):
        buckets = {}
        labels = graph.labels
        adjacency = graph.graph
        for idx in graph.label2idxs[label]:
            if self.binary:
                nbrs = {}
                for nbr in adjacency.get(idx, {}):
                    nbr_label = labels[nbr]
                    nbrs[nbr_label] = nbrs.get(nbr_label, 0) + 1
                if self.deterministic_labels:
                    signature = (label, tuple(sorted(nbrs.items())))
                else:
                    signature = (label, frozenset(nbrs.items()))
            else:
                nbrs = defaultdict(list)
                for nbr, weight in adjacency.get(idx, {}).items():
                    nbrs[labels[nbr]].append(weight)
                nbrs = {
                    nbr_label: sorted(weights, reverse=True)
                    for nbr_label, weights in nbrs.items()
                }
                signature = (
                    label,
                    tuple(
                        (nbr_label, tuple(weights))
                        for nbr_label, weights in sorted(nbrs.items())
                    ),
                )
            color = self._label_token(signature)
            if color not in buckets:
                buckets[color] = [color, [], nbrs]
            buckets[color][1].append(idx)
        return list(buckets.values())

    def _label_token(self, signature):
        if self.deterministic_labels:
            return stable_int_token(signature)
        return hash(signature) & 0x7FFFFFFFFFFFFFFF

    def _neighborhood_cost(self, left, right):
        if self.binary:
            cost = 0
            for label, count in left.items():
                other = right.get(label)
                cost += count if other is None else abs(count - other)
            for label, count in right.items():
                if label not in left:
                    cost += count
            return cost

        cost = 0
        for label, weights in left.items():
            cost += self._weight_multiset_cost(weights, right.get(label, []))
        for label, weights in right.items():
            if label not in left:
                cost += sum(weights)
        return cost

    def _weight_multiset_cost(self, left, right):
        if self.binary:
            return abs(left - right)

        count = min(len(left), len(right))
        paired = sum(abs(left[i] - right[i]) for i in range(count))
        return paired + sum(left[count:]) + sum(right[count:])

    def _enumerate_lap_fingerprints(self, cost, masks, n_classes):
        if n_classes[0] == 1 or n_classes[1] == 1:
            row, col = linear_sum_assignment(cost)
            sols = [
                {
                    "row": row,
                    "col": col,
                    "val": np.sum(cost[row, col]),
                    "fingerprint": None,
                    "groups_pair": [[0] * n_classes[0], [0] * n_classes[1]],
                }
            ]
        else:
            sols = self._enumerate_structural_fingerprints(cost, masks, n_classes)

        self._mark_proper_solutions(sols)
        return sols

    def _enumerate_structural_fingerprints(self, cost, masks, n_classes):
        solutions = []
        best = None
        perturbed = 100 * cost
        for iteration in range(self.max_lap_fingerprints):
            row, col = linear_sum_assignment(perturbed)
            value = np.sum(cost[row, col])
            if iteration == 0:
                best = value
            if value != best:
                break

            fingerprint = masks[0][:, row] @ masks[1][:, col].T
            if any(np.all(fingerprint == sol["fingerprint"]) for sol in solutions):
                break

            solutions.append(
                {
                    "row": row,
                    "col": col,
                    "val": value,
                    "fingerprint": fingerprint,
                    "groups_pair": _fingerprint_components(fingerprint, n_classes),
                }
            )
            perturbed += ((masks[0].T @ fingerprint @ masks[1]) > 0).astype(int)
        return solutions

    def _mark_proper_solutions(self, sols):
        if not sols:
            return
        group_rows = [sol["groups_pair"][0] + sol["groups_pair"][1] for sol in sols]
        width = len(group_rows[0])
        for sol in sols:
            sol["proper"] = True

        for i, row_i in enumerate(group_rows):
            for j, row_j in enumerate(group_rows):
                if i == j or not sols[j]["proper"]:
                    continue
                image = defaultdict(set)
                for idx in range(width):
                    image[row_i[idx]].add(row_j[idx])
                if all(len(values) == 1 for values in image.values()):
                    sols[i]["proper"] = False
                    break

    def _refine_by_solution(self, pair, label, sol, class_pair):
        refined = [pair[0].copy(), pair[1].copy()]

        group_to_label = {}
        for group, (next_label, _, _) in zip(sol["groups_pair"][1], class_pair[1]):
            group_to_label[group] = next_label

        fresh = _fresh_label(refined)
        for group in sol["groups_pair"][0]:
            if group not in group_to_label:
                group_to_label[group] = fresh
                fresh += 1

        for side in range(2):
            merged = defaultdict(list)
            for group, (next_label, idxs, _) in zip(
                sol["groups_pair"][side], class_pair[side]
            ):
                merged[group_to_label[group]].extend(idxs)

            del refined[side].label2idxs[label]
            for next_label, idxs in merged.items():
                refined[side].label2idxs[next_label] = sorted(idxs)
                for idx in idxs:
                    refined[side].labels[idx] = next_label

        self._clear_cached_labels(refined, group_to_label.values())
        return refined

    def _clear_cached_labels(self, pair, parent_labels):
        neighboring_labels = defaultdict(set)
        for parent in parent_labels:
            for graph in pair:
                for idx in graph.label2idxs[parent]:
                    neighboring_labels[parent].update(
                        graph.labels[nbr] for nbr in graph.graph[idx]
                    )

        dirty = set(parent_labels)
        for left_label, left_neighbors in neighboring_labels.items():
            for right_label, right_neighbors in neighboring_labels.items():
                if left_label > right_label:
                    dirty.update(left_neighbors & right_neighbors)

        for label in dirty:
            pair[0]._irred_labels.pop(label, None)


def _classes_as_legacy_pair(class_pair):
    return [_classes_as_legacy_dict(classes) for classes in class_pair]


def _block_mask(n_classes, offsets):
    mask = np.zeros((n_classes, offsets[-1]), dtype=int)
    for row in range(n_classes):
        mask[row, offsets[row] : offsets[row + 1]] = 1
    return mask


def _classes_as_legacy_dict(classes):
    return {
        color: {
            "idxs": list(idxs),
            "nbrs": {
                label: ([1] * weights if isinstance(weights, int) else list(weights))
                for label, weights in nbrs.items()
            },
        }
        for color, idxs, nbrs in classes
    }


def _fingerprint_components(fingerprint, n_classes):
    n_left, n_right = n_classes
    groups_left = [-1] * n_left
    groups_right = [-1] * n_right
    group_id = 0

    for start in range(n_left):
        if groups_left[start] != -1:
            continue
        frontier_left = {start}
        seen_left = set()
        seen_right = set()
        while frontier_left:
            fresh_left = frontier_left - seen_left
            if not fresh_left:
                break
            frontier_right = set()
            for left in fresh_left:
                groups_left[left] = group_id
                frontier_right.update(np.where(fingerprint[left] > 0)[0])
            seen_left.update(fresh_left)

            fresh_right = frontier_right - seen_right
            if not fresh_right:
                break
            frontier_left = set()
            for right in fresh_right:
                groups_right[right] = group_id
                frontier_left.update(np.where(fingerprint.T[right] > 0)[0])
            seen_right.update(fresh_right)
        group_id += 1

    for right in range(n_right):
        if groups_right[right] == -1:
            groups_right[right] = group_id
            group_id += 1

    return [groups_left, groups_right]


def _fresh_label(pair):
    labels = set(pair[0].label2idxs) | set(pair[1].label2idxs)
    return max(labels, default=999) + 1


def _initial_wl_labels(graph):
    if graph._ini_wl_labels is None:
        graph._ini_wl_labels = graph.get_WL_labels()
    return graph._ini_wl_labels


def _assign_singleton_label(graph, old_label, idx, new_label):
    graph.labels[idx] = new_label
    graph.label2idxs[old_label].remove(idx)
    graph.label2idxs[new_label].append(idx)


def _merge_pair(pair):
    shift = len(pair[0].labels)
    left = {idx: dict(nbrs) for idx, nbrs in pair[0].graph.items()}
    right = {
        idx + shift: {nbr + shift: weight for nbr, weight in nbrs.items()}
        for idx, nbrs in pair[1].graph.items()
    }
    return LabeledGraph({**left, **right}, pair[0].labels + pair[1].labels)


def _compress_pair_labels(pair):
    old_to_new = {}
    next_label = 1
    for label in pair[0].labels:
        if label not in old_to_new:
            old_to_new[label] = next_label
            next_label += 1

    for graph in pair:
        graph.labels = [old_to_new[label] for label in graph.labels]
        graph.build_label2idxs()


def _label_state_key(pair):
    old_to_new = {}
    next_label = 1
    key = []
    for graph in pair:
        for label in graph.labels:
            if label not in old_to_new:
                old_to_new[label] = next_label
                next_label += 1
            key.append(old_to_new[label])
    return tuple(key)
