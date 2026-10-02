"""Encode controlled ITS colors directly into the exact native canonicalizer."""

import ctypes
import os
import struct
import time
import weakref
from collections import Counter, OrderedDict
from itertools import pairwise
from pathlib import Path

import numpy as np

from synkit.Graph.Canon.exact import _encode_color

from ..spectrum import _typed
from .native_canonical import (
    PackedCanonicalKey,
    _canonical_function,
    _packed_palette,
    pack_canonical_key,
)


class NativeITSCanonicalizer:
    """Equivalent to _attributed_its_graph followed by native canonicalization.

    The palettes are ordered by the existing exact encoded color tokens. Gaps
    in palette IDs do not change refinement or certificate ordering. Zero is
    reserved for an absent edge, including the deliberately absent diagonal.
    """

    def __init__(self, observer, generators, library_path):
        self.profile = bool(os.environ.get("SYNKIT_NATIVE_PROFILE"))
        self.timings = Counter()
        self.observer = observer
        self.n = len(observer.reactant)
        self.generators = np.ascontiguousarray(generators, dtype=np.int32)
        self.pointer = ctypes.POINTER(ctypes.c_int32)
        self.library, self.function = _canonical_function(
            str(Path(library_path).resolve())
        )
        _, self.certificate_function = _canonical_function(
            str(Path(library_path).resolve()), require_group=False
        )
        self.edge_function = getattr(self.library, "synkit_its_sparse_edges", None)
        if self.edge_function is not None:
            self.edge_function.argtypes = [
                ctypes.c_int,
                self.pointer,
                self.pointer,
                self.pointer,
            ]
            self.edge_function.restype = ctypes.c_int
        self.edge_buffer = np.empty(
            self.n * max(0, self.n - 1) // 2 * 3, dtype=np.int32
        )
        self.pair_function = getattr(self.library, "synkit_its_pair", None)
        if self.pair_function is not None:
            self.pair_function.argtypes = [
                ctypes.c_int,
                self.pointer,
                self.pointer,
                self.pointer,
                ctypes.c_int,
                self.pointer,
                self.pointer,
            ]
            self.pair_function.restype = ctypes.c_int
        self.packing_palettes = OrderedDict()
        self.template_packing_palettes = OrderedDict()
        self.typed_elements = tuple(_typed(v) for v in observer.structure.elements)
        self.node_parts, palette = {}, {}
        self.encoded = {}
        self.seed_cache = OrderedDict()
        raw_matrix = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                unary = tuple(
                    (name, _typed(left[i]), _typed(right[j]))
                    for name, (left, right) in observer.properties.items()
                )
                part = self.typed_elements[i], unary
                self.node_parts[i, j] = part
                raw = (*part, ())
                token = self.encode(raw)
                palette[token] = None
                row.append(token)
            raw_matrix.append(row)
        baseline_tokens = []
        for i in range(self.n):
            unary = tuple(
                (name, _typed(left[i]), _typed(left[i]))
                for name, (left, _right) in observer.properties.items()
            )
            token = self.encode((self.typed_elements[i], unary, ()))
            baseline_tokens.append(token)
            palette[token] = None
        self.node_tokens = tuple(sorted(palette))
        ids = {token: i for i, token in enumerate(self.node_tokens)}
        self.node_matrix = np.array(
            [[ids[v] for v in row] for row in raw_matrix], dtype=np.int32
        )
        self.baseline_colors = np.array(
            [ids[token] for token in baseline_tokens], dtype=np.int32
        )
        levels = tuple(
            sorted(set(observer.reactant.ravel()) | set(observer.product.ravel()))
        )
        values = {value: i for i, value in enumerate(levels)}
        self.a = np.array(
            [[values[v] for v in row] for row in observer.reactant], dtype=np.int32
        )
        self.b = np.array(
            [[values[v] for v in row] for row in observer.product], dtype=np.int32
        )
        pairs = {
            (i, j): self.encode((float(a), float(b)))
            for i, a in enumerate(levels)
            for j, b in enumerate(levels)
            if a != 0 or b != 0
        }
        self.edge_tokens = (None,) + tuple(sorted(set(pairs.values())))
        ids = {
            token: i for i, token in enumerate(self.edge_tokens) if token is not None
        }
        self.pair_ids = np.zeros((len(levels), len(levels)), dtype=np.int32)
        for pair, token in pairs.items():
            self.pair_ids[pair] = ids[token]
        # Reporting equality intentionally differs from typed canonical colors.
        self.unary_changes = np.array(
            [
                [
                    any(
                        left[i] != right[j]
                        for left, right in observer.properties.values()
                    )
                    for j in range(self.n)
                ]
                for i in range(self.n)
            ],
            dtype=np.int32,
        )
        self.changed_colors = np.zeros(len(self.edge_tokens), dtype=np.int32)
        bond_values = {0: (0.0, 0.0)}
        for (i, j), token in pairs.items():
            color = ids[token]
            bond_values[color] = (float(levels[i]), float(levels[j]))
            self.changed_colors[color] = not np.isclose(
                levels[i], levels[j], atol=observer.tolerance, rtol=0
            )
        resources = sorted(
            {
                (element, *bond_values[color])
                for element in self.typed_elements
                for color in range(1, len(self.edge_tokens))
            }
        )
        self.boundary_tokens = tuple(resources)
        resource_ids = {value: i for i, value in enumerate(resources)}
        self.boundary_resources = np.array(
            [
                [0]
                + [
                    resource_ids[(element, *bond_values[color])]
                    for color in range(1, len(self.edge_tokens))
                ]
                for element in self.typed_elements
            ],
            dtype=np.int32,
        )
        self.changes_function = getattr(self.library, "synkit_its_changes", None)
        if self.changes_function is not None:
            self.changes_function.argtypes = [
                ctypes.c_int,
                self.pointer,
                self.pointer,
                self.pointer,
                self.pointer,
                ctypes.c_int,
                self.pointer,
            ]
            self.changes_function.restype = ctypes.c_int
            self.changes_buffer = np.empty(self.n * (self.n + 1), dtype=np.int32)
            self.boundary_function = self.library.synkit_its_boundary
            self.boundary_function.argtypes = [
                ctypes.c_int,
                self.pointer,
                ctypes.c_int,
                self.pointer,
                ctypes.c_int,
                self.pointer,
                self.pointer,
                self.pointer,
            ]
            self.boundary_function.restype = ctypes.c_int
            self.boundary_offsets = np.empty(self.n + 1, dtype=np.int32)
            self.boundary_buffer = np.empty(self.n * self.n, dtype=np.int32)
        self.indices = np.arange(self.n)
        self.full_seeds = self.generators
        self.pattern_cache = None
        if (
            hasattr(self.library, "synkit_pattern_cache_create")
            and os.environ.get("SYNKIT_NATIVE_PATTERN_CACHE", "1") != "0"
        ):
            create = self.library.synkit_pattern_cache_create
            create.argtypes = [
                ctypes.c_int,
                self.pointer,
                self.pointer,
                self.pointer,
                ctypes.c_int,
                ctypes.c_int,
            ]
            create.restype = ctypes.c_void_p
            baseline_edges = np.ascontiguousarray(self.pair_ids[self.a, self.a])
            np.fill_diagonal(baseline_edges, 0)
            self.pattern_cache = create(
                self.n,
                self.baseline_colors.ctypes.data_as(self.pointer),
                baseline_edges.ctypes.data_as(self.pointer),
                self.generators.ctypes.data_as(self.pointer),
                len(self.generators),
                64,
            )
            if not self.pattern_cache:
                raise RuntimeError("native pattern cache initialization failed")
            self.pattern_lookup = self.library.synkit_pattern_cache_lookup
            self.pattern_lookup.argtypes = [ctypes.c_void_p, self.pointer, self.pointer]
            self.pattern_lookup.restype = ctypes.c_int
            self.pattern_remember = self.library.synkit_pattern_cache_remember
            self.pattern_remember.argtypes = [ctypes.c_void_p]
            self.pattern_remember.restype = ctypes.c_int
            destroy = self.library.synkit_pattern_cache_destroy
            destroy.argtypes = [ctypes.c_void_p]
            destroy.restype = None
            self._pattern_finalizer = weakref.finalize(
                self, destroy, self.pattern_cache
            )

        # Instance-local scratch is consumed synchronously, like edge_buffer.
        # Published keys copy their bytes and never retain these arrays.
        self._canonical_order = np.empty(self.n, dtype=np.int32)
        self._canonical_group = ctypes.c_uint64()
        self._canonical_visited = ctypes.c_int64()
        self._canonical_group_pointer = ctypes.byref(self._canonical_group)
        self._canonical_visited_pointer = ctypes.byref(self._canonical_visited)
        self._empty_seeds = np.empty((0,), dtype=np.int32)
        for name in (
            "a", "b", "pair_ids", "edge_buffer", "unary_changes",
            "changed_colors", "changes_buffer", "boundary_resources",
            "boundary_offsets", "boundary_buffer", "_canonical_order",
            "full_seeds", "_empty_seeds",
        ):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name + "_pointer", value.ctypes.data_as(self.pointer))

    def changes(self, mapping, paired):
        """Return reporting unary changes, changed bonds, and exact context."""
        if self.changes_function is None:
            return None
        permutation = np.ascontiguousarray(mapping, dtype=np.int32)
        count = self.changes_function(
            self.n,
            permutation.ctypes.data_as(self.pointer),
            self.unary_changes_pointer,
            paired.ctypes.data_as(self.pointer),
            self.changed_colors_pointer,
            min(self.n, self.observer.structure.template_radius),
            self.changes_buffer_pointer,
        )
        if count < 0:
            raise RuntimeError("native ITS change extraction failed")
        buffer = self.changes_buffer
        atoms = np.flatnonzero(buffer[: self.n]).tolist()
        context = np.flatnonzero(buffer[self.n : 2 * self.n]).tolist()
        bonds = [
            tuple(pair)
            for pair in buffer[2 * self.n : 2 * self.n + 2 * count]
            .reshape(-1, 2)
            .tolist()
        ]
        return atoms, bonds, context

    def boundaries(self, selected, matrix):
        indices = np.ascontiguousarray(selected, dtype=np.int32)
        status = self.boundary_function(
            self.n,
            indices.ctypes.data_as(self.pointer),
            len(selected),
            matrix.ctypes.data_as(self.pointer),
            len(self.edge_tokens),
            self.boundary_resources_pointer,
            self.boundary_offsets_pointer,
            self.boundary_buffer_pointer,
        )
        if status < 0:
            raise RuntimeError("native ITS boundary extraction failed")
        resources = self.boundary_tokens
        values = self.boundary_buffer[:status].tolist()
        offsets = self.boundary_offsets[: len(selected) + 1].tolist()
        return [
            tuple(resources[index] for index in values[left:right])
            for left, right in pairwise(offsets)
        ]

    def known_pattern(self, mapping, paired):
        if self.pattern_cache is None:
            return False
        colors = self.node_matrix[self.indices, mapping]
        status = self.pattern_lookup(
            self.pattern_cache,
            colors.ctypes.data_as(self.pointer),
            paired.ctypes.data_as(self.pointer),
        )
        if status < 0:
            raise RuntimeError("native pattern cache lookup failed")
        return bool(status)

    def remember_pattern(self):
        if self.pattern_cache is not None and self.pattern_remember(self.pattern_cache):
            raise RuntimeError("native pattern cache update failed")

    def encode(self, raw):
        token = self.encoded.get(raw)
        if token is None:
            token = _encode_color(raw)
            self.encoded[raw] = token
        return token

    def context_seeds(self, selected):
        if selected in self.seed_cache:
            self.seed_cache.move_to_end(selected)
            return self.seed_cache[selected]
        positions = {node: i for i, node in enumerate(selected)}
        seeds = []
        for generator in self.generators:
            if all(int(generator[node]) in positions for node in selected):
                seeds.append([positions[int(generator[node])] for node in selected])
        result = np.ascontiguousarray(seeds, dtype=np.int32)
        self.seed_cache[selected] = result
        if len(self.seed_cache) > 256:
            self.seed_cache.popitem(last=False)
        return result

    def paired_matrix(self, mapping):
        if self.pair_function is not None:
            permutation = np.ascontiguousarray(mapping, dtype=np.int32)
            matrix = np.empty((self.n, self.n), dtype=np.int32)
            status = self.pair_function(
                self.n,
                permutation.ctypes.data_as(self.pointer),
                self.a_pointer,
                self.b_pointer,
                len(self.pair_ids),
                self.pair_ids_pointer,
                matrix.ctypes.data_as(self.pointer),
            )
            if status:
                raise RuntimeError("native ITS pairing failed")
            return matrix
        matrix = self.pair_ids[self.a, self.b[np.ix_(mapping, mapping)]]
        np.fill_diagonal(matrix, 0)
        return matrix

    def packed_key(self, colors, tokens, order, matrix, full):
        n = len(colors)
        count = self.edge_function(
            n,
            order.ctypes.data_as(self.pointer),
            matrix.ctypes.data_as(self.pointer),
            self.edge_buffer_pointer,
        )
        if count < 0:
            raise RuntimeError("native canonical edge extraction failed")
        edges = self.edge_buffer[: 3 * count].reshape(-1, 3)
        node_ids = (
            tuple(sorted(set(colors.tolist()))) if full else tuple(range(len(tokens)))
        )
        edge_ids = tuple(sorted(set(edges[:, 2].tolist())))
        # Template numeric IDs are local to tokens; bind the exact token tuple.
        # This reuses only injective encoding tables, never search results.
        cache_key = (node_ids, edge_ids) if full else (tokens, edge_ids)
        palettes = self.packing_palettes if full else self.template_packing_palettes
        stored = palettes.get(cache_key)
        if stored is None:
            palette = tuple(
                sorted(
                    {tokens[index] for index in node_ids}
                    | {self.edge_tokens[index] for index in edge_ids}
                )
            )
            payload, positions = _packed_palette(palette)
            node_map = np.zeros(len(tokens), dtype="<u2")
            edge_map = np.zeros(len(self.edge_tokens), dtype="<u2")
            for index in node_ids:
                node_map[index] = positions[tokens[index]]
            for index in edge_ids:
                edge_map[index] = positions[self.edge_tokens[index]]
            stored = (palette, payload, node_map, edge_map)
            palettes[cache_key] = stored
            if len(palettes) > 256:
                palettes.popitem(last=False)
        palette, payload, node_map, edge_map = stored
        values = np.empty(n + 3 * count, dtype="<u2")
        values[:n] = node_map[colors[order]]
        output = values[n:].reshape(-1, 3)
        output[:, :2] = edges[:, :2]
        output[:, 2] = edge_map[edges[:, 2]]
        key = PackedCanonicalKey(
            b"SKC1" + struct.pack("<IH", len(payload), n) + payload + values.tobytes()
        )
        object.__setattr__(key, "_palette", palette)
        return key

    def canonical(
        self,
        mapping,
        transported,
        context=None,
        *,
        paired=None,
        require_group=True,
        packed=False,
    ):
        started = time.perf_counter() if self.profile else 0
        matrix = self.paired_matrix(mapping) if paired is None else paired
        if context is None:
            colors = self.node_matrix[self.indices, mapping]
            tokens = self.node_tokens
            seeds = self.full_seeds
        else:
            selected = tuple(sorted(context))
            if not selected:
                return ((), ()), 1
            if self.changes_function is not None:
                boundaries = self.boundaries(selected, matrix)
            else:
                outside = np.ones(self.n, dtype=bool)
                outside[list(selected)] = False
                a = self.observer.reactant
                boundaries = [
                    tuple(
                        sorted(
                            (
                                self.typed_elements[int(j)],
                                float(a[node, j]),
                                float(transported[node, j]),
                            )
                            for j in np.flatnonzero(outside & (matrix[node] != 0))
                        )
                    )
                    for node in selected
                ]
            node_tokens = [
                self.encode((*self.node_parts[node, mapping[node]], boundary))
                for node, boundary in zip(selected, boundaries)
            ]
            tokens = tuple(sorted(set(node_tokens)))
            ids = {token: i for i, token in enumerate(tokens)}
            colors = np.array([ids[token] for token in node_tokens], dtype=np.int32)
            matrix = np.ascontiguousarray(matrix[np.ix_(selected, selected)])
            # These are optional pruning witnesses. Certificate-only template
            # search remains complete without constructing restricted R seeds.
            seeds = (
                self.context_seeds(selected)
                if require_group
                else self._empty_seeds
            )
        n = len(colors)
        if not n:
            return ((), ()), 1
        order = self._canonical_order[:n]
        group = self._canonical_group
        config = self.observer.structure
        function = self.function if require_group else self.certificate_function
        native_started = time.perf_counter() if self.profile else 0
        status = function(
            n,
            colors.ctypes.data_as(self.pointer),
            matrix.ctypes.data_as(self.pointer),
            (self.full_seeds_pointer if seeds is self.full_seeds else
             self._empty_seeds_pointer if seeds is self._empty_seeds else
             seeds.ctypes.data_as(self.pointer)),
            len(seeds),
            config.timeout_seconds,
            (1 << 63) - 1
            if config.max_search_nodes is None
            else config.max_search_nodes,
            self._canonical_order_pointer,
            self._canonical_group_pointer,
            self._canonical_visited_pointer,
        )
        native_finished = time.perf_counter() if self.profile else 0
        if status:
            raise RuntimeError(
                f"native ITS canonical search incomplete or failed ({status})"
            )
        if packed and self.edge_function is not None:
            key = self.packed_key(colors, tokens, order, matrix, context is None)
        else:
            inverse = np.empty(n, dtype=np.int32)
            inverse[order] = np.arange(n)
            left, right = np.nonzero(np.triu(matrix, 1))
            edges = tuple(
                sorted(
                    (
                        min(int(inverse[i]), int(inverse[j])),
                        max(int(inverse[i]), int(inverse[j])),
                        self.edge_tokens[int(matrix[i, j])],
                    )
                    for i, j in zip(left, right)
                )
            )
            key = (tuple(tokens[int(colors[i])] for i in order), edges)
            if packed:
                key = PackedCanonicalKey(pack_canonical_key(key))
        if self.profile:
            kind = "full" if context is None else "template"
            self.timings[kind + "_calls"] += 1
            self.timings[kind + "_encoding_seconds"] += native_started - started
            self.timings[kind + "_native_seconds"] += native_finished - native_started
            self.timings[kind + "_certificate_seconds"] += (
                time.perf_counter() - native_finished
            )
        return key, group.value
