"""Encode controlled ITS colors directly into the exact native canonicalizer."""

import ctypes
import os
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from synkit.Graph.Canon.exact import _encode_color

from ..spectrum import _typed
from .native_canonical import _canonical_function


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
        self.node_tokens = tuple(sorted(palette))
        ids = {token: i for i, token in enumerate(self.node_tokens)}
        self.node_matrix = np.array(
            [[ids[v] for v in row] for row in raw_matrix], dtype=np.int32
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
        self.indices = np.arange(self.n)
        self.full_seeds = self.generators

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
        matrix = self.pair_ids[self.a, self.b[np.ix_(mapping, mapping)]]
        np.fill_diagonal(matrix, 0)
        return matrix

    def canonical(
        self, mapping, transported, context=None, *, paired=None, require_group=True
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
            outside = np.ones(self.n, dtype=bool)
            outside[list(selected)] = False
            node_tokens = []
            a = self.observer.reactant
            for node in selected:
                boundary = tuple(
                    sorted(
                        (
                            self.typed_elements[int(j)],
                            float(a[node, j]),
                            float(transported[node, j]),
                        )
                        for j in np.flatnonzero(outside & (matrix[node] != 0))
                    )
                )
                node_tokens.append(
                    self.encode((*self.node_parts[node, mapping[node]], boundary))
                )
            tokens = tuple(sorted(set(node_tokens)))
            ids = {token: i for i, token in enumerate(tokens)}
            colors = np.array([ids[token] for token in node_tokens], dtype=np.int32)
            matrix = np.ascontiguousarray(matrix[np.ix_(selected, selected)])
            seeds = self.context_seeds(selected)
        n = len(colors)
        if not n:
            return ((), ()), 1
        order = np.empty(n, dtype=np.int32)
        group, visited = ctypes.c_uint64(), ctypes.c_int64()
        config = self.observer.structure
        function = self.function if require_group else self.certificate_function
        native_started = time.perf_counter() if self.profile else 0
        status = function(
            n,
            colors.ctypes.data_as(self.pointer),
            matrix.ctypes.data_as(self.pointer),
            seeds.ctypes.data_as(self.pointer),
            len(seeds),
            config.timeout_seconds,
            (1 << 63) - 1
            if config.max_search_nodes is None
            else config.max_search_nodes,
            order.ctypes.data_as(self.pointer),
            ctypes.byref(group),
            ctypes.byref(visited),
        )
        native_finished = time.perf_counter() if self.profile else 0
        if status:
            raise RuntimeError(
                f"native ITS canonical search incomplete or failed ({status})"
            )
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
        if self.profile:
            kind = "full" if context is None else "template"
            self.timings[kind + "_calls"] += 1
            self.timings[kind + "_encoding_seconds"] += native_started - started
            self.timings[kind + "_native_seconds"] += native_finished - native_started
            self.timings[kind + "_certificate_seconds"] += (
                time.perf_counter() - native_finished
            )
        return key, group.value
