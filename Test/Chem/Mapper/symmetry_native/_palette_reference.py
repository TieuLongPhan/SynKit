"""Frozen, uncached constructor oracle predating production memoization.

This intentional test-only copy establishes exact palette/validation equivalence.
It must not delegate palette preparation to the production constructor.
Only the dimension guard is parameterized for larger pure-Python palette
fixtures; that does not grant capacity to an older native library.
"""

import ctypes as ct
import json
import time
from pathlib import Path

import numpy as np

from synkit.Chem.Mapper._native.classification import (
    NativeITSBatchClassifier,
    _INT,
    _Problem,
    _load,
)
from synkit.Chem.Mapper.reporting.spectrum import _typed
from synkit.Graph.Canon.exact import _encode_color


class UncachedReferenceClassifier(NativeITSBatchClassifier):
    MAX_VERTICES = 256

    def __init__(
        self,
        reactant,
        product,
        elements,
        properties,
        *,
        library_path,
        edge_properties=None,
    ):
        preparation_started = time.perf_counter()
        a, b = np.asarray(reactant), np.asarray(product)
        self.n = n = len(elements)
        if not 0 <= n <= self.MAX_VERTICES or a.shape != (n, n) or b.shape != (n, n):
            raise ValueError(
                f"Native ITS requires square endpoints with 0..{self.MAX_VERTICES} vertices"
            )
        for matrix in (a, b):
            if (
                not np.isfinite(matrix).all()
                or np.any(matrix < 0)
                or not np.array_equal(matrix, matrix.T)
                or np.any(np.diag(matrix))
            ):
                raise ValueError(
                    "ITS endpoints must be finite nonnegative symmetric and loop-free"
                )
        for name, sides in properties.items():
            if len(sides) != 2 or any(len(side) != n for side in sides):
                raise ValueError(f"Invalid atom property {name}")
        edge_properties = edge_properties or {}
        for name, sides in edge_properties.items():
            if len(sides) != 2 or any(np.shape(side) != (n, n) for side in sides):
                raise ValueError(f"Invalid edge property {name}")
            if any(
                any(side[i][j] != side[j][i] for i in range(n) for j in range(n))
                for side in sides
            ):
                raise ValueError(f"Asymmetric edge property {name}")
        node_tokens = [
            [
                _encode_color(
                    (
                        _typed(elements[i]),
                        tuple(
                            (name, _typed(left[i]), _typed(right[j]))
                            for name, (left, right) in properties.items()
                        ),
                        (),
                    )
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        states, matrices = [], []
        for side, matrix in enumerate((a, b)):
            raw = [
                [
                    (
                        float(matrix[i, j]),
                        *(values[side][i][j] for values in edge_properties.values()),
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
            # Intern exact tokens rather than Python equality: bool/int and
            # signed floating zero may differ in the certificate contract.
            encoded_states = {
                _encode_color(value): value for row in raw for value in row
            }
            palette = [encoded_states[token] for token in sorted(encoded_states)]
            if not palette:
                palette = [(0.0,)]
            ids = {_encode_color(value): index for index, value in enumerate(palette)}
            states.append(palette)
            matrices.append(
                np.ascontiguousarray(
                    [[ids[_encode_color(v)] for v in row] for row in raw],
                    dtype=np.int32,
                ).reshape(n, n)
            )
        edge_tokens = []
        for left in states[0]:
            row = []
            for right in states[1]:
                raw = (
                    left[0],
                    right[0],
                    *(v for pair in zip(left[1:], right[1:]) for v in pair),
                )
                row.append(_encode_color(raw) if left[0] or right[0] else None)
            edge_tokens.append(row)
        tokens = sorted(
            {v for row in node_tokens for v in row}
            | {v for row in edge_tokens for v in row if v is not None}
        )
        if len(tokens) >= 65536:
            raise ValueError("Native ITS exact palette exceeds uint16")
        ids = {token: index for index, token in enumerate(tokens)}
        self.node_colors = np.ascontiguousarray(
            [[ids[v] for v in row] for row in node_tokens], dtype=np.int32
        ).reshape(n, n)
        self.edge_colors = np.ascontiguousarray(
            [[ids[v] + 1 if v is not None else 0 for v in row] for row in edge_tokens],
            dtype=np.int32,
        )
        self.edge_matrices = matrices
        self.palette = (ct.c_char_p * len(tokens))(
            *(
                json.dumps(token, separators=(",", ":")).encode("ascii")
                for token in tokens
            )
        )
        self.problem = _Problem(
            n,
            self.node_colors.ctypes.data_as(_INT),
            matrices[0].ctypes.data_as(_INT),
            matrices[1].ctypes.data_as(_INT),
            len(states[0]),
            len(states[1]),
            self.edge_colors.ctypes.data_as(_INT),
            len(tokens),
            self.palette,
        )
        self.library = _load(str(Path(library_path).resolve()))
        self.preparation_seconds = time.perf_counter() - preparation_started
