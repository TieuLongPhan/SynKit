"""Prepare exact palettes once; construct, canonically label and dedup ITS in C++."""

from __future__ import annotations

import ctypes as ct
import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from synkit.Graph.Canon.exact import _encode_color

from ..exact.native.canonical import PackedCanonicalKey
from ...reporting.spectrum import _typed

_INT = ct.POINTER(ct.c_int32)
_LONG = ct.POINTER(ct.c_int64)


class _Problem(ct.Structure):
    _fields_ = [
        ("n", ct.c_int32),
        ("node_pair_colors", _INT),
        ("edges_r", _INT),
        ("edges_p", _INT),
        ("edge_states_r", ct.c_int32),
        ("edge_states_p", ct.c_int32),
        ("edge_pair_colors", _INT),
        ("palette_size", ct.c_int32),
        ("palette_json", ct.POINTER(ct.c_char_p)),
    ]


class _Statistics(ct.Structure):
    _fields_ = [
        ("candidates", ct.c_int64),
        ("classes", ct.c_int64),
        ("canonical_nodes", ct.c_int64),
        ("complete", ct.c_int32),
        ("reason", ct.c_int32),
        ("elapsed_seconds", ct.c_double),
    ]


_CALLBACK = ct.CFUNCTYPE(
    ct.c_int,
    _INT,
    ct.c_int64,
    ct.c_int64,
    ct.POINTER(ct.c_ubyte),
    ct.c_int64,
    ct.c_char_p,
    ct.c_void_p,
)


@lru_cache(maxsize=8)
def _load(path):
    library = ct.CDLL(str(Path(path).resolve()))
    if library.synkit_mapper_classification_abi() != 1:
        raise RuntimeError("Unsupported native classification ABI")
    if hasattr(library, "synkit_mapper_classification_max_vertices"):
        library.synkit_mapper_classification_max_vertices.argtypes = []
        library.synkit_mapper_classification_max_vertices.restype = ct.c_int
    library.synkit_mapper_classification_error.restype = ct.c_char_p
    library.synkit_mapper_classify_batch.argtypes = [
        ct.POINTER(_Problem),
        ct.c_int64,
        _INT,
        _LONG,
        ct.c_double,
        ct.c_int64,
        ct.c_int64,
        _CALLBACK,
        ct.c_void_p,
        ct.POINTER(_Statistics),
    ]
    library.synkit_mapper_classify_batch.restype = ct.c_int
    library.synkit_mapper_classify_batch_with_cancel.argtypes = [
        ct.POINTER(_Problem),
        ct.c_int64,
        _INT,
        _LONG,
        ct.c_double,
        ct.c_int64,
        ct.c_int64,
        ct.c_void_p,
        _CALLBACK,
        ct.c_void_p,
        ct.POINTER(_Statistics),
    ]
    library.synkit_mapper_classify_batch_with_cancel.restype = ct.c_int
    if hasattr(library, "synkit_mapper_classify_batch_updates_with_cancel"):
        function = library.synkit_mapper_classify_batch_updates_with_cancel
        function.argtypes = library.synkit_mapper_classify_batch_with_cancel.argtypes
        function.restype = ct.c_int
    if hasattr(library, "synkit_mapper_classify_batch_ex"):
        function = library.synkit_mapper_classify_batch_ex
        function.argtypes = [
            *library.synkit_mapper_classify_batch_with_cancel.argtypes,
            ct.c_int32,
            ct.c_int32,
        ]
        function.restype = ct.c_int
    return library


@dataclass(frozen=True)
class NativeITSClass:
    representative: tuple[int, ...]
    cost2: int
    observed_mapping_count: int
    key: PackedCanonicalKey
    identifier: str


@dataclass(frozen=True)
class NativeClassificationResult:
    classes: tuple[NativeITSClass, ...]
    candidates: int
    complete: bool
    reason: str | None
    canonical_nodes: int
    elapsed_seconds: float


def _records_callback(n, classes, errors, class_callback, *, upserts=False):
    positions = {}

    @_CALLBACK
    def receive(mapping, cost2, observed, packed, size, identifier, _):
        try:
            name = identifier.decode("ascii")
            key = PackedCanonicalKey(ct.string_at(packed, size), identifier=name)
            record = NativeITSClass(
                tuple(mapping[i] for i in range(n)), cost2, observed, key, name
            )
            if upserts and key.parts in positions:
                index = positions[key.parts]
                old = classes[index]
                if (
                    old.representative != record.representative
                    or old.cost2 != record.cost2
                    or old.identifier != record.identifier
                    or record.observed_mapping_count < old.observed_mapping_count
                ):
                    raise RuntimeError("Invalid cumulative native class update")
                classes[index] = record
                return 0
            positions[key.parts] = len(classes)
            classes.append(record)
            return (
                int(bool(class_callback(record))) if class_callback is not None else 0
            )
        except BaseException as error:  # noqa: BLE001 - C callback boundary.
            errors.append(error)
            return 1

    return receive


def _classification_result(classes, statistics, seconds):
    return NativeClassificationResult(
        tuple(classes),
        statistics.candidates,
        bool(statistics.complete),
        {
            0: None,
            1: "time_limit",
            2: "canonical_node_limit",
            3: "class_limit",
            4: "callback_stop",
            5: "cancelled",
        }[statistics.reason],
        statistics.canonical_nodes,
        seconds,
    )


def _cap(value, name):
    if value is None:
        return -1
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 < value < 2**63
    ):
        raise ValueError(f"{name} must be positive int64 or None")
    return int(value)


class NativeITSBatchClassifier:
    """Exact attributed ITS classes with the existing SKC1 and class-ID contracts.

    ``properties`` preserves its insertion order and pairs endpoint atom arrays.
    ``edge_properties`` pairs dense endpoint attribute matrices. For E/Z, use
    ``{"stereo": (reactant_stereo, product_stereo)}``; colors then exactly match
    reaction_search's ``(weight_r, weight_p, stereo_r, stereo_p)``. Elements
    include isotope/type information under the caller's established convention.
    The C++ loop performs all per-candidate construction, exact search, hashing
    and deduplication. Python receives one callback per distinct complete key.
    Counts refer to supplied mappings, never inferred labeled multiplicities.
    Palette preparation memoizes repeated exact typed states within this
    constructor call; tokens, ordering and native certificate bytes are unchanged.
    Inputs above 256 vertices require the library's explicit capacity accessor;
    this implementation supports at most 512 vertices.
    """

    def __init__(
        self,
        reactant,
        product,
        elements,
        properties,
        *,
        library_path,
        edge_properties=None,
        identifier_mode="legacy",
    ):
        preparation_started = time.perf_counter()
        if identifier_mode not in {"legacy", "packed"}:
            raise ValueError("identifier_mode must be 'legacy' or 'packed'")
        self.identifier_mode = identifier_mode
        a, b = np.asarray(reactant), np.asarray(product)
        self.n = n = len(elements)
        if not 0 <= n <= 512 or a.shape != (n, n) or b.shape != (n, n):
            raise ValueError(
                "Native ITS requires square endpoints with 0..512 vertices"
            )
        # Old libraries retain their qualified 256-vertex limit. Only large
        # inputs load early; the established <=256 preparation order is intact.
        library = None
        if identifier_mode == "packed":
            library = _load(str(Path(library_path).resolve()))
            if not all(
                hasattr(library, name)
                for name in (
                    "synkit_mapper_classify_batch_ex",
                    "synkit_mapper_solve_classified_ex",
                )
            ):
                raise RuntimeError("Native library does not support packed identifiers")
        if n > 256:
            library = _load(str(Path(library_path).resolve()))
            capacity = min(
                512,
                library.synkit_mapper_classification_max_vertices()
                if hasattr(library, "synkit_mapper_classification_max_vertices")
                else 256,
            )
            if n > capacity:
                raise ValueError(
                    f"Native ITS library supports at most {capacity} vertices"
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
        # Chemistry repeats a small number of exact atom states across n*n
        # possible pairs. _typed preserves the established literal-type contract.
        left_states = [
            (
                _typed(elements[i]),
                tuple(_typed(sides[0][i]) for sides in properties.values()),
            )
            for i in range(n)
        ]
        right_states = [
            tuple(_typed(sides[1][j]) for sides in properties.values())
            for j in range(n)
        ]
        node_cache = {
            (left, right): _encode_color(
                (
                    left[0],
                    tuple(
                        (name, x, y) for name, x, y in zip(properties, left[1], right)
                    ),
                    (),
                )
            )
            for left in dict.fromkeys(left_states)
            for right in dict.fromkeys(right_states)
        }
        node_tokens = [
            [node_cache[left, right] for right in right_states] for left in left_states
        ]
        states, matrices = [], []
        for side, matrix in enumerate((a, b)):
            # Intern exact tokens rather than Python equality: bool/int and
            # signed floating zero may differ in the certificate contract.
            # Edge attributes use their exact encoder, not repr as a cache key:
            # distinct supported mappings/containers may have equal reprs.
            cache, encoded_states, encoded_rows = {}, {}, []
            for i in range(n):
                row = []
                for j in range(n):
                    weight = float(matrix[i, j])
                    attrs = tuple(
                        values[side][i][j] for values in edge_properties.values()
                    )
                    key = (weight.hex(), tuple(_encode_color(v) for v in attrs))
                    if key not in cache:
                        raw = (weight, *attrs)
                        token = _encode_color(raw)
                        cache[key] = token
                        encoded_states[token] = raw
                    row.append(cache[key])
                encoded_rows.append(row)
            palette = [encoded_states[token] for token in sorted(encoded_states)]
            if not palette:
                palette = [(0.0,)]
            ids = {_encode_color(value): index for index, value in enumerate(palette)}
            states.append(palette)
            matrices.append(
                np.ascontiguousarray(
                    [[ids[v] for v in row] for row in encoded_rows],
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
        self.library = (
            library if library is not None else _load(str(Path(library_path).resolve()))
        )
        self.preparation_seconds = time.perf_counter() - preparation_started

    @classmethod
    def from_endpoint_graphs(
        cls,
        endpoint_mols,
        endpoint_graphs,
        *,
        library_path,
        binary=False,
        identifier_mode="legacy",
    ):
        """Prepare the exact existing reaction_search class contract, including E/Z.

        Preparation still uses RDKit Python, as does the shared chemistry
        frontend. Canonical classification of every subsequent candidate is
        native. Objective anchors/dummies outside a molecule have empty bond
        stereo, matching the historical frontend exactly.
        """
        from ..slap.lap import _adjacency_and_elements

        a, elements = _adjacency_and_elements(endpoint_graphs[0], binary)
        b, _ = _adjacency_and_elements(endpoint_graphs[1], binary)
        properties = {
            name: (endpoint_graphs[0].props[name], endpoint_graphs[1].props[name])
            for name in ("charge", "radical", "hcount", "stereo", "dummy")
        }
        stereo = []
        for mol in endpoint_mols:
            matrix = np.full((len(elements), len(elements)), "", dtype=object)
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                matrix[i, j] = matrix[j, i] = str(bond.GetStereo())
            stereo.append(matrix)
        return cls(
            a,
            b,
            elements,
            properties,
            library_path=library_path,
            edge_properties={"stereo": stereo},
            identifier_mode=identifier_mode,
        )

    def classify(
        self,
        mappings,
        costs2=None,
        *,
        time_limit_seconds=None,
        max_canonical_nodes=None,
        max_classes=None,
        class_callback=None,
        cancel_token=None,
        stream=False,
    ):
        """Classify mappings, optionally delivering each key on discovery.

        ``stream=True`` uses the additive cumulative-count interface when
        available, otherwise the legacy batch API. User callbacks run once
        per new key; final result records merge subsequent count updates.
        Counts in interrupted streaming output are observed lower bounds.
        Legacy default callbacks retain their final batch-count semantics.
        """
        started = time.perf_counter()
        values = list(mappings)
        if any(len(mapping) != self.n for mapping in values):
            raise ValueError("Mapping width differs from ITS size")
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < self.n
            for mapping in values
            for v in mapping
        ):
            raise ValueError("Mappings require valid integer vertex IDs")
        permutations = np.ascontiguousarray(values, dtype=np.int32).reshape(
            len(values), self.n
        )
        costs2 = [0] * len(values) if costs2 is None else list(costs2)
        if len(costs2) != len(values) or any(
            isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 2**63
            for v in costs2
        ):
            raise ValueError("costs2 must contain one nonnegative int64 per mapping")
        costs = np.ascontiguousarray(costs2, dtype=np.int64)
        if time_limit_seconds is not None and (
            isinstance(time_limit_seconds, bool)
            or not isinstance(time_limit_seconds, Real)
            or not math.isfinite(time_limit_seconds)
            or time_limit_seconds < 0
        ):
            raise ValueError("time_limit_seconds must be finite and nonnegative")
        classes, errors = [], []

        packed_identifier = self.identifier_mode == "packed"
        updates = stream and (
            packed_identifier
            or hasattr(self.library, "synkit_mapper_classify_batch_updates_with_cancel")
        )
        receive = _records_callback(
            self.n, classes, errors, class_callback, upserts=updates
        )
        function = (
            self.library.synkit_mapper_classify_batch_ex
            if packed_identifier
            else (
                self.library.synkit_mapper_classify_batch_updates_with_cancel
                if updates
                else self.library.synkit_mapper_classify_batch_with_cancel
            )
        )

        statistics = _Statistics()
        status = function(
            ct.byref(self.problem),
            len(values),
            permutations.ctypes.data_as(_INT),
            costs.ctypes.data_as(_LONG),
            (
                -1
                if time_limit_seconds is None
                else max(0, time_limit_seconds - (time.perf_counter() - started))
            ),
            _cap(max_canonical_nodes, "max_canonical_nodes"),
            _cap(max_classes, "max_classes"),
            None if cancel_token is None else cancel_token.pointer,
            receive,
            None,
            ct.byref(statistics),
            *((int(updates), 1) if packed_identifier else ()),
        )
        if errors:
            raise errors[0]
        if status:
            raise RuntimeError(
                self.library.synkit_mapper_classification_error().decode()
            )
        return _classification_result(
            classes, statistics, time.perf_counter() - started
        )

    def solve_prepared(
        self,
        problem,
        request,
        *,
        max_canonical_nodes=None,
        class_callback=None,
        allowed_domains=None,
        stream=False,
    ):
        """Native search and classification using the shared prepared C structs.

        ``problem`` and ``request`` are ``_native.runtime.Problem/Request``.
        Their input buffers must stay alive for this synchronous call. This
        method copies the request and sets classify=1. The endpoint palette
        preparation is reusable and is reported separately by preparation_seconds.
        No intermediate mapping calls Python. Unsupported resource/backend
        settings are returned explicitly in the returned common Statistics.
        Return ``(NativeClassificationResult, runtime.Statistics)``.
        Optional ``allowed_domains`` is a dense n*n zero/one mask passed intact
        through the restricted native dispatcher. Its proof/completeness scope
        remains the supplied domains; class batching does not relax them.
        ``stream=True`` publishes new keys immediately and merges cumulative
        count updates; user callbacks run only on first discovery. Older
        libraries retain batch delivery. The default keeps legacy callbacks.
        """
        from .runtime import Problem, Request, Statistics

        started = time.perf_counter()
        if not isinstance(problem, Problem) or not isinstance(request, Request):
            raise TypeError("Expected shared native Problem and Request structs")
        prepared = Request.from_buffer_copy(request)
        prepared.classify = 1
        mask = None
        prefix = [ct.byref(problem), ct.byref(prepared)]
        prefix_types = [ct.POINTER(Problem), ct.POINTER(Request)]
        if allowed_domains is not None:
            values = np.asarray(allowed_domains)
            if (
                values.shape != (problem.n, problem.n)
                or not np.isin(values, [0, 1]).all()
            ):
                raise ValueError("allowed_domains must be an n-by-n zero/one mask")
            mask = np.ascontiguousarray(values, dtype=np.uint8)
            mask_pointer = ct.POINTER(ct.c_uint8)
            prefix.append(mask.ctypes.data_as(mask_pointer))
            prefix_types.append(mask_pointer)
        name = (
            "synkit_mapper_solve_classified"
            if mask is None
            else "synkit_mapper_solve_domains_classified"
        )
        packed_identifier = self.identifier_mode == "packed"
        updates = stream and (
            packed_identifier or hasattr(self.library, name + "_updates")
        )
        if packed_identifier:
            # The explicit interface always has a nullable mask argument.
            if mask is None:
                prefix.append(None)
                prefix_types.append(ct.POINTER(ct.c_uint8))
            function = self.library.synkit_mapper_solve_classified_ex
        else:
            function = getattr(self.library, name + "_updates" if updates else name)
        self.delivery_mode = "incremental_discovery" if updates else "legacy_batch"
        function.argtypes = [
            *prefix_types,
            ct.POINTER(_Problem),
            ct.c_int64,
            _CALLBACK,
            ct.c_void_p,
            ct.POINTER(Statistics),
            ct.POINTER(_Statistics),
            *((ct.c_int32, ct.c_int32) if packed_identifier else ()),
        ]
        function.restype = ct.c_int
        classes, errors = [], []
        receive = _records_callback(
            self.n, classes, errors, class_callback, upserts=updates
        )
        search_statistics, class_statistics = Statistics(), _Statistics()
        if prepared.seconds >= 0:
            prepared.seconds = max(
                0, prepared.seconds - (time.perf_counter() - started)
            )
        status = function(
            *prefix,
            ct.byref(self.problem),
            _cap(max_canonical_nodes, "max_canonical_nodes"),
            receive,
            None,
            ct.byref(search_statistics),
            ct.byref(class_statistics),
            *((int(updates), 1) if packed_identifier else ()),
        )
        if errors:
            raise errors[0]
        if status:
            core_error = getattr(self.library, "synkit_mapper_core_error", None)
            message = b""
            if core_error is not None:
                core_error.restype = ct.c_char_p
                message = core_error() or b""
            message = message or self.library.synkit_mapper_classification_error()
            raise RuntimeError((message or b"Native classified search failed").decode())
        return (
            _classification_result(
                classes, class_statistics, class_statistics.elapsed_seconds
            ),
            search_statistics,
        )
