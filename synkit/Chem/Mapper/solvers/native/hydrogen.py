"""Native terminal-H reduction, exact all-shell transports and full native fallback."""

from __future__ import annotations

import ctypes as ct
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from ..exact.distance_search.records import (
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    normalize_distance_target,
)
from ..exact.hydrogen import HydrogenTransferPlan
from ..slap.lap import _adjacency_and_elements
from .classification import (
    _CALLBACK as CLASS_CALLBACK,
)
from .classification import (
    _Problem as ClassProblem,
)
from .classification import (
    _records_callback,
)
from .classification import (
    _Statistics as ClassStatistics,
)
from .runtime import (
    BACKENDS,
    CALLBACK,
    INT,
    LONG,
    REASONS,
    Problem,
    Request,
    Statistics,
    load_core,
)


class Attributes(ct.Structure):
    _fields_ = [
        (name + "_" + side, INT)
        for name in (
            "atomic_numbers",
            "isotopes",
            "charges",
            "radicals",
            "dummies",
            "stereo",
        )
        for side in ("r", "p")
    ]


class HydrogenStatistics(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in ("supported", "eligibility_reason", "transport_complete", "reason")]
    _fields_ += [
        (name, ct.c_int64)
        for name in (
            "full_nodes",
            "quotient_nodes",
            "heavy_candidates",
            "transport_plans",
            "transport_nodes",
        )
    ]
    _fields_ += [(name, ct.c_double) for name in ("reduction_seconds", "heavy_seconds", "transport_seconds")]


class TransportStatistics(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("minimum2", "plans", "nodes")]
    _fields_ += [
        ("complete", ct.c_int32),
        ("reason", ct.c_int32),
        ("elapsed_seconds", ct.c_double),
    ]


FLOW_CALLBACK = ct.CFUNCTYPE(ct.c_int, INT, ct.c_int64, ct.c_char_p, ct.c_void_p)
ELIGIBILITY = {
    0: None,
    1: "no_hydrogen_reduction",
    2: "isotopic_hydrogen",
    3: "annotated_stereochemistry",
    4: "bonded_hydrogen_placeholder",
    5: "nonneutral_or_radical_hydrogen",
    6: "nonterminal_hydrogen",
    7: "nonordinary_hydrogen_bond",
    8: "hydrogen_class_attributes",
    9: "quotient_backend_unsupported",
}


@lru_cache(maxsize=8)
def _load(path):
    library = load_core(str(Path(path).resolve()))
    if library.synkit_mapper_hydrogen_abi() != 1:
        raise RuntimeError("Unsupported native hydrogen ABI")
    library.synkit_mapper_hydrogen_error.restype = ct.c_char_p
    library.synkit_mapper_hydrogen_transport.argtypes = [
        ct.c_int,
        INT,
        INT,
        ct.c_int,
        ct.c_int,
        ct.c_int64,
        ct.c_double,
        ct.c_int64,
        ct.c_int64,
        ct.c_void_p,
        FLOW_CALLBACK,
        ct.c_void_p,
        ct.POINTER(TransportStatistics),
    ]
    library.synkit_mapper_hydrogen_transport.restype = ct.c_int
    library.synkit_mapper_hydrogen_solve.argtypes = [
        ct.POINTER(Problem),
        ct.POINTER(Attributes),
        ct.POINTER(Request),
        ct.c_int64,
        CALLBACK,
        ct.c_void_p,
        ct.POINTER(Statistics),
        ct.POINTER(HydrogenStatistics),
    ]
    library.synkit_mapper_hydrogen_solve.restype = ct.c_int
    library.synkit_mapper_hydrogen_solve_classified.argtypes = [
        ct.POINTER(Problem),
        ct.POINTER(Attributes),
        ct.POINTER(Request),
        ct.POINTER(ClassProblem),
        ct.c_int64,
        ct.c_int64,
        CLASS_CALLBACK,
        ct.c_void_p,
        ct.POINTER(Statistics),
        ct.POINTER(HydrogenStatistics),
        ct.POINTER(ClassStatistics),
    ]
    library.synkit_mapper_hydrogen_solve_classified.restype = ct.c_int
    if hasattr(library, "synkit_mapper_hydrogen_solve_classified_updates"):
        function = library.synkit_mapper_hydrogen_solve_classified_updates
        function.argtypes = library.synkit_mapper_hydrogen_solve_classified.argtypes
        function.restype = ct.c_int
    return library


def _cap(value, name):
    if value is None:
        return -1
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 < value < 2**63:
        raise ValueError(f"{name} must be positive int64 or None")
    return int(value)


def _decimal_integer(payload):
    # Convert exact native counts without changing Python's process-wide
    # decimal-string security limit. Each parsed chunk is at most nine digits.
    value = 0
    for start in range(0, len(payload), 9):
        chunk = payload[start : start + 9]
        value = value * 10 ** len(chunk) + int(chunk)
    return value


@dataclass(frozen=True)
class NativeHydrogenEnumerationResult:
    plans: tuple[HydrogenTransferPlan, ...]
    minimum_distance: int
    labeled_mapping_count: int
    complete: bool
    observed_plan_count: int
    truncation_reason: str | None
    backend_statistics: dict


def enumerate_hydrogen_transfers(
    rows,
    columns,
    *,
    library_path,
    reservoir=None,
    CD="minimal",
    CD_mode="exact",
    max_plans=None,
    max_nodes=None,
    collect_plans=True,
    time_limit_seconds=None,
    cancel_token=None,
    plan_callback=None,
):
    """Enumerate every parent-flow orbit in a minimum, exact or at-most shell.

    For a real-parent transfer the cost is two bond edits; entering/leaving
    the isolated dummy reservoir costs one. Multiplicities use exact native
    integer arithmetic and are converted from decimal strings here.
    """
    started = time.perf_counter()
    rows, columns = tuple(rows), tuple(columns)
    if (
        len(rows) != len(columns)
        or not rows
        or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 2**31 for v in rows + columns)
    ):
        raise ValueError("H counts must be equally sized nonnegative integer arrays")
    if reservoir is not None and (
        isinstance(reservoir, bool) or not isinstance(reservoir, Integral) or not 0 <= reservoir < len(rows)
    ):
        raise ValueError("reservoir must be a valid parent index or None")
    target = normalize_distance_target(CD)
    if CD_mode not in {"exact", "at_most"} or (target == "minimal" and CD_mode != "exact"):
        raise ValueError("at_most requires numeric CD")
    if not isinstance(collect_plans, bool):
        raise TypeError("collect_plans must be boolean")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    query = 0 if target == "minimal" else 2 if CD_mode == "at_most" else 1
    # Every H transport has integral CD. An odd target2 denotes an empty shell.
    target2 = (
        0
        if target == "minimal"
        else (math.floor(target * 2) if query == 2 else int(target * 2) if target.is_integer() else 1)
    )
    r, p = (np.ascontiguousarray(v, dtype=np.int32) for v in (rows, columns))
    plans, errors = [], []
    observed_count = 0
    labeled_total = 0

    @FLOW_CALLBACK
    def receive(cells, cost2, multiplicity, _):
        nonlocal observed_count, labeled_total
        try:
            size = len(rows)
            plan = HydrogenTransferPlan(
                tuple(
                    (i, j, cells[i * size + j])
                    for i in range(size)
                    for j in range(size)
                    if i != j and cells[i * size + j]
                ),
                tuple(cells[i * size + i] for i in range(size)),
                cost2 // 2,
                _decimal_integer(multiplicity),
            )
            observed_count += 1
            labeled_total += plan.labeled_mapping_count
            if collect_plans:
                plans.append(plan)
            return int(bool(plan_callback(plan))) if plan_callback is not None else 0
        except BaseException as error:  # noqa: BLE001 - C callback boundary.
            errors.append(error)
            return 1

    library = _load(str(Path(library_path).resolve()))
    statistics = TransportStatistics()
    status = library.synkit_mapper_hydrogen_transport(
        len(rows),
        r.ctypes.data_as(INT),
        p.ctypes.data_as(INT),
        -1 if reservoir is None else reservoir,
        query,
        target2,
        (-1 if time_limit_seconds is None else max(0, time_limit_seconds - (time.perf_counter() - started))),
        _cap(max_plans, "max_plans"),
        _cap(max_nodes, "max_nodes"),
        None if cancel_token is None else cancel_token.pointer,
        receive,
        None,
        ct.byref(statistics),
    )
    if errors:
        raise errors[0]
    if status:
        raise RuntimeError(library.synkit_mapper_hydrogen_error().decode())
    return NativeHydrogenEnumerationResult(
        tuple(plans),
        statistics.minimum2 // 2,
        labeled_total,
        bool(statistics.complete),
        observed_count,
        "plan_limit" if statistics.reason == 3 else REASONS.get(statistics.reason),
        {name: getattr(statistics, name) for name, _ in TransportStatistics._fields_},
    )


def enumerate_hydrogen_mappings(
    lgp,
    *,
    library_path,
    backend="pabs",
    CD="minimal",
    CD_mode="exact",
    binary=False,
    symmetry=True,
    node_properties=(),
    time_limit_seconds=None,
    max_states=None,
    max_mappings=None,
    max_classes=None,
    max_frontier=None,
    memory_limit_bytes=None,
    collect_mappings=True,
    mapping_callback=None,
    compute_minimum_cost=None,
    max_bijections=None,
    tolerance=1e-9,
    proof_only=False,
    cancel_token=None,
    max_transport_plans=None,
    max_canonical_nodes=None,
    fallback=True,
    _native_classifier=None,
    _native_class_registry=None,
    _stereo_annotated=False,
):
    """Search explicit-H classes natively using a checked terminal-H quotient.

    Numeric queries enumerate all bounded transports after a heavy minimum-H
    lower-envelope search. Unsupported chemistry explicitly invokes the full
    requested native backend on the SAME remaining wall budget, and records
    why reduction was declined. ``fallback=False`` returns unsupported instead.
    """
    from .core import _ids, enumerate_core_mappings

    started = time.perf_counter()
    target = normalize_distance_target(CD)
    if backend not in BACKENDS or backend == "slap":
        raise ValueError("Hydrogen quotient requires an exact native heavy backend")
    if CD_mode not in {"exact", "at_most"} or (CD_mode == "at_most" and target == "minimal"):
        raise ValueError("at_most requires numeric CD")
    if proof_only and target != "minimal":
        raise ValueError("proof_only requires minimum CD")
    if compute_minimum_cost and target != "minimal":
        raise ValueError("Numeric H queries do not run a separate minimum proof")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable")
    for name, value in (
        ("binary", binary),
        ("symmetry", symmetry),
        ("collect_mappings", collect_mappings),
        ("fallback", fallback),
        ("proof_only", proof_only),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    if (
        not isinstance(tolerance, Real)
        or isinstance(tolerance, bool)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 0.25
    ):
        raise ValueError("tolerance must be finite and in [0,0.25)")
    if len(lgp) != 2:
        raise ValueError("Expected two endpoint graphs")
    matrices, elements = zip(*(_adjacency_and_elements(graph, binary) for graph in lgp))
    n = len(elements[0])
    if (
        not 1 <= n <= 1024
        or any(matrix.shape != (n, n) for matrix in matrices)
        or Counter(elements[0]) != Counter(elements[1])
    ):
        raise ValueError("Hydrogen endpoints require equal compatible inventories")
    keepalive = []
    for matrix in matrices:
        if (
            not np.isfinite(matrix).all()
            or np.any(matrix < 0)
            or np.any(matrix > 500000)
            or np.any(np.diag(matrix))
            or not np.array_equal(matrix, matrix.T)
            or not np.allclose(matrix * 2, np.rint(matrix * 2), atol=tolerance, rtol=0)
        ):
            raise ValueError("Native H requires nonnegative half-integer symmetric loop-free weights")
        keepalive.append(np.ascontiguousarray(np.rint(matrix * 2), dtype=np.int64))
    registry = {}
    types = [_ids(side, registry) for side in elements]
    registry = {}
    nodes = []
    for graph, side in zip(lgp, elements):
        if set(node_properties) - graph.props.keys():
            raise ValueError("Missing symmetry node properties")
        nodes.append(
            _ids(
                [(v, *(graph.props[name][i] for name in node_properties)) for i, v in enumerate(side)],
                registry,
            )
        )
    edges = [
        np.ascontiguousarray(graph.props.get("_native_edge_colors", np.zeros((n, n))), dtype=np.int32) for graph in lgp
    ]
    if any(edge.shape != (n, n) for edge in edges):
        raise ValueError("Invalid native edge colors")
    keepalive.extend(types + nodes + edges)
    problem = Problem(
        1,
        n,
        *(a.ctypes.data_as(LONG) for a in keepalive[:2]),
        *(a.ctypes.data_as(INT) for a in types + nodes + edges),
    )
    metadata = []
    for name in (
        "atomic_numbers",
        "isotopes",
        "charges",
        "radicals",
        "dummies",
        "stereo",
    ):
        for graph, labels in zip(lgp, elements):
            if name in {"atomic_numbers", "isotopes"}:
                index = 0 if name == "atomic_numbers" else 1
                values = [
                    (v[index] if isinstance(v, tuple) and len(v) == 2 else v if index == 0 else 0) for v in labels
                ]
            elif name == "stereo":
                values = [
                    bool(a) or bool(b)
                    for a, b in zip(
                        graph.props.get("stereo", [0] * n),
                        graph.props.get("_stereo_anchor", [0] * n),
                    )
                ]
                if _stereo_annotated:
                    values[0] = True
            else:
                values = graph.props.get(
                    {"charges": "charge", "radicals": "radical", "dummies": "dummy"}[name],
                    [0] * n,
                )
            metadata.append(np.ascontiguousarray(values, dtype=np.int32))
    attrs = Attributes(*(v.ctypes.data_as(INT) for v in metadata))
    maximum = float((matrices[0].sum() + matrices[1].sum()) / 2)
    query = 3 if proof_only else 0 if target == "minimal" else 2 if CD_mode == "at_most" else 1
    impossible = False
    if target == "minimal":
        target2 = 0
    elif query == 2:
        target2 = math.floor((min(target, maximum) + tolerance) * 2)
    else:
        target2 = round(target * 2)
        impossible = target > maximum + tolerance or abs(target - target2 / 2) > tolerance
    request = Request()
    (
        request.abi_version,
        request.backend,
        request.query,
        request.workers,
        request.symmetry,
    ) = (1, BACKENDS[backend], query, 1, int(symmetry))
    request.target2 = target2
    request.seconds = -1 if time_limit_seconds is None else max(0, time_limit_seconds - (time.perf_counter() - started))
    for name, value in (
        ("max_states", max_states),
        ("max_candidates", max_mappings),
        ("max_classes", max_classes),
        ("max_frontier", max_frontier),
        ("memory_limit_bytes", memory_limit_bytes),
    ):
        setattr(request, name, _cap(value, name))
    request.cancel_token = None if cancel_token is None else cancel_token.pointer
    total = math.prod(math.factorial(count) for count in Counter(elements[0]).values())
    if max_bijections is not None and total > _cap(max_bijections, "max_bijections"):
        raise ExactEnumerationLimitError("Full explicit-H bijections exceed max_bijections")
    library = _load(str(Path(library_path).resolve()))
    stats, hstats, cstats = Statistics(), HydrogenStatistics(), ClassStatistics()
    mappings, distances, errors = [], [], []

    def receive_map(mapping, cost):
        if collect_mappings:
            mappings.append(list(mapping))
            distances.append(cost)
        if mapping_callback is not None:
            return mapping_callback(list(mapping), cost)
        return False

    @CALLBACK
    def receive(pointer, cost2, _):
        try:
            return int(bool(receive_map(pointer[:n], cost2 / 2)))
        except BaseException as error:  # noqa: BLE001 - C callback boundary.
            errors.append(error)
            return 1

    def receive_class(record):
        _native_class_registry[record.representative] = record
        return receive_map(record.representative, record.cost2 / 2)

    request.seconds = -1 if time_limit_seconds is None else max(0, time_limit_seconds - (time.perf_counter() - started))

    if impossible:
        stats.search_complete = stats.classification_complete = stats.hydrogen_complete = 1
        stats.minimum2 = -1
    elif _native_classifier is not None and not proof_only:
        if _native_class_registry is None:
            raise ValueError("Native class registry required")
        request.classify = 1
        updates = hasattr(library, "synkit_mapper_hydrogen_solve_classified_updates")
        delivered_classes = []
        class_callback = _records_callback(n, delivered_classes, errors, receive_class, upserts=updates)
        function = (
            library.synkit_mapper_hydrogen_solve_classified_updates
            if updates
            else library.synkit_mapper_hydrogen_solve_classified
        )
        status = function(
            ct.byref(problem),
            ct.byref(attrs),
            ct.byref(request),
            ct.byref(_native_classifier.problem),
            _cap(max_transport_plans, "max_transport_plans"),
            _cap(max_canonical_nodes, "max_canonical_nodes"),
            class_callback,
            None,
            ct.byref(stats),
            ct.byref(hstats),
            ct.byref(cstats),
        )
        if errors:
            raise errors[0]
        if status:
            raise RuntimeError(library.synkit_mapper_hydrogen_error().decode())
    else:
        status = library.synkit_mapper_hydrogen_solve(
            ct.byref(problem),
            ct.byref(attrs),
            ct.byref(request),
            _cap(max_transport_plans, "max_transport_plans"),
            receive,
            None,
            ct.byref(stats),
            ct.byref(hstats),
        )
        if errors:
            raise errors[0]
        if status:
            raise RuntimeError(library.synkit_mapper_hydrogen_error().decode())
    if stats.status == 2 and hstats.eligibility_reason and fallback:
        result = enumerate_core_mappings(
            lgp,
            library_path=library_path,
            backend=backend,
            CD=CD,
            CD_mode=CD_mode,
            binary=binary,
            symmetry=symmetry and hstats.eligibility_reason != 8,
            node_properties=node_properties,
            time_limit_seconds=(
                None if time_limit_seconds is None else max(0, time_limit_seconds - (time.perf_counter() - started))
            ),
            max_states=max_states,
            max_mappings=max_mappings,
            max_classes=max_classes,
            max_frontier=max_frontier,
            memory_limit_bytes=memory_limit_bytes,
            collect_mappings=collect_mappings,
            mapping_callback=mapping_callback,
            compute_minimum_cost=compute_minimum_cost,
            max_bijections=max_bijections,
            tolerance=tolerance,
            proof_only=proof_only,
            cancel_token=cancel_token,
            _native_classifier=_native_classifier,
            _native_class_registry=_native_class_registry,
        )
        details = dict(result.backend_statistics or {})
        details.update(
            hydrogen_quotient=False,
            hydrogen_quotient_fallback=ELIGIBILITY[hstats.eligibility_reason],
            hydrogen_reduction_seconds=hstats.reduction_seconds,
            hydrogen_fallback_symmetry_disabled=hstats.eligibility_reason == 8,
        )
        return replace(
            result,
            backend_statistics=details,
            elapsed_seconds=time.perf_counter() - started,
        )
    details = {name: getattr(stats, name) for name, _ in Statistics._fields_}
    details.update(
        hydrogen_quotient=bool(hstats.supported),
        hydrogen_quotient_fallback=ELIGIBILITY[hstats.eligibility_reason],
        hydrogen_statistics={name: getattr(hstats, name) for name, _ in HydrogenStatistics._fields_},
        hydrogen_complete=bool(stats.hydrogen_complete),
        core_abi=1,
        backend=backend,
        solver_version=(
            library.synkit_mapper_backend_version(BACKENDS[backend]).decode()
            if hasattr(library, "synkit_mapper_backend_version")
            else "unknown"
        ),
        library_path=str(Path(library_path).resolve()),
        threads=1,
        CD_mode=CD_mode,
        proof_only=proof_only,
        requested_symmetry=symmetry,
        cost_scale=2,
        enumeration_search_seconds=max(0, stats.enumeration_seconds - stats.callback_seconds),
        mapping_count_scope="lifted_hydrogen_transport_orbits",
        classification_reason={
            0: None,
            1: "time_limit",
            2: "canonical_node_limit",
            3: "class_limit",
            4: "callback_stop",
            5: "cancelled",
        }[cstats.reason],
        classification_delivery=(
            "incremental_discovery"
            if _native_classifier is not None and hasattr(library, "synkit_mapper_hydrogen_solve_classified_updates")
            else "legacy_batch" if _native_classifier is not None else None
        ),
        delivered_class_count=(
            len(delivered_classes) if _native_classifier is not None and not impossible and not proof_only else None
        ),
    )
    if backend == "scip":
        details.update(
            numerical_qualification=(
                "solver OPTIMAL under floating feasibility tolerances; " "no checked formal certificate"
            ),
            relative_gap=0,
            absolute_gap=0,
            feasibility_tolerance=1e-9,
        )
    complete = bool(stats.search_complete) and not proof_only
    reason = "hydrogen_plan_limit" if hstats.reason == 11 else REASONS.get(stats.reason)
    return DistanceEnumerationResult(
        target=target,
        cost=next(iter(set(distances))) if len(set(distances)) == 1 else None,
        minimum_cost=stats.minimum2 / 2 if stats.minimum_proven else None,
        mappings=mappings,
        distances=distances,
        total_bijections=total,
        maximum_cost_upper_bound=maximum,
        visited_leaves=stats.leaves,
        pruned_branches=0,
        visited_nodes=stats.nodes,
        elapsed_seconds=time.perf_counter() - started,
        status=(
            ("complete" if stats.candidate_count else "no_solutions")
            if complete
            else "unsupported" if stats.status == 2 else "timeout"
        ),
        complete=complete,
        truncation_reason=reason or ("proof_only" if proof_only else None),
        selected_mapping_count=stats.candidate_count,
        selected_labeled_mapping_count=None,
        scope="full_weighted_distance_via_terminal_hydrogen_quotient",
        symmetry_group_order=None,
        symmetry_search_complete=False,
        symmetry_quotient_complete=False,
        backend=f"hydrogen_{backend}_cpp",
        backend_statistics=details,
    )


def map_reaction_hydrogen(
    reaction,
    *,
    library_path,
    backend="pabs",
    max_transport_plans=None,
    max_canonical_nodes=None,
    native_classification=True,
    **options,
):
    """Map all explicit-H classes with native reduction, flows and classification."""
    from ...reaction_search import _map_reaction_with_search

    def search(graphs, **kwargs):
        kwargs.pop("backend")
        return enumerate_hydrogen_mappings(
            graphs,
            library_path=kwargs.pop("library_path"),
            backend=backend,
            max_transport_plans=max_transport_plans,
            max_canonical_nodes=max_canonical_nodes,
            _stereo_annotated=isinstance(reaction, str) and ("@" in reaction or "|" in reaction),
            **kwargs,
        )

    options.setdefault("objective", "lewis")
    options.setdefault("hydrogen", "explicit")
    return _map_reaction_with_search(
        reaction,
        search_function=search,
        backend="cpp",
        library_path=library_path,
        native_classification_library=library_path if native_classification else None,
        **options,
    )
