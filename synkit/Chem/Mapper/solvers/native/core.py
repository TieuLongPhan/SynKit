"""Prepared-graph conversion and result records for the shared native core."""

import ctypes as ct
import json
import math
import time
from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from ..exact.distance_search.records import (
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    normalize_distance_target,
)
from ..slap.lap import _adjacency_and_elements
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
    resolve_library_path,
)


def _ids(values, registry):
    return np.asarray([registry.setdefault(value, len(registry)) for value in values], dtype=np.int32)


# Validation, native buffers and callbacks belong to the same solve lifetime.
def enumerate_core_mappings(  # noqa: C901
    lgp,
    *,
    library_path=None,
    backend="pabs",
    CD="minimal",
    CD_mode="exact",
    binary=False,
    symmetry=False,
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
    initial_mapping=None,
    allowed_domains=None,
    _native_classifier=None,
    _native_class_registry=None,
):
    started = time.perf_counter()
    library_path = resolve_library_path(library_path)
    target = normalize_distance_target(CD)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown native backend: {backend}")
    if allowed_domains is not None and backend not in {"kbb", "kmilp"}:
        raise ValueError("allowed_domains requires a kernel backend: kbb or kmilp")
    if CD_mode not in {"exact", "at_most"} or (CD_mode == "at_most" and target == "minimal"):
        raise ValueError("at_most requires numeric CD; otherwise use exact")
    for name, value in (
        ("binary", binary),
        ("symmetry", symmetry),
        ("collect_mappings", collect_mappings),
        ("proof_only", proof_only),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if proof_only and target != "minimal":
        raise ValueError("proof_only requires minimal CD")
    if initial_mapping is not None:
        raise ValueError("Shared-core explicit warm starts are not qualified yet")
    if compute_minimum_cost and target != "minimal":
        raise ValueError("Numeric core queries do not run a separate minimum proof")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable")
    for name, value in (
        ("max_states", max_states),
        ("max_mappings", max_mappings),
        ("max_classes", max_classes),
        ("max_frontier", max_frontier),
        ("memory_limit_bytes", memory_limit_bytes),
        ("max_bijections", max_bijections),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value < 2**63):
            raise ValueError(f"{name} must be a positive integer or None")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 0.25
    ):
        raise ValueError("tolerance must be finite and in [0,0.25)")
    if len(lgp) != 2:
        raise ValueError("Expected two endpoint graphs")
    matrices, elements = zip(*[_adjacency_and_elements(graph, binary) for graph in lgp])
    n = len(elements[0])
    if not 1 <= n <= 1024 or any(matrix.shape != (n, n) for matrix in matrices):
        raise ValueError("Core requires compatible 1..1024 vertex graphs")
    if Counter(elements[0]) != Counter(elements[1]):
        raise ValueError("Unequal atom-type inventories")
    encoded = []
    for graph, matrix in zip(lgp, matrices):
        if (
            any(graph.graph[i].get(i, 0) for i in range(n))
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
            or np.any(matrix > 500000)
            or not np.array_equal(matrix, matrix.T)
            or not np.allclose(matrix * 2, np.rint(matrix * 2), atol=tolerance, rtol=0)
        ):
            raise ValueError("Core requires symmetric loop-free nonnegative half-integer weights")
        encoded.append(np.ascontiguousarray(np.rint(matrix * 2), dtype=np.int64))
    total = math.prod(math.factorial(count) for count in Counter(elements[0]).values())
    if max_bijections is not None and total > max_bijections:
        raise ExactEnumerationLimitError("Bijections exceed max_bijections")
    registry = {}
    types = [_ids(side, registry) for side in elements]
    colors = {}
    nodes = []
    for graph, side in zip(lgp, elements):
        if set(node_properties) - graph.props.keys():
            raise ValueError("Missing symmetry node properties")
        nodes.append(
            _ids(
                [(element, *(graph.props[name][i] for name in node_properties)) for i, element in enumerate(side)],
                colors,
            )
        )
    edges = [np.zeros((n, n), dtype=np.int32) for _ in lgp]
    # Extra native edge class tokens may be supplied by a frontend; absent
    # matrices mean no extra edge attributes beyond the weighted objective.
    for i, graph in enumerate(lgp):
        if "_native_edge_colors" in graph.props:
            values = np.asarray(graph.props["_native_edge_colors"])
            if values.shape != (n, n) or not np.array_equal(values, values.T):
                raise ValueError("Invalid native edge class attributes")
            edges[i] = np.ascontiguousarray(values, dtype=np.int32)
    maximum = float((matrices[0].sum() + matrices[1].sum()) / 2)
    query = 3 if proof_only else 0 if target == "minimal" else 2 if CD_mode == "at_most" else 1
    impossible = False
    if target == "minimal":
        target2 = 0
    elif query == 2:
        target2 = math.floor((min(target, maximum) + tolerance) * 2)
    elif target > maximum + tolerance:
        target2 = 0
        impossible = True
    else:
        target2 = round(target * 2)
        impossible = abs(target - target2 / 2) > tolerance
    problem = Problem(
        1,
        n,
        *[array.ctypes.data_as(LONG) for array in encoded],
        *[array.ctypes.data_as(INT) for array in [*types, *nodes, *edges]],
    )
    domain_mask = None
    if allowed_domains is not None:
        values = np.asarray(allowed_domains)
        if values.shape != (n, n) or not np.isin(values, [0, 1]).all():
            raise ValueError("allowed_domains must be an n-by-n zero/one mask")
        domain_mask = np.ascontiguousarray(values, dtype=np.uint8)
    request = Request()
    request.abi_version = 1
    request.backend = BACKENDS[backend]
    request.query = query
    request.symmetry = int(symmetry)
    request.workers = 1
    request.target2 = target2
    request.seconds = -1 if time_limit_seconds is None else max(0, time_limit_seconds - (time.perf_counter() - started))
    for name, value in (
        ("max_states", max_states),
        ("max_candidates", max_mappings),
        ("max_classes", max_classes),
        ("max_frontier", max_frontier),
        ("memory_limit_bytes", memory_limit_bytes),
    ):
        setattr(request, name, -1 if value is None else int(value))
    request.slap_max_fingerprints = 10000
    stop_flags = np.asarray([int(value == 1) for value in lgp[0].labels], dtype=np.int32)
    request.slap_stop_flags = stop_flags.ctypes.data_as(INT)
    request.cancel_token = None if cancel_token is None else cancel_token.pointer
    mappings = []
    distances = []
    observed = set()
    errors = []
    classification = None

    def receive(pointer, cost2, _userdata):
        try:
            mapping = list(pointer[:n])
            cost = cost2 / 2
            if sorted(mapping) != list(range(n)) or any(
                elements[0][i] != elements[1][j] for i, j in enumerate(mapping)
            ):
                raise RuntimeError("Native core returned incompatible bijection")
            observed.add(cost)
            if collect_mappings:
                mappings.append(mapping)
                distances.append(cost)
            if mapping_callback is not None:
                mapping_callback(mapping, cost)
            return 0
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
            return 1

    callback = CALLBACK(receive)
    stats = Statistics()
    library = load_core(str(Path(library_path).resolve()))
    request.seconds = -1 if time_limit_seconds is None else max(0, time_limit_seconds - (time.perf_counter() - started))
    if impossible:
        stats.search_complete = stats.classification_complete = stats.hydrogen_complete = 1
        stats.minimum2 = stats.incumbent2 = stats.lower_bound2 = -1
        stats.labeled_count = 0
    elif _native_classifier is not None:
        if _native_class_registry is None:
            raise ValueError("Native class registry is required")

        def receive_class(record):
            _native_class_registry[record.representative] = record
            array = (ct.c_int32 * n)(*record.representative)
            return receive(array, record.cost2, None)

        classification, stats = _native_classifier.solve_prepared(
            problem,
            request,
            class_callback=receive_class,
            max_canonical_nodes=getattr(_native_classifier, "frontend_max_canonical_nodes", None),
            allowed_domains=domain_mask,
            stream=True,
        )
        if errors:
            raise errors[0]
    else:
        if domain_mask is not None:
            if not hasattr(library, "synkit_mapper_solve_domains"):
                raise RuntimeError("Library lacks the native kernel domain API")
            result = library.synkit_mapper_solve_domains(
                ct.byref(problem),
                ct.byref(request),
                domain_mask.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                callback,
                None,
                ct.byref(stats),
            )
        else:
            result = library.synkit_mapper_solve(ct.byref(problem), ct.byref(request), callback, None, ct.byref(stats))
        if errors:
            raise errors[0]
        if result:
            raise RuntimeError(library.synkit_mapper_core_error().decode())
    detail = {name: getattr(stats, name) for name, _ in stats._fields_}
    detail.update(
        core_abi=1,
        backend=backend,
        library_path=str(Path(library_path).resolve()),
        CD_mode=CD_mode,
        proof_only=proof_only,
        cost_scale=2,
        requested_symmetry=symmetry,
        domain_restricted=domain_mask is not None,
        enumeration_search_seconds=max(0, stats.enumeration_seconds - stats.callback_seconds),
    )
    metadata_function = getattr(library, "synkit_mapper_backend_metadata", None)
    backend_executed = (
        not impossible
        and stats.status != 2
        and not (stats.reason == 10 and stats.nodes == 0 and stats.model_seconds == 0 and stats.candidate_count == 0)
    )
    detail["backend_executed"] = backend_executed
    if metadata_function and backend_executed:
        detail["native_metadata"] = json.loads(metadata_function(BACKENDS[backend]).decode())
    version_function = getattr(library, "synkit_mapper_backend_version", None)
    detail.update(
        solver_version=version_function(BACKENDS[backend]).decode() if version_function else None,
        threads=1,
        classification_reason=None if classification is None else classification.reason,
        classification_delivery=(None if classification is None else _native_classifier.delivery_mode),
        delivered_class_count=(None if classification is None else len(classification.classes)),
    )
    if backend == "scip":
        detail.update(
            numerical_qualification=(
                "solver OPTIMAL under floating feasibility tolerances; " "no checked formal certificate"
            ),
            relative_gap=0,
            absolute_gap=0,
            feasibility_tolerance=1e-9,
        )
    if backend in {"cp_sat", "scip", "maxsat"}:
        detail["implementation"] = "cpp_sparse_full_assignment_linearization"
    complete = bool(stats.search_complete) and not proof_only and backend != "slap"
    reason = REASONS.get(stats.reason, "unknown_stop")
    return DistanceEnumerationResult(
        target=target,
        cost=next(iter(observed)) if len(observed) == 1 else None,
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
        selected_labeled_mapping_count=None if stats.labeled_count < 0 else stats.labeled_count,
        scope=(
            "restricted_kernel_assignment_space"
            if backend in {"kbb", "kmilp"}
            else (
                "heuristic_candidates"
                if backend == "slap"
                else (
                    "atom_compatible_assignment_space_modulo_verified_symmetry"
                    if symmetry or backend == "cpabs"
                    else "complete_atom_compatible_assignment_space"
                )
            )
        ),
        symmetry_group_order=None if symmetry or backend == "cpabs" else 1,
        symmetry_search_complete=not symmetry,
        symmetry_quotient_complete=False,
        backend=f"{backend}_cpp",
        backend_statistics=detail,
    )


def map_reaction_core(reaction, *, library_path, backend="pabs", native_classification=None, **options):
    from ...reaction_search import _map_reaction_with_search

    domains = options.pop("allowed_domains", None)
    if native_classification is None:
        native_classification = True

    def search(graphs, **kwargs):
        kwargs.pop("backend")
        kwargs["symmetry"] = backend in {"pabs", "cpabs"}
        return enumerate_core_mappings(graphs, backend=backend, allowed_domains=domains, **kwargs)

    return _map_reaction_with_search(
        reaction,
        search_function=search,
        backend="cpp",
        library_path=library_path,
        native_classification_library=library_path if native_classification else None,
        **options,
    )
