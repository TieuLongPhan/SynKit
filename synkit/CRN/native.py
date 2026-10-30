"""Optional native exact semiflow enumeration and siphon/trap closure.

Importing never invokes a compiler. Native builds are explicit and immutable.
"""

from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
from time import monotonic
import numpy as np
from .analysis import AnalysisResult
from .Petrinet.net import PetriNet


from ._build import boost_include
from . import _native_build

# Retain the private discovery alias used by existing source-checkout tools.
_boost_include = boost_include
build_native = _native_build.build_native


def bundled_native_library():
    """Return the installed wheel's CRN library without compiling anything."""
    from synkit._native import bundled_library

    try:
        return bundled_library(Path(__file__).parent, "synkit_crn")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No bundled CRN library is installed. Install a SynKit platform wheel "
            "or supply a library created by build_native(..., with_semiflows=True)."
        ) from exc


def _library_manifest(library_file):
    library_file = Path(library_file).resolve()
    manifest = json.loads(library_file.with_suffix(".json").read_text())
    if manifest.get("integrity") == "wheel-record":
        if library_file != bundled_native_library():
            raise RuntimeError(
                "Wheel-record verification requires the installed bundled CRN library"
            )
        from synkit._native import verify_wheel_library

        actual = verify_wheel_library(library_file)
        return {**manifest, "library_sha256": actual}
    else:
        expected = manifest["library_sha256"]
    actual = hashlib.sha256(library_file.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("CRN native library checksum mismatch")
    return {**manifest, "library_sha256": actual}


def _load_library(library_path):
    if library_path is None:
        from .backend import resolve_backend

        _, library_path = resolve_backend("cpp")
    library_file = Path(library_path).resolve()
    _library_manifest(library_file)
    library = ctypes.CDLL(str(library_file))
    if library.synkit_crn_abi() != 1:
        raise RuntimeError("Unsupported CRN native ABI")
    return library


def enumerate_semiflows(
    stoich,
    *,
    kind="p",
    library_path=None,
    time_limit=30.0,
    max_candidates=1000000,
    max_rows=100000,
    max_results=10000,
    max_coefficient_bits=4096,
):
    """Enumerate exact support-minimal P/T-semiflows entirely in C++.

    Output integers have arbitrary precision. On a limit, no intermediate rows
    are returned as alleged semiflows. Native witnesses are checked in Python
    against the original integer matrix. Time includes preparation/verification
    in reported seconds; the deadline applies to native search.
    """
    from math import gcd
    from functools import reduce
    from .Petrinet.minimal_semiflows import _integer_stoichiometry

    if kind not in {"p", "t"}:
        raise ValueError("kind must be 'p' or 't'")
    if not np.isfinite(time_limit) or not 0 < time_limit <= 1e9:
        raise ValueError("time_limit must be finite and between zero and 1e9 seconds")
    limits = (max_candidates, max_rows, max_results, max_coefficient_bits)
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 or n >= 2**63
        for n in limits
    ):
        raise ValueError("Native search limits must be positive integers below 2**63")
    started = monotonic()
    s = _integer_stoichiometry(stoich)
    matrix = np.ascontiguousarray(s if kind == "p" else s.T, dtype=np.int64)
    library = _load_library(library_path)
    if (
        not hasattr(library, "synkit_crn_semiflows_abi")
        or library.synkit_crn_semiflows_abi() != 1
    ):
        raise RuntimeError(
            "This library has no exact semiflow backend; rebuild with with_semiflows=True"
        )
    from .backend import record_operation

    record_operation("semiflows")
    function = library.synkit_crn_semiflows
    function.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    function.restype = ctypes.c_int
    library.synkit_crn_free_text.argtypes = [ctypes.c_void_p]
    library.synkit_crn_free_text.restype = None
    output = ctypes.c_void_p()
    try:
        code = function(
            *matrix.shape,
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            time_limit,
            *limits,
            ctypes.byref(output),
        )
        if code or not output.value:
            raise RuntimeError("Native semiflow backend failed")
        payload = json.loads(ctypes.string_at(output).decode("ascii"))
    finally:
        if output.value:
            library.synkit_crn_free_text(output)
    flows = payload["flows"]
    # Preserve the established empty-matrix API.
    if not s.size:
        flows = []
    for row in flows:
        if len(row) != matrix.shape[0] or not any(row) or any(v < 0 for v in row):
            raise RuntimeError("Invalid native semiflow")
        if reduce(gcd, row, 0) != 1:
            raise RuntimeError("Native semiflow is not primitive")
        if any(
            sum(row[i] * int(matrix[i, j]) for i in range(len(row)))
            for j in range(matrix.shape[1])
        ):
            raise RuntimeError("Native semiflow fails exact kernel verification")
    flows.sort(key=lambda row: (sum(v != 0 for v in row), row))
    complete = payload["termination_reason"] == "complete"
    payload.update(
        flows=flows,
        backend="cpp",
        arithmetic="arbitrary_precision_integer",
        witness_verification="python_exact",
        seconds=monotonic() - started,
    )
    return AnalysisResult(
        payload,
        "complete" if complete else "inconclusive",
        complete,
        "native exact Farkas enumeration",
        termination_reason=payload["termination_reason"],
    )


def enumerate_structural_sets(
    crn,
    *,
    kind="siphon",
    backend="auto",
    library_path=None,
    max_nodes=200000,
    time_limit=30.0,
    max_results=10000,
):
    """Budgeted minimal siphons/traps, with explicit completeness and backend.

    Set any limit to None to disable it explicitly. Completed searches return
    the complete family of inclusion-minimal sets; partial searches expose
    their stopping reason and cannot certify the enumeration count.
    """
    from .backend import resolve_backend

    backend, library_path = resolve_backend(backend, library_path)
    if kind not in {"siphon", "trap"} or backend not in {"auto", "python", "cpp"}:
        raise ValueError("Invalid kind or backend")
    if time_limit is not None and (
        not np.isfinite(time_limit) or not 0 < time_limit <= 1e9
    ):
        raise ValueError(
            "time_limit must be None or finite and between zero and 1e9 seconds"
        )
    if any(
        v is not None
        and (isinstance(v, bool) or not isinstance(v, int) or not 0 < v < 2**63)
        for v in (max_nodes, max_results)
    ):
        raise ValueError("Search limits must be positive integers below 2**63")
    if backend == "auto":
        backend = "cpp" if library_path is not None else "python"
    net = crn if isinstance(crn, PetriNet) else PetriNet.from_syncrn(crn)
    places, transitions = net.place_order, net.transition_order
    index = {s: i for i, s in enumerate(places)}
    words = (len(places) + 63) // 64
    pre = np.zeros((len(transitions), words), dtype=np.uint64)
    post = pre.copy()
    for t, rid in enumerate(transitions):
        for side, array in (
            (net.transitions[rid].pre, pre),
            (net.transitions[rid].post, post),
        ):
            for s in side:
                i = index[s]
                array[t, i // 64] |= np.uint64(1 << (i % 64))
    trigger, required = (post, pre) if kind == "siphon" else (pre, post)
    if backend == "cpp":
        return _enumerate_structural_native(
            places,
            trigger,
            required,
            library_path=library_path,
            time_limit=time_limit,
            max_nodes=max_nodes,
            max_results=max_results,
        )

    return _enumerate_structural_python(
        places,
        trigger,
        required,
        time_limit=time_limit,
        max_nodes=max_nodes,
        max_results=max_results,
    )


def _structural_closure(allowed, trigger, required):
    current = np.zeros(trigger.shape[1], dtype=np.uint64)
    for i in allowed:
        current[i // 64] |= np.uint64(1 << (i % 64))
    changed = True
    while changed:
        changed = False
        for t in range(trigger.shape[0]):
            if np.any(trigger[t] & current) and not np.any(required[t] & current):
                current &= ~trigger[t]
                changed = True
    return frozenset(i for i in allowed if int(current[i // 64]) & (1 << (i % 64)))


def _enumerate_structural_python(
    places, trigger, required, *, time_limit, max_nodes, max_results
):
    def closure(allowed):
        return _structural_closure(allowed, trigger, required)

    started = monotonic()
    root = closure(frozenset(range(len(places))))
    stack, seen, results = ([root] if root else []), {root}, []
    nodes, reason = 0, "complete"
    while stack:
        current = stack.pop()
        minimal = True
        for p in sorted(current):
            if (max_nodes is not None and nodes >= max_nodes) or (
                time_limit is not None and monotonic() - started >= time_limit
            ):
                reason = (
                    "node_limit"
                    if max_nodes is not None and nodes >= max_nodes
                    else "time_limit"
                )
                break
            nodes += 1
            child = closure(current - {p})
            if child:
                minimal = False
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        if reason != "complete":
            break
        if minimal:
            results.append(current)
            if max_results is not None and len(results) >= max_results and stack:
                reason = "result_limit"
                break
    results = [s for s in results if not any(t < s for t in results)]
    value = {
        "sets": sorted([sorted(places[i] for i in s) for s in results]),
        "backend": "python",
        "closure_backend": "python",
        "enumeration_backend": "python",
        "closure_evaluations": nodes,
        "seconds": monotonic() - started,
    }
    return AnalysisResult(
        value,
        "complete" if reason == "complete" else "inconclusive",
        reason == "complete",
        "bitset closure descent",
        termination_reason=reason,
    )


def _enumerate_structural_native(
    places, trigger, required, *, library_path, time_limit, max_nodes, max_results
):
    library = _load_library(library_path)
    if not hasattr(
        library, "synkit_crn_structural_abi"
    ) or library.synkit_crn_structural_abi() not in (1, 2):
        raise RuntimeError(
            "Rebuild the CRN library to enable full native structural enumeration"
        )
    if library.synkit_crn_structural_abi() < 2 and any(
        value is None for value in (time_limit, max_nodes, max_results)
    ):
        raise RuntimeError(
            "Rebuild the CRN library to disable structural search limits"
        )
    from .backend import record_operation

    record_operation("structural_sets")
    function = library.synkit_crn_structural_sets
    pointer = ctypes.POINTER(ctypes.c_uint64)
    function.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        pointer,
        pointer,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    function.restype = ctypes.c_int
    library.synkit_crn_free_text.argtypes = [ctypes.c_void_p]
    library.synkit_crn_free_text.restype = None
    output = ctypes.c_void_p()
    started = monotonic()
    try:
        code = function(
            len(places),
            trigger.shape[0],
            trigger.ctypes.data_as(pointer),
            required.ctypes.data_as(pointer),
            0.0 if time_limit is None else time_limit,
            0 if max_nodes is None else max_nodes,
            0 if max_results is None else max_results,
            ctypes.byref(output),
        )
        if code or not output.value:
            raise RuntimeError("Native structural enumeration failed")
        payload = json.loads(ctypes.string_at(output).decode("ascii"))
    finally:
        if output.value:
            library.synkit_crn_free_text(output)
    payload["sets"] = sorted(
        sorted(places[i] for i in group) for group in payload["sets"]
    )
    payload.update(
        backend="cpp",
        closure_backend="cpp",
        enumeration_backend="cpp",
        seconds=monotonic() - started,
    )
    complete = payload["termination_reason"] == "complete"
    return AnalysisResult(
        payload,
        "complete" if complete else "inconclusive",
        complete,
        "native " + payload.get("algorithm", "bitset closure descent"),
        termination_reason=payload["termination_reason"],
    )
