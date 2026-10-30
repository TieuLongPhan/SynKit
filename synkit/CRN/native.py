"""Optional native exact semiflow enumeration and siphon/trap closure.

Importing never invokes a compiler. Native builds are explicit and immutable.
"""

from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
import numpy as np
from .analysis import AnalysisResult
from .Petrinet.net import PetriNet


from ._build import CAPABILITIES, boost_include as _boost_include


def build_native(
    output_dir, compiler="c++", *, boost_include_dir=None, with_semiflows=None
):
    """Build the exact C++17 backend; auto-detect installed Boost headers.

    Set with_semiflows=True to require exact enumeration support, or False to
    build the standard-library-only structural backend. Nothing is downloaded.
    """
    import shlex
    import tempfile

    source_dir = Path(__file__).parent / "cpp"
    sources = [source_dir / "closure.cpp", source_dir / "structural.cpp"]
    flags = ["-std=c++17", "-O3", "-shared", "-fPIC"]
    boost = _boost_include(boost_include_dir)
    if with_semiflows is True and boost is None:
        raise RuntimeError(
            "Exact native semiflows require Boost.Multiprecision headers; supply boost_include_dir or BOOST_INCLUDE_DIR"
        )
    enabled = with_semiflows is not False and boost is not None
    if enabled:
        sources += [
            source_dir / name
            for name in ("semiflows.cpp", "algebra.cpp", "pathway.cpp", "symmetry.cpp")
        ]
        flags += ["-I", str(boost)]
    version = subprocess.check_output([compiler, "--version"], text=True)
    # Hash the compiler-reported transitive header dependencies as well as sources.
    dependencies = {}
    for source in sources:
        listing = subprocess.check_output(
            [
                compiler,
                "-MM",
                "-std=c++17",
                *(["-I", str(boost)] if enabled else []),
                str(source),
            ],
            text=True,
        )
        for filename in shlex.split(listing.replace("\\\n", " ").split(":", 1)[1]):
            path = Path(filename)
            dependencies[str(path.resolve())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    capabilities = list(CAPABILITIES if enabled else CAPABILITIES[:4])
    digest = hashlib.sha256(
        json.dumps(
            [dependencies, flags, version, sys.platform, capabilities], sort_keys=True
        ).encode()
    ).hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = output / ("libsynkit_crn_" + digest[:16] + suffix)
    manifest_path = library.with_suffix(".json")
    if library.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            hashlib.sha256(library.read_bytes()).hexdigest()
            != manifest["library_sha256"]
        ):
            raise RuntimeError("Existing immutable CRN library checksum mismatch")
        return library
    with tempfile.TemporaryDirectory(prefix="crn-build-", dir=output) as temp:
        temporary = Path(temp) / library.name
        subprocess.run(
            [compiler, *flags, *map(str, sources), "-o", str(temporary)], check=True
        )
        manifest = {
            "abi": 1,
            "semiflows_abi": 1 if enabled else None,
            "capabilities": capabilities,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
            "dependencies_sha256": dependencies,
            "build_sha256": digest,
            "compiler": version,
            "flags": flags,
            "library_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
        }
        temporary.replace(library)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return library


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

    def closure(allowed):
        current = np.zeros(words, dtype=np.uint64)
        for i in allowed:
            current[i // 64] |= np.uint64(1 << (i % 64))
        changed = True
        while changed:
            changed = False
            for t in range(len(transitions)):
                if np.any(trigger[t] & current) and not np.any(required[t] & current):
                    current &= ~trigger[t]
                    changed = True
        return frozenset(i for i in allowed if int(current[i // 64]) & (1 << (i % 64)))

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
        "backend": backend,
        "closure_backend": backend,
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
