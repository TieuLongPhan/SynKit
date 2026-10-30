"""Versioned ctypes declarations for the common mapper C ABI."""

import ctypes as ct
import sys
from functools import lru_cache
from pathlib import Path

BACKENDS = {
    "slap": 0,
    "pabs": 1,
    "cp_sat": 2,
    "scip": 3,
    "maxsat": 4,
    "astar": 5,
    "cpabs": 6,
    "abb": 7,
    "kbb": 8,
    "kmilp": 9,
    "edit": 10,
    "dfs": 11,
    "mces": 12,
    "sepdp": 13,
}


def resolve_library_path(library_path=None):
    """Use an explicit library or the wheel's bundled core; never compile implicitly."""
    if library_path is not None:
        return Path(library_path).expanduser().resolve()
    name = (
        "synkit_mapper.dll" if sys.platform == "win32"
        else "libsynkit_mapper.dylib" if sys.platform == "darwin"
        else "libsynkit_mapper.so"
    )
    library = Path(__file__).resolve().parent / "lib" / name
    if not library.is_file():
        raise FileNotFoundError(
            "The bundled C++ mapper library is missing. Install a SynKit platform "
            "wheel, or build with build_mapper_native and pass library_path explicitly. "
            "Source checkouts and editable installs require an explicit native build."
        )
    return library


def available_native_methods(library_path=None):
    """Return the solver configurations advertised by the selected compiled core."""
    library = load_core(str(resolve_library_path(library_path)))
    capabilities = library.synkit_mapper_capabilities()
    return tuple(name for name, index in BACKENDS.items() if capabilities & (1 << index))


@lru_cache(maxsize=8)
def has_core(path):
    library = ct.CDLL(str(Path(path).resolve()))
    if not hasattr(library, "synkit_mapper_core_abi"):
        return False
    if library.synkit_mapper_core_abi() != 1:
        raise RuntimeError("Unsupported common mapper ABI")
    return True


REASONS = {
    0: None,
    1: "time_limit",
    2: "max_states",
    3: "max_mappings",
    4: "max_classes",
    5: "callback",
    6: "max_frontier",
    7: "memory_limit",
    8: "unsupported",
    9: "heuristic",
    10: "cancelled",
}
INT = ct.POINTER(ct.c_int32)
LONG = ct.POINTER(ct.c_int64)
CALLBACK = ct.CFUNCTYPE(ct.c_int, INT, ct.c_int64, ct.c_void_p)


class Problem(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_int32),
        ("n", ct.c_int32),
        ("a2", LONG),
        ("b2", LONG),
    ] + [
        (name, INT)
        for name in (
            "types_r",
            "types_p",
            "node_colors_r",
            "node_colors_p",
            "edge_colors_r",
            "edge_colors_p",
        )
    ]


class Request(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_int32)
            for name in (
                "abi_version",
                "backend",
                "query",
                "symmetry",
                "classify",
                "workers",
            )
        ]
        + [("target2", ct.c_int64), ("seconds", ct.c_double)]
        + [
            (name, ct.c_int64)
            for name in (
                "max_states",
                "max_candidates",
                "max_classes",
                "max_frontier",
                "memory_limit_bytes",
            )
        ]
        + [
            ("slap_max_fingerprints", ct.c_int32),
            ("slap_stop_flags", INT),
            ("cancel_token", ct.c_void_p),
        ]
    )


class Statistics(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_int32)
            for name in (
                "abi_version",
                "backend",
                "status",
                "reason",
                "minimum_proven",
                "search_complete",
                "classification_complete",
                "hydrogen_complete",
                "heuristic_finished",
            )
        ]
        + [
            (name, ct.c_int64)
            for name in (
                "minimum2",
                "incumbent2",
                "lower_bound2",
                "nodes",
                "leaves",
                "candidate_count",
                "class_count",
                "labeled_count",
                "peak_frontier",
                "peak_frontier_bytes",
                "peak_memory_bytes",
            )
        ]
        + [
            (name, ct.c_double)
            for name in (
                "model_seconds",
                "optimization_seconds",
                "enumeration_seconds",
                "classification_seconds",
                "callback_seconds",
                "elapsed_seconds",
            )
        ]
    )


@lru_cache(maxsize=8)
def load_core(path):
    path = Path(path).resolve()
    if path.parent == Path(__file__).resolve().parent / "lib":
        from synkit._native import verify_wheel_library

        verify_wheel_library(path)
    library = ct.CDLL(str(path))
    try:
        if library.synkit_mapper_core_abi() != 1:
            raise RuntimeError("Unsupported common mapper ABI")
    except AttributeError as error:
        raise RuntimeError(
            "Library lacks common mapper ABI; build with build_mapper_native"
        ) from error
    library.synkit_mapper_capabilities.restype = ct.c_uint64
    library.synkit_mapper_core_error.restype = ct.c_char_p
    if hasattr(library, "synkit_mapper_backend_version"):
        library.synkit_mapper_backend_version.argtypes = [ct.c_int]
        library.synkit_mapper_backend_version.restype = ct.c_char_p
    if hasattr(library, "synkit_mapper_backend_metadata"):
        library.synkit_mapper_backend_metadata.argtypes = [ct.c_int]
        library.synkit_mapper_backend_metadata.restype = ct.c_char_p
    library.synkit_mapper_solve.argtypes = [
        ct.POINTER(Problem),
        ct.POINTER(Request),
        CALLBACK,
        ct.c_void_p,
        ct.POINTER(Statistics),
    ]
    library.synkit_mapper_solve.restype = ct.c_int
    if hasattr(library, "synkit_mapper_solve_domains"):
        library.synkit_mapper_solve_domains.argtypes = [
            ct.POINTER(Problem),
            ct.POINTER(Request),
            ct.POINTER(ct.c_uint8),
            CALLBACK,
            ct.c_void_p,
            ct.POINTER(Statistics),
        ]
        library.synkit_mapper_solve_domains.restype = ct.c_int
    library.synkit_mapper_cancel_create.restype = ct.c_void_p
    for name in ("synkit_mapper_cancel_request", "synkit_mapper_cancel_free"):
        getattr(library, name).argtypes = [ct.c_void_p]
    return library


class CancellationToken:
    def __init__(self, library_path):
        self.library = load_core(str(Path(library_path).resolve()))
        self.pointer = self.library.synkit_mapper_cancel_create()

    def cancel(self):
        if self.pointer:
            self.library.synkit_mapper_cancel_request(self.pointer)

    def close(self):
        if self.pointer:
            self.library.synkit_mapper_cancel_free(self.pointer)
            self.pointer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
