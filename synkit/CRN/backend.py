"""Explicit, task-local native backend selection; never compiles on import."""

from contextlib import contextmanager
from contextvars import ContextVar

_LIBRARY = ContextVar("synkit_crn_native_library", default=None)
_OPERATIONS = ContextVar("synkit_crn_native_operations", default=None)


def record_operation(name):
    operations = _OPERATIONS.get()
    if operations is not None:
        operations[name] = operations.get(name, 0) + 1


def backend_metadata():
    from .native import _library_manifest

    library = _LIBRARY.get()
    metadata = {
        "production_backend": "cpp" if library is not None else "python",
        "library_path": str(library) if library is not None else None,
        "native_operations": dict(_OPERATIONS.get() or {}),
    }
    if library is not None:
        manifest = _library_manifest(library)
        metadata.update(
            library_sha256=manifest["library_sha256"],
            build_sha256=manifest["build_sha256"],
            capabilities=manifest.get("capabilities", []),
        )
    return metadata


def resolve_backend(backend="auto", library_path=None):
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("backend must be auto, python or cpp")
    library_path = library_path if library_path is not None else _LIBRARY.get()
    if backend == "python":
        return "python", None
    if backend == "cpp" and library_path is None:
        from .native import bundled_native_library

        try:
            library_path = bundled_native_library()
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
    return ("cpp", library_path) if library_path is not None else ("python", None)


@contextmanager
def native_backend(library_path=None):
    """Use a verified native library throughout a Python analysis workflow."""
    from .native import _load_library

    _, library_path = resolve_backend("cpp", library_path)
    _load_library(library_path)
    token = _LIBRARY.set(library_path)
    operations_token = _OPERATIONS.set({})
    try:
        yield
    finally:
        _LIBRARY.reset(token)
        _OPERATIONS.reset(operations_token)


@contextmanager
def python_backend():
    """Temporarily select the independent Python reference implementation."""
    token = _LIBRARY.set(None)
    try:
        yield
    finally:
        _LIBRARY.reset(token)
