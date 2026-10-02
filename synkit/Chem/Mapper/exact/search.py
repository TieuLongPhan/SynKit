"""PABS: Propagation and Assignment-Bounded Search, with explicit backends."""

from dataclasses import replace

from .propagation import PropagationConfig, enumerate_synister_cp_mappings
from .multi_shell import enumerate_pabs_shells


def enumerate_pabs_mappings(lgp, *, backend="python", library_path=None, **options):
    """Enumerate minimum or specific-CD mappings through a Python interface.

    ``python`` selects the propagation solver. ``cpp`` selects the optional
    compiled integer shell search and requires an explicitly built library.
    Both return DistanceEnumerationResult. Backend-specific unsupported options
    raise errors; selecting C++ never silently executes Python search.
    The C++ traversal is distinct from the Python propagation implementation.
    """
    if backend not in ("python", "cpp"):
        raise ValueError("backend must be 'python' or 'cpp'")
    if backend == "cpp":
        if library_path is None:
            raise ValueError(
                "backend='cpp' requires library_path; build with native_build"
            )
        from .native_search import enumerate_cpp_mappings

        return enumerate_cpp_mappings(lgp, library_path=library_path, **options)
    if library_path is not None:
        raise ValueError("library_path is only used by backend='cpp'")
    result = enumerate_synister_cp_mappings(lgp, **options)
    statistics = dict(result.backend_statistics or {})
    statistics.update(
        method="PABS", requested_backend="python", implementation=result.backend
    )
    return replace(result, backend_statistics=statistics)


__all__ = [
    "PropagationConfig",
    "enumerate_pabs_mappings",
    "enumerate_pabs_shells",
]
