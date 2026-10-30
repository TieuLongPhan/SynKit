"""Dispatch full-space exact mapping to Python propagation or C++ shell search.

PABS stands for Propagation and Assignment-Bounded Search. It is this package's
descriptive interface name. The Python implementation combines one-to-one
domain propagation and LAP cost bounds; the C++ implementation has a separate
integer-shell traversal. These routes do not use the reduced SLAP uncertainty
kernel or its MILP solver.
"""

from dataclasses import replace

from .pabs.propagation import PropagationConfig, enumerate_synister_cp_mappings
from .pabs.multi_shell import enumerate_pabs_shells


def enumerate_pabs_mappings(lgp, *, backend="python", library_path=None, **options):
    """Enumerate a minimum or selected-distance shell with an exact backend.

    PABS means Propagation and Assignment-Bounded Search. ``backend='python'``
    is the default and selects full-space propagation with linear-assignment
    bounds. It does not run the SLAP mapper or extract its uncertainty kernel.
    Explicit fixed assignments restrict the search if supplied by the caller.
    Unsupported Python inputs use a recorded full-distance-search fallback.

    ``cpp`` selects the optional
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
        from .native.search import enumerate_cpp_mappings

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
