"""Opt-in canonical construction traversal and its measured symmetry work."""

import ctypes as ct
from pathlib import Path


class _Profile(ct.Structure):
    _fields_ = [
        (name, ct.c_int64)
        for name in (
            "discovery_nodes",
            "subgroup_order",
            "canonical_calls",
            "augmentations",
            "canonical_rejections",
            "bound_rejections",
        )
    ] + [
        ("identity_fallback_r", ct.c_int32),
        ("identity_fallback_p", ct.c_int32),
        ("discovery_seconds", ct.c_double),
        ("canonical_seconds", ct.c_double),
    ]


class _ProfileV2(ct.Structure):
    _fields_ = (
        [("base", _Profile)]
        + [
            (name, ct.c_int64)
            for name in (
                "twin_blocks_r",
                "twin_blocks_p",
                "quotient_order_r",
                "quotient_order_p",
                "bound_calls",
                "seed_cost2",
            )
        ]
        + [("bound_seconds", ct.c_double), ("seed_seconds", ct.c_double)]
    )


def cpabs_profile(library_path):
    """Read same-thread diagnostics from the latest cPABS run in this library."""
    library = ct.CDLL(str(Path(library_path).resolve()))
    function = library.synkit_mapper_cpabs_profile
    function.argtypes = [ct.POINTER(_Profile)]
    function.restype = None
    value = _Profile()
    function(ct.byref(value))
    result = {name: getattr(value, name) for name, _ in _Profile._fields_}
    if hasattr(library, "synkit_mapper_cpabs_profile_v2"):
        function = library.synkit_mapper_cpabs_profile_v2
        function.argtypes = [ct.POINTER(_ProfileV2)]
        function.restype = None
        detail = _ProfileV2()
        function(ct.byref(detail))
        result.update(
            {
                name: getattr(detail, name)
                for name, _ in _ProfileV2._fields_
                if name != "base"
            }
        )
    return result


def enumerate_cpabs_mappings(lgp, *, library_path, **kwargs):
    """Run the complete canonical-parent traversal through the shared adapter.

    Symmetry pruning is explicitly enabled with ``symmetry=True``. The method
    changes the extension tree to unordered partial matchings; it does not add
    a canonical deletion filter to fixed/adaptive source-row PABS. Exact twin
    factors remain active even when quotient-group discovery is capped.
    Raw output enumerates subgroup orbits and requires class deduplication.
    """
    from .core import enumerate_core_mappings

    return enumerate_core_mappings(
        lgp, library_path=library_path, backend="cpabs", **kwargs
    )


def map_reaction_cpabs(reaction, *, library_path, **kwargs):
    """Use the shared chemistry frontend with the explicit native variant."""
    from .core import map_reaction_core

    return map_reaction_core(
        reaction, backend="cpabs", library_path=library_path, **kwargs
    )
