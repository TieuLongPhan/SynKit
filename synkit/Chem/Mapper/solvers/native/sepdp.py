"""Exact separator-table DP with global target injectivity and weighted costs."""

import ctypes as ct
from pathlib import Path

from .core import enumerate_core_mappings, map_reaction_core


class _Profile(ct.Structure):
    _fields_ = [
        (name, ct.c_int64)
        for name in (
            "separator_width",
            "states",
            "predecessor_arcs",
            "transitions",
            "peak_accounted_table_bytes",
        )
    ]


def sepdp_profile(library_path):
    library = ct.CDLL(str(Path(library_path).resolve()))
    function = library.synkit_mapper_sepdp_profile
    function.argtypes = [ct.POINTER(_Profile)]
    function.restype = None
    profile = _Profile()
    function(ct.byref(profile))
    return {name: getattr(profile, name) for name, _ in _Profile._fields_}


def enumerate_sepdp_mappings(graphs, *, library_path, **options):
    """Exact arbitrary weighted graphs up to the declared 63-vertex mask limit.

    State and memory growth remain exponential in target subsets and separator
    width. The memory cap measures accounted retained table/DAG payload, not
    process RSS or a theorem about polynomial general graph edit distance.
    """
    return enumerate_core_mappings(
        graphs, library_path=library_path, backend="sepdp", **options
    )


def map_reaction_sepdp(reaction, *, library_path, **options):
    return map_reaction_core(
        reaction, library_path=library_path, backend="sepdp", **options
    )
