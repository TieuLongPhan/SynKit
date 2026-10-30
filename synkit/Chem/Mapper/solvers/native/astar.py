"""Thin adapter for the full-space native best-first weighted A* backend."""

from .core import enumerate_core_mappings, map_reaction_core


def enumerate_astar_mappings(graphs, *, library_path, **options):
    """Enumerate the requested labeled shell; frontend classifies exact ITSs.

    All search decisions and admissible assignment bounds execute in C++.
    The backend uses no symmetry pruning. Its memory cap measures accounted
    frontier allocations, not whole-process RSS. A weighted chemical objective
    is optimized with heuristic multiplier one, preserving exactness.
    """
    return enumerate_core_mappings(
        graphs, library_path=library_path, backend="astar", **options
    )


def map_reaction_astar(reaction, *, library_path, **options):
    return map_reaction_core(
        reaction, library_path=library_path, backend="astar", **options
    )
