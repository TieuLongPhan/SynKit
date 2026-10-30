"""Native exhaustive DFS and binary edit-support enumeration adapters."""

from .core import enumerate_core_mappings, map_reaction_core


def enumerate_dfs_mappings(graphs, *, library_path, **options):
    """Exhaustive compatible permutation traversal with no assignment bound."""
    return enumerate_core_mappings(
        graphs, library_path=library_path, backend="dfs", **options
    )


def enumerate_edit_mappings(graphs, *, library_path, **options):
    """Enumerate binary edge-flip supports then exact typed graph isomorphisms.

    This backend supports only binary bond presence. Supplying ``binary=True``
    explicitly converts a weighted input to that separate objective; requesting
    weighted EDIT without conversion returns unsupported from the native core.
    """
    return enumerate_core_mappings(
        graphs, library_path=library_path, backend="edit", **options
    )


def map_reaction_dfs(reaction, *, library_path, **options):
    return map_reaction_core(
        reaction, library_path=library_path, backend="dfs", **options
    )


def map_reaction_edit(reaction, *, library_path, **options):
    return map_reaction_core(
        reaction, library_path=library_path, backend="edit", **options
    )
