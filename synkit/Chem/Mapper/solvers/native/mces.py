"""Weighted common-edge overlap mapping through a native association graph."""

from .core import enumerate_core_mappings, map_reaction_core


def enumerate_mces_mappings(lgp, *, library_path, **options):
    """Enumerate full-bijection minimum/exact/at-most weighted Lewis mappings."""
    return enumerate_core_mappings(
        lgp, library_path=library_path, backend="mces", **options
    )


def map_reaction_mces(reaction, *, library_path, **options):
    return map_reaction_core(
        reaction, library_path=library_path, backend="mces", **options
    )
