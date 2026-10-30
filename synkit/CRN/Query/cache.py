"""User-supplied offline KEGG caches, independent of research fixtures."""

from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Mapping
from .to_syncrn import syncrn_from_kegg_equations


def load_kegg_cache(path):
    """Read a KEGG module cache from an explicit path; never fetch data.

    Expected schema: {"modules": {module_id: {"equations": {...},
    "compound_names": {...}}}}. No manuscript modules or default file are implied.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("modules"), dict):
        raise ValueError("KEGG cache must contain a modules object")
    for module_id, entry in data["modules"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("equations"), dict):
            raise ValueError(f"{module_id}: expected an equations object")
        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in entry["equations"].items()
        ):
            raise ValueError(f"{module_id}: equations must map ids to equation strings")
    return data


def build_cached_kegg_crn(
    cache, module_id, *, drop_compounds=(), expand_reversible=True
):
    """Convert one cached module; retain all participants by default."""
    data = cache if isinstance(cache, Mapping) else load_kegg_cache(cache)
    entry = data["modules"][module_id]
    return syncrn_from_kegg_equations(
        entry["equations"],
        names=entry.get("compound_names"),
        drop_compounds=drop_compounds,
        expand_reversible=expand_reversible,
    )
