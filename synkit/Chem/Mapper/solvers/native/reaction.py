"""Thin bindings for the complete optional RDKit C++ reaction pipeline."""

import ctypes as ct
import json
import math
from functools import lru_cache
from numbers import Integral
from pathlib import Path

from ..exact.distance_search.records import normalize_distance_target
from .runtime import BACKENDS, Request


class _Options(ct.Structure):
    _fields_ = [
        (name, ct.c_int32)
        for name in (
            "abi_version",
            "explicit_hydrogens",
            "lewis_objective",
            "dummy_balance",
            "binary",
            "hydrogen_quotient",
            "empty_numeric_shell",
        )
    ] + [("max_canonical_nodes", ct.c_int64), ("max_hydrogen_plans", ct.c_int64)]


@lru_cache(maxsize=8)
def _load(path):
    library = ct.CDLL(str(Path(path).resolve()))
    if library.synkit_mapper_reaction_abi() != 1:
        raise RuntimeError("Unsupported native reaction ABI")
    library.synkit_mapper_reaction_error.restype = ct.c_char_p
    library.synkit_mapper_rdkit_version.restype = ct.c_char_p
    library.synkit_mapper_reaction_solve.argtypes = [
        ct.c_char_p,
        ct.POINTER(_Options),
        ct.POINTER(Request),
        ct.POINTER(ct.c_void_p),
    ]
    library.synkit_mapper_reaction_free.argtypes = [ct.c_void_p]
    return library


def _cap(value, name):
    if value is None:
        return -1
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 < value < 2**63
    ):
        raise ValueError(f"{name} must be a positive int64 or None")
    return int(value)


# ABI option checks and hydrogen fallback share the native reaction deadline.
def map_reaction_native(  # noqa: C901
    reaction,
    *,
    library_path,
    backend="pabs",
    CD="minimal",
    CD_mode="exact",
    objective="lewis",
    hydrogen=None,
    balance="strict",
    binary=False,
    hydrogen_quotient=False,
    time_limit_seconds=10,
    max_mappings=100000,
    max_classes=None,
    max_states=None,
    max_frontier=None,
    memory_limit_bytes=None,
    classification_max_search_nodes=100000,
    max_hydrogen_plans=10000,
    proof_only=False,
    cancel_token=None,
):
    """Return native JSON records after C++ parsing, search and mapped rendering.

    This adapter imports no Python RDKit chemistry functions. It loads the
    explicitly built reaction library, converts options and decodes its JSON.
    Unsupported H quotient states fall back to full explicit native search on
    the remaining shared budget; the result records that fallback and its reason.
    """
    if not isinstance(reaction, str):
        raise TypeError("reaction must be reaction SMILES")
    if "\x00" in reaction:
        raise ValueError("reaction cannot contain a NUL character")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown native backend: {backend}")
    if objective not in {"lewis", "bond"} or balance not in {"strict", "dummy"}:
        raise ValueError("objective must be lewis/bond and balance strict/dummy")
    hydrogen = hydrogen or ("explicit" if objective == "lewis" else "heavy")
    if hydrogen not in {"explicit", "heavy"}:
        raise ValueError(
            "Native reaction frontend requires explicit or heavy hydrogens"
        )
    for name, flag in (
        ("binary", binary),
        ("hydrogen_quotient", hydrogen_quotient),
        ("proof_only", proof_only),
    ):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be boolean")
    if objective == "lewis" and (hydrogen != "explicit" or binary):
        raise ValueError("Lewis objective requires explicit H and weighted bonds")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, (int, float))
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative or None")
    target = normalize_distance_target(CD)
    if CD_mode not in {"exact", "at_most"} or (
        CD_mode == "at_most" and target == "minimal"
    ):
        raise ValueError("at_most requires numeric CD; otherwise use exact")
    if proof_only and target != "minimal":
        raise ValueError("proof_only requires minimal CD")
    query = (
        3
        if proof_only
        else 0
        if target == "minimal"
        else 2
        if CD_mode == "at_most"
        else 1
    )
    empty = False
    target2 = 0
    if target != "minimal":
        scaled = min(target * 2, 2**62)
        target2 = math.floor(scaled + 1e-9) if query == 2 else round(scaled)
        empty = query == 1 and abs(target - target2 / 2) > 1e-9
    request = Request()
    request.abi_version = 1
    request.backend = BACKENDS[backend]
    request.query = query
    request.symmetry = int(backend in {"pabs", "cpabs"})
    request.workers = 1
    request.target2 = target2
    request.seconds = -1 if time_limit_seconds is None else time_limit_seconds
    request.slap_max_fingerprints = 10000
    for name, value in (
        ("max_states", max_states),
        ("max_candidates", max_mappings),
        ("max_classes", max_classes),
        ("max_frontier", max_frontier),
        ("memory_limit_bytes", memory_limit_bytes),
    ):
        setattr(request, name, _cap(value, name))
    request.cancel_token = None if cancel_token is None else cancel_token.pointer
    options = _Options(
        1,
        hydrogen == "explicit",
        objective == "lewis",
        balance == "dummy",
        binary,
        hydrogen_quotient,
        empty,
        _cap(classification_max_search_nodes, "classification_max_search_nodes"),
        _cap(max_hydrogen_plans, "max_hydrogen_plans"),
    )
    library = _load(str(library_path))
    output = ct.c_void_p()
    result = library.synkit_mapper_reaction_solve(
        reaction.encode(), ct.byref(options), ct.byref(request), ct.byref(output)
    )
    if result:
        raise ValueError(library.synkit_mapper_reaction_error().decode())
    try:
        return json.loads(ct.string_at(output))
    finally:
        library.synkit_mapper_reaction_free(output)
