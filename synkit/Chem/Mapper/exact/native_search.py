"""Labeled mapping adapter for the optional C++ integer shell kernel."""

import math
import time
from collections import Counter
from numbers import Integral, Real

import numpy as np

from ..slap.lap import _adjacency_and_elements
from .distance_records import (
    DistanceEnumerationResult,
    ExactEnumerationLimitError,
    normalize_distance_target,
)
from .native_candidates import enumerate_native_candidates, prepare_native_candidates
from .cost_lattice import CostLattice


def enumerate_cpp_mappings(  # noqa: C901
    lgp,
    *,
    library_path,
    CD="minimal",
    binary=True,
    initial_mapping=None,
    max_bijections=1_000_000,
    max_mappings=None,
    time_limit_seconds=None,
    collect_mappings=True,
    mapping_callback=None,
    compute_minimum_cost=True,
    tolerance=1e-9,
):
    """Search exact labeled shells, proving minima by exhausting lower shells.

    Requires 1..256 atoms, symmetric loop-free half-integer weights, <=16 bond
    levels, and the native kernel's integer range. No automorphism quotient is
    applied, so output is directly comparable with Python labeled mappings.
    Fixed assignments, propagation config and replay certificates are currently
    Python-only. All passes share one deadline; an interrupted proof never sets
    minimum_cost. Compilation is explicit and never occurs during import.
    """
    started = time.perf_counter()
    target = normalize_distance_target(CD)
    for name, value in (
        ("binary", binary),
        ("collect_mappings", collect_mappings),
        ("compute_minimum_cost", compute_minimum_cost),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    for name, value in (
        ("max_bijections", max_bijections),
        ("max_mappings", max_mappings),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and non-negative")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 0.25
    ):
        raise ValueError("C++ tolerance must be finite, nonnegative and less than 0.25")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable")
    a, labels = _adjacency_and_elements(lgp[0], binary)
    b, product_labels = _adjacency_and_elements(lgp[1], binary)
    if a.shape != b.shape or Counter(labels) != Counter(product_labels):
        raise ValueError("Endpoints must have equal atom-type inventories")
    total = math.prod(math.factorial(count) for count in Counter(labels).values())
    if max_bijections is not None and total > max_bijections:
        raise ExactEnumerationLimitError(
            f"{total:,} bijections exceed max_bijections={max_bijections:,}"
        )
    maximum = float(np.abs(a).sum() + np.abs(b).sum()) / 2
    if maximum > 250000:
        raise ValueError("C++ total distance range exceeds 250000")
    prepared = prepare_native_candidates(
        lgp,
        0,
        library_path=library_path,
        initial_mapping=initial_mapping,
        binary=binary,
        symmetry=False,
    )
    lattice = CostLattice.from_matrices(np.rint(a * 4), np.rint(b * 4))
    deadline = None if time_limit_seconds is None else started + time_limit_seconds
    nodes = leaves = pruned = probes = 0
    minimum = None
    mappings = []
    count = 0
    reason = None

    def run(shell, cap, callback):
        nonlocal nodes, leaves, pruned
        remaining = (
            1e100 if deadline is None else max(0, deadline - time.perf_counter())
        )
        record = enumerate_native_candidates(
            lgp,
            shell,
            library_path=library_path,
            time_limit_seconds=remaining,
            max_mappings=cap,
            callback=callback,
            _prepared=prepared,
        )
        nodes += int(record["visited_nodes"])
        leaves += int(record["visited_leaves"])
        pruned += int(record["pruned"])
        return record

    if target == "minimal" or compute_minimum_cost:
        # Every pair cost lies on the half-unit lattice, including signed bonds.
        for tick in range(round(maximum * 2) + 1):
            if not lattice.intersects(tick * 2, tick * 2):
                continue
            found = []
            record = run(tick / 2, 1, lambda mapping, cost: found.append(mapping))
            probes += 1
            if found:
                minimum = tick / 2
                break
            if not record["complete"]:
                reason = record["reason"]
                break

    def receive(mapping, cost):
        nonlocal count
        count += 1
        if collect_mappings:
            mappings.append(list(mapping))
        if mapping_callback is not None:
            mapping_callback(list(mapping), cost)

    selected = minimum if target == "minimal" else float(target)
    selected_cost = None
    if reason is None and selected is not None:
        lattice_shell = round(selected * 2) / 2
        if abs(selected - lattice_shell) <= tolerance and lattice_shell <= maximum:
            selected_cost = lattice_shell
            record = (
                run(
                    lattice_shell,
                    None if max_mappings is None else int(max_mappings),
                    receive,
                )
                if lattice.intersects(
                    round(lattice_shell * 4), round(lattice_shell * 4)
                )
                else None
            )
            reason = None if record is None else record["reason"]
    complete = reason is None
    return DistanceEnumerationResult(
        target=target,
        cost=(selected_cost if count else None),
        minimum_cost=minimum,
        mappings=mappings,
        distances=[selected_cost] * len(mappings),
        total_bijections=total,
        maximum_cost_upper_bound=maximum,
        visited_leaves=leaves,
        pruned_branches=pruned,
        visited_nodes=nodes,
        elapsed_seconds=time.perf_counter() - started,
        status=("complete" if count else "no_solutions") if complete else "timeout",
        complete=complete,
        truncation_reason=reason,
        selected_mapping_count=count,
        selected_labeled_mapping_count=count,
        backend="pabs_cpp",
        backend_statistics={
            "method": "PABS",
            "requested_backend": "cpp",
            "implementation": "native_integer_shell_search",
            "minimum_shell_probes": probes,
            "library_path": str(library_path),
            "cost_lattice_modulus_quarters": lattice.modulus,
            "cost_lattice_residue_quarters": lattice.residue,
        },
    )


__all__ = ["enumerate_cpp_mappings"]
