"""Isolated component-removal controls with identical indexed-map outputs.

The initial mapping is a deterministic, feasible heuristic, not an optimum or
database map. It is recomputed and charged to each seeded attempt.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import time


CONFIGURATIONS = {
    "full": {},
    "no_seed": {},
    "no_symmetry": {"symmetry_pruning": False, "expand_symmetry": False},
    # This removes the cost-bound bundle, including root profile filtering.
    # It is not a one-factor intervention on assignment bounds alone.
    "basic_bounds": {"assignment_lower_bound": False,
                     "assignment_upper_bound": False, "atom_profile_pruning": False},
    "no_profiles": {"atom_profile_pruning": False},
}


def feasible_seed(r, p):
    """Independent incident-histogram LAP followed by two greedy swap sweeps.

    No search bounds or reference mapping are used. Each sweep visits every
    compatible atom pair once and accepts strictly improving swaps only.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if Counter(r.atomic_numbers) != Counter(p.atomic_numbers):
        raise ValueError("Seed requires matching element inventories")
    n = len(r.atomic_numbers)
    def matrix(endpoint):
        a = np.zeros((n, n), dtype=np.int64)
        for i, j, order in endpoint.bonds:
            a[i, j] = a[j, i] = order
        return a
    a, b = matrix(r), matrix(p)
    def profiles(endpoint):
        result = [Counter() for _ in range(n)]
        for i, j, order in endpoint.bonds:
            result[i][endpoint.atomic_numbers[j], order] += 1
            result[j][endpoint.atomic_numbers[i], order] += 1
        return result
    rp, pp = profiles(r), profiles(p)
    mapping = np.zeros(n, dtype=int)
    for element in sorted(set(r.atomic_numbers)):
        rows = [i for i, z in enumerate(r.atomic_numbers) if z == element]
        cols = [j for j, z in enumerate(p.atomic_numbers) if z == element]
        costs = [[sum(abs(rp[i][k]-pp[j][k]) for k in rp[i].keys() | pp[j].keys())
                  for j in cols] for i in rows]
        rr, cc = linear_sum_assignment(costs)
        for i, j in zip(rr, cc):
            mapping[rows[i]] = cols[j]
    swaps = 0
    for _ in range(2):
        for i in range(n):
            for j in range(i+1, n):
                if r.atomic_numbers[i] != r.atomic_numbers[j]:
                    continue
                # The i--j bond is unchanged by exchanging its endpoints.
                keep = np.arange(n)
                keep = keep[(keep != i) & (keep != j)]
                images = mapping[keep]
                old = np.abs(a[i, keep]-b[mapping[i], images]).sum()
                old += np.abs(a[j, keep]-b[mapping[j], images]).sum()
                new = np.abs(a[i, keep]-b[mapping[j], images]).sum()
                new += np.abs(a[j, keep]-b[mapping[i], images]).sum()
                if new < old:
                    mapping[i], mapping[j] = mapping[j], mapping[i]
                    swaps += 1
    return tuple(map(int, mapping)), swaps


def enumerate_variant(r, p, *, variant, target, seconds=None, max_maps=None):
    from Experiment.Synister.global_milp import doubled_distance
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings

    if variant not in CONFIGURATIONS:
        raise ValueError("Unknown ablation configuration")
    started = time.perf_counter()
    seed, swaps = (None, 0) if variant == "no_seed" else feasible_seed(r, p)
    seed_seconds = time.perf_counter()-started
    seed_cost = doubled_distance(r, p, seed) if seed is not None else None
    maps, first = [], None
    def emit(mapping, cost):
        nonlocal first
        if first is None:
            first = time.perf_counter()-started
        maps.append(tuple(mapping))
    options = dict(symmetry_pruning=True, expand_symmetry=True,
                   assignment_lower_bound=True, assignment_upper_bound=True,
                   atom_profile_pruning=True, seed_center_order=True)
    options.update(CONFIGURATIONS[variant])
    before = time.perf_counter()
    raw = enumerate_distance_mappings(
        [r.graph(), p.graph()], CD="minimal" if target == "minimal" else target/2,
        binary=False, max_bijections=None, max_mappings=max_maps, tolerance=0,
        time_limit_seconds=None if seconds is None else max(0, seconds-(before-started)),
        symmetry_node_properties=("charges", "hcounts"), initial_mapping=seed,
        collect_mappings=False, mapping_callback=emit,
        compute_minimum_cost=target == "minimal", **options)
    finished = time.perf_counter()
    minimum = None if raw.minimum_cost is None else int(2*raw.minimum_cost)
    expected = minimum if target == "minimal" else target
    if len(maps) != len(set(maps)) or len(maps) != raw.selected_mapping_count:
        raise ValueError("Duplicate maps or callback count mismatch")
    if any(doubled_distance(r, p, m) != expected for m in maps):
        raise ValueError("Returned map has incorrect integer chemical distance")
    stats = (raw.backend_statistics or {}).get("search", {})
    return {"variant": variant, "options": options, "complete": raw.complete,
            "minimum_proved": target == "minimal" and minimum is not None,
            "minimum_doubled_cd": minimum, "target_doubled_cd": target,
            "termination": raw.truncation_reason or raw.status,
            "mappings": sorted(maps), "mapping_count": len(maps),
            "seed_mapping": seed, "seed_doubled_cd": seed_cost, "seed_swaps": swaps,
            "seed_seconds": seed_seconds, "first_map_seconds": first,
            "minimum_proof_seconds": stats.get("minimum_proof_seconds"),
            "search_seconds": finished-before,
            "checking_seconds": time.perf_counter()-finished,
            "variant_seconds": time.perf_counter()-started,
            "visited_nodes": raw.visited_nodes,
            "minimum_proof_nodes": stats.get("minimum_proof_nodes"),
            "pruned_branches": raw.pruned_branches,
            "lower_bound_pruned_branches": raw.lower_bound_pruned_branches,
            "upper_bound_pruned_branches": raw.upper_bound_pruned_branches,
            "symmetry_pruned_branches": raw.symmetry_pruned_branches,
            "symmetry_group_order": raw.symmetry_group_order,
            "backend_statistics": raw.backend_statistics,
            "output_unit": "all_indexed_atom_maps"}


def perform(task):
    from synkit.Chem.Mapper.identifiability import parse_reaction
    started = time.perf_counter()
    r, p = parse_reaction(task["reaction"])
    parsed = time.perf_counter()
    result = enumerate_variant(r, p, variant=task["variant"],
                               target=task["target_doubled_cd"],
                               seconds=max(0, task["seconds"]-(parsed-started)),
                               max_maps=task["max_maps"])
    exporting = time.perf_counter()
    maps = result.pop("mappings")
    data = json.dumps(maps, separators=(",", ":")).encode()
    with Path(task["map_path"]).open("xb") as stream:
        stream.write(data)
    result.update(mapping_sha256=sha256(data).hexdigest(), output_bytes=len(data),
                  parse_seconds=parsed-started, export_seconds=time.perf_counter()-exporting,
                  end_to_end_seconds=time.perf_counter()-started)
    return result


if __name__ == "__main__":
    task = json.load(sys.stdin)
    limit = int(task["memory_gib"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    before = time.perf_counter()
    try:
        result = perform(task)
    except Exception as exc:
        result = {"complete": False, "minimum_proved": False, "termination": "worker_error",
                  "error_type": type(exc).__name__, "error": str(exc)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(worker_seconds=time.perf_counter()-before,
                  cpu_seconds=usage.ru_utime+usage.ru_stime, peak_rss_kib=usage.ru_maxrss)
    print(json.dumps(result, allow_nan=False))
