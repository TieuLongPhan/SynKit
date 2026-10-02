"""Isolated, resource-limited worker for matched indexed-map enumeration."""

import json
from math import gcd
from pathlib import Path
import resource
import sys
import time


def perform(task):
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from Experiment.Synister.global_milp import doubled_distance, enumerate_milp

    started = time.perf_counter()
    r, p = parse_reaction(task["reaction"])
    parsed = time.perf_counter()
    target = task["target_doubled_cd"]
    remaining = max(0, task["seconds"] - (parsed-started))
    if task["method"] not in ("milp", "synister"):
        raise ValueError("Unknown enumeration method")
    constant = sum(w for _, _, w in r.bonds) + sum(w for _, _, w in p.bonds)
    divisor = gcd(*(w for _, _, w in (*r.bonds, *p.bonds)))
    rejection = None
    if target != "minimal":
        if target > constant:
            rejection = "bond_mass_upper_bound"
        elif (not divisor and target != constant) or (divisor and (constant-target) % (2*divisor)):
            rejection = "bond_order_congruence"
    if rejection:
        result = {"method": task["method"], "target_doubled_cd": target,
                  "complete": True, "minimum_proved": False, "minimum_doubled_cd": None,
                  "minimum_proof_seconds": None, "first_map_seconds": None,
                  "mappings": [], "termination": "proved_empty_precheck", "precheck": rejection,
                  "elapsed_seconds": time.perf_counter()-parsed}
    elif task["method"] == "milp":
        result = enumerate_milp(r, p, target=target, seconds=remaining, max_maps=task["max_maps"])
    elif task["method"] == "synister":
        from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
        maps, first = [], None
        def emit(mapping, cost):
            nonlocal first
            if first is None:
                first = time.perf_counter()-started
            maps.append(tuple(mapping))
        before = time.perf_counter()
        raw = enumerate_distance_mappings(
            [r.graph(), p.graph()], CD="minimal" if target == "minimal" else target/2,
            binary=False, max_bijections=None, max_mappings=task["max_maps"], tolerance=0,
            time_limit_seconds=max(0, task["seconds"]-(before-started)),
            symmetry_pruning=True, expand_symmetry=True,
            symmetry_node_properties=("charges", "hcounts"),
            collect_mappings=False, mapping_callback=emit, compute_minimum_cost=target == "minimal")
        search = (raw.backend_statistics or {}).get("search", {})
        result = {"method": "synister", "target_doubled_cd": target,
                  "complete": raw.complete, "minimum_proved": target == "minimal" and raw.minimum_cost is not None,
                  "minimum_doubled_cd": None if raw.minimum_cost is None else int(2*raw.minimum_cost),
                  "minimum_proof_seconds": search.get("minimum_proof_seconds"),
                  "first_map_seconds": first, "mappings": maps,
                  "termination": raw.truncation_reason or raw.status,
                  "visited_nodes": raw.visited_nodes,
                  "pruned_branches": raw.pruned_branches,
                  "backend_statistics": raw.backend_statistics,
                  "symmetry_group_order": raw.symmetry_group_order,
                  "reported_mapping_count": raw.selected_mapping_count,
                  "elapsed_seconds": time.perf_counter()-before}
    else:
        raise ValueError("Unknown enumeration method")
    finished_search = time.perf_counter()
    mappings = result.pop("mappings")
    expected = result["minimum_doubled_cd"] if target == "minimal" else target
    # Partial sets are checked too; validity is not a completeness guarantee.
    for mapping in mappings:
        if doubled_distance(r, p, mapping) != expected:
            raise ValueError("Returned map differs from the requested integer CD")
    unique = set(map(tuple, mappings))
    if len(unique) != len(mappings):
        raise ValueError("Duplicate indexed map")
    if "reported_mapping_count" in result and result["reported_mapping_count"] != len(mappings):
        raise ValueError("Callback output disagrees with reported count")
    checked = time.perf_counter()
    from hashlib import sha256
    canonical = json.dumps(sorted(unique), separators=(",", ":")).encode()
    export = Path(task["map_path"])
    with export.open("xb") as handle:
        handle.write(canonical)
    result.update(mapping_count=len(mappings), mapping_sha256=sha256(canonical).hexdigest(),
                  output_bytes=len(canonical), parse_seconds=parsed-started,
                  search_finished_seconds=finished_search-started,
                  checking_seconds=checked-finished_search,
                  export_seconds=time.perf_counter()-checked,
                  end_to_end_seconds=time.perf_counter()-started,
                  output_unit="all_indexed_atom_maps")
    return result


def main():
    task = json.load(sys.stdin)
    limit = int(task["memory_gib"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    try:
        result = perform(task)
    except Exception as exc:
        result = {"complete": False, "minimum_proved": False,
                  "termination": "worker_error", "error_type": type(exc).__name__, "error": str(exc)}
    result.update(worker_seconds=time.perf_counter()-started,
                  cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                              resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
