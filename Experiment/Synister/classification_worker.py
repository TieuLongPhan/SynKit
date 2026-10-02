"""Resource-limited full-ITS classification of an already enumerated CD set.

The upstream result supplies indexed maps, not a chosen representative. A newly
declared verified cyclic product subgroup accelerates structural classification;
it need not be the subgroup used during the earlier search.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import time


def verify_cyclic_group(endpoint, group):
    n = len(endpoint.atomic_numbers)
    identity = tuple(range(n))
    group = tuple(tuple(g) for g in group)
    if not group or group[0] != identity or len(set(group)) != len(group):
        raise ValueError("Invalid cyclic subgroup identity or duplicate elements")
    original = set(endpoint.bonds)
    for g in group:
        if sorted(g) != list(range(n)):
            raise ValueError("Product symmetry is not a permutation")
        for values in (endpoint.atomic_numbers, endpoint.charges, endpoint.hcounts):
            if any(values[i] != values[g[i]] for i in range(n)):
                raise ValueError("Product symmetry changes an ITS attribute")
        moved = {(*sorted((g[i], g[j])), order) for i, j, order in endpoint.bonds}
        if moved != original:
            raise ValueError("Product symmetry changes a bond")
    if len(group) > 1:
        generator = group[1]
        for i, g in enumerate(group):
            if tuple(generator[j] for j in g) != group[(i+1) % len(group)]:
                raise ValueError("Product permutations do not form the declared cyclic group")
    return group


def classify(r, p, maps, target, *, seconds=30, group=None, code_function=None):
    started = time.perf_counter()
    import numpy as np
    from synkit.Chem.Mapper.graph.automorphism import bounded_automorphism_permutations
    from synkit.Chem.Mapper.exact.symmetry import largest_cyclic_subgroup
    from synkit.Chem.Mapper.identifiability import extract_label
    from synkit.Chem.Mapper.spectrum import _attributed_its_graph, _ExactCodeCache

    deadline = started+seconds
    mappings = set(map(tuple, maps))
    if len(mappings) != len(maps):
        raise ValueError("Duplicate indexed input map")
    n = len(r.atomic_numbers)
    discovery_complete = None
    before_group = time.perf_counter()
    if not mappings:
        group = (tuple(range(n)),)
    elif group is None and time.perf_counter() >= deadline:
        group = (tuple(range(n)),)
    elif group is None:
        permutations, discovery_complete = bounded_automorphism_permutations(
            p.graph(), binary=False, limit=256, timeout_seconds=min(.25, max(0, deadline-time.perf_counter())),
            max_search_nodes=10000, node_properties=("charges", "hcounts"))
        group = largest_cyclic_subgroup(permutations)
    group = verify_cyclic_group(p, group)
    if len(mappings) % len(group):
        raise ValueError("Indexed count is not divisible by the verified subgroup order")
    orbit_count = len(mappings)//len(group)
    group_seconds = time.perf_counter()-before_group
    matrices = [np.zeros((n, n)), np.zeros((n, n))]
    for endpoint, matrix in zip((r, p), matrices):
        for i, j, order in endpoint.bonds:
            matrix[i, j] = matrix[j, i] = order/2
    properties = {"charges": (r.charges, p.charges), "hcounts": (r.hcounts, p.hcounts)}
    cache = _ExactCodeCache(max_entries=256)
    if code_function is None:
        def code_function(mapping, remaining):
            images = np.asarray(mapping, dtype=int)
            graph = _attributed_its_graph(matrices[0], matrices[1][images[:, None], images[None, :]],
                                          r.atomic_numbers, properties, mapping)
            return cache.code(graph, timeout_seconds=min(1, remaining), max_search_nodes=1000000)
    covered, codes, bond_patterns, joint_patterns = set(), {}, Counter(), Counter()
    representatives, failures, coded_orbits = [], [], 0
    orbit_seconds = label_seconds = canonical_seconds = 0.0
    for mapping in sorted(mappings):
        if mapping in covered:
            continue
        if time.perf_counter() >= deadline:
            break
        before = time.perf_counter()
        members = {tuple(g[j] for j in mapping) for g in group}
        if len(members) != len(group) or not members <= mappings or members & covered:
            raise ValueError("Saved mapping set is not partitioned by the verified subgroup")
        covered.update(members)
        orbit_seconds += time.perf_counter()-before
        before = time.perf_counter()
        label = extract_label(r, p, mapping)
        if 2*label.weighted_distance != target:
            raise ValueError("Representative differs from the requested CD")
        bond_key = repr(tuple(sorted(label.changed_bonds)))
        joint_key = repr((label.typed_bond_edits, label.unary_changes))
        bond_patterns[bond_key] += len(group)
        joint_patterns[joint_key] += len(group)
        label_seconds += time.perf_counter()-before
        before = time.perf_counter()
        remaining = deadline-before
        code, reason = code_function(mapping, remaining) if remaining > 0 else (None, "classification_time_limit")
        canonical_seconds += time.perf_counter()-before
        if code is not None and reason is None:
            key = repr(code)
            if key not in codes:
                codes[key] = len(codes)
            cid = codes[key]
            coded_orbits += 1
        else:
            cid = None
            failures.append({"representative": mapping, "reason": reason})
        representatives.append({"mapping": mapping, "its_class": cid, "multiplicity": len(group),
                                "bond_pattern": bond_key, "joint_pattern": joint_key})
    partition_complete = len(covered) == len(mappings)
    complete = partition_complete and not failures
    lower = max(int(bool(mappings)), len(codes))
    upper = min(orbit_count, len(codes)+(orbit_count-coded_orbits))
    return {"complete": complete, "classification_complete": complete,
            "orbit_partition_complete": partition_complete,
            "termination": "complete" if complete else "classification_time_limit" if not partition_complete else "canonicalization_incomplete",
            "indexed_maps": len(mappings), "product_orbits": orbit_count,
            "its_classes": len(codes) if complete else None,
            "its_class_lower_bound": lower, "its_class_upper_bound": upper,
            "observed_canonical_classes": len(codes), "classified_orbits": coded_orbits,
            "processed_orbits": len(representatives), "processed_indexed_maps": len(covered),
            "bond_pattern_count": len(bond_patterns) if partition_complete else None,
            "joint_pattern_count": len(joint_patterns) if partition_complete else None,
            "bond_pattern_lower_bound": len(bond_patterns), "joint_pattern_lower_bound": len(joint_patterns),
            "group_permutations": group, "group_order": len(group),
            "group_scope": "Verified cyclic product subgroup, not claimed to be the full group or the earlier search subgroup",
            "symmetry_discovery_finished": discovery_complete,
            "group_seconds": group_seconds, "orbit_seconds": orbit_seconds,
            "label_seconds": label_seconds, "canonical_seconds": canonical_seconds,
            "elapsed_seconds": time.perf_counter()-started, "cache_statistics": cache.statistics(),
            "class_codes": [{"class_id": cid, "canonical_code": code} for code, cid in codes.items()],
            "representatives": representatives, "failures": failures,
            "bond_pattern_frequencies": dict(bond_patterns), "joint_pattern_frequencies": dict(joint_patterns)}


def perform(task):
    started = time.perf_counter()
    from synkit.Chem.Mapper.identifiability import parse_reaction
    data = Path(task["map_path"]).read_bytes()
    if sha256(data).hexdigest() != task["mapping_sha256"]:
        raise ValueError("Upstream mapping file changed")
    maps = json.loads(data)
    if len(maps) != task["mapping_count"]:
        raise ValueError("Upstream mapping count changed")
    r, p = parse_reaction(task["reaction"])
    parsed = time.perf_counter()
    result = classify(r, p, maps, task["target_doubled_cd"],
                      seconds=max(0, task["seconds"]-(parsed-started)))
    exporting = time.perf_counter()
    details = {key: result.pop(key) for key in
               ("class_codes", "representatives", "failures", "bond_pattern_frequencies", "joint_pattern_frequencies")}
    encoded = (json.dumps(details, indent=2, allow_nan=False)+"\n").encode()
    with Path(task["detail_path"]).open("xb") as stream:
        stream.write(encoded)
    result.update(detail_sha256=sha256(encoded).hexdigest(), output_bytes=len(encoded),
                  input_seconds=parsed-started, export_seconds=time.perf_counter()-exporting,
                  end_to_end_seconds=time.perf_counter()-started)
    return result


if __name__ == "__main__":
    task = json.load(sys.stdin)
    limit = int(task["memory_gib"]*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    try:
        result = perform(task)
    except Exception as exc:
        result = {"complete": False, "classification_complete": False, "termination": "worker_error",
                  "error_type": type(exc).__name__, "error": str(exc)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(worker_seconds=time.perf_counter()-started,
                  cpu_seconds=usage.ru_utime+usage.ru_stime, peak_rss_kib=usage.ru_maxrss)
    print(json.dumps(result, allow_nan=False))
