"""Independent complete mapping sets at every attainable chemical distance.

The oracle uses literal compatible permutations and integer doubled bond orders;
it does not import the search's cost, bounds, symmetry or canonicalization code.
Synthetic inputs are graph controls, not valence-validated chemical reactions.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations, permutations, product
from importlib.metadata import version
import json
from math import factorial, prod
from pathlib import Path
import random
import sys
import time

from synkit.Chem.Mapper.identifiability import Endpoint, parse_reaction
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from Experiment.Synister.worked_oracle import REACTION


SEED = 20260920


def compatible_count(reactant, product_endpoint):
    counts = Counter(reactant.atomic_numbers)
    if counts != Counter(product_endpoint.atomic_numbers):
        return 0
    return prod(factorial(n) for n in counts.values())


def literal_sets(reactant, product_endpoint, fixed=None):
    """Return doubled-CD -> actual indexed mapping set, without search pruning."""
    if not compatible_count(reactant, product_endpoint):
        return {}
    fixed = fixed or {}
    sources, targets = defaultdict(list), defaultdict(list)
    for i, element in enumerate(reactant.atomic_numbers):
        sources[element].append(i)
    for i, element in enumerate(product_endpoint.atomic_numbers):
        targets[element].append(i)
    elements = sorted(sources)
    rb = {(i, j): order for i, j, order in reactant.bonds}
    pb = {(i, j): order for i, j, order in product_endpoint.bonds}
    pairs = tuple(combinations(range(len(reactant.atomic_numbers)), 2))
    result = defaultdict(set)
    for choices in product(*(permutations(targets[z]) for z in elements)):
        mapping = [0] * len(reactant.atomic_numbers)
        for z, images in zip(elements, choices):
            for i, p in zip(sources[z], images):
                mapping[i] = p
        if any(mapping[i] != p for i, p in fixed.items()):
            continue
        distance = sum(abs(rb.get((i, j), 0) -
                           pb.get(tuple(sorted((mapping[i], mapping[j]))), 0))
                       for i, j in pairs)
        result[distance].add(tuple(mapping))
    return dict(result)


def map_digest(mappings):
    return sha256(json.dumps(sorted(mappings), separators=(",", ":")).encode()).hexdigest()


def binary_endpoint(mask, n=4):
    return Endpoint((6,) * n, (0,) * n, (0,) * n,
                    tuple((i, j, 2) for bit, (i, j) in
                          enumerate(combinations(range(n), 2)) if mask & (1 << bit)))


def weighted_cases(count, seed=SEED):
    rng = random.Random(seed)
    for case in range(count):
        n = rng.randrange(4, 9)
        elements = [rng.choice((6, 6, 7, 8)) for _ in range(n)]
        def endpoint():
            colors = elements.copy()
            rng.shuffle(colors)
            return Endpoint(tuple(colors), tuple(rng.choice((-1, 0, 0, 1)) for _ in colors),
                            tuple(rng.randrange(4) for _ in colors),
                            tuple((i, j, rng.choice((2, 3, 4, 6)))
                                  for i, j in combinations(range(n), 2)
                                  if rng.random() < 0.35))
        yield f"weighted_{case:04d}", endpoint(), endpoint()


def targets_for(oracle, binary=False):
    if binary:
        # All possible distances for four binary vertices, plus impossible 7.
        return list(range(0, 15, 2))
    maximum = max(oracle)
    empty = [x for x in range(maximum + 1) if x not in oracle]
    return sorted(set(oracle) | set(empty[:3]) | {maximum + 1, maximum + 2})


def check_case(case_id, reactant, product_endpoint, *, binary=False, fixed=None,
               enumerator=enumerate_distance_mappings):
    started = time.perf_counter()
    oracle = literal_sets(reactant, product_endpoint, fixed)
    if not oracle:
        raise ValueError("This experiment requires at least one compatible mapping")
    space = sum(map(len, oracle.values()))
    if fixed is None and space != compatible_count(reactant, product_endpoint):
        raise AssertionError("Independent permutation partition is incomplete")
    optimum = min(oracle)
    graphs = [reactant.graph(), product_endpoint.graph()]
    queries = []
    for target in ["minimal", *targets_for(oracle, binary)]:
        expected = oracle.get(optimum if target == "minimal" else target, set())
        # The public API explicitly rejects expansion with fixed correspondences.
        # Test the supported raw conditioned query; do not call it a symmetry test.
        for expand in ((False,) if fixed else (False, True)):
            kwargs = dict(CD="minimal" if target == "minimal" else target / 2,
                          binary=binary, max_bijections=None, tolerance=0,
                          symmetry_pruning=expand, expand_symmetry=expand,
                          symmetry_node_properties=("charges", "hcounts"),
                          compute_minimum_cost=target == "minimal", fixed_mapping=fixed)
            before = time.perf_counter()
            result = enumerator(graphs, **kwargs)
            wall = time.perf_counter() - before
            observed = {tuple(m) for m in result.mappings}
            missing, extra = expected - observed, observed - expected
            duplicate_count = len(result.mappings) - len(observed)
            minimum_ok = (target != "minimal" or result.minimum_cost == optimum / 2)
            passed = (result.complete and not missing and not extra and not duplicate_count
                      and minimum_ok and result.selected_mapping_count == len(expected))
            queries.append({
                "target_doubled_cd": target, "symmetry_expansion": expand,
                "complete": result.complete, "status": result.status,
                "expected_count": len(expected), "observed_count": len(observed),
                "missing_count": len(missing), "extra_count": len(extra),
                "duplicate_count": duplicate_count, "minimum_ok": minimum_ok,
                "expected_sha256": map_digest(expected), "observed_sha256": map_digest(observed),
                "missing_examples": sorted(missing)[:10], "extra_examples": sorted(extra)[:10],
                "wall_seconds": wall, "visited_nodes": result.visited_nodes,
                "symmetry_group_order": result.symmetry_group_order, "passed": bool(passed),
            })
    return {"case_id": case_id, "reactant": asdict(reactant), "product": asdict(product_endpoint),
            "binary": binary, "fixed_mapping": fixed, "compatible_maps": space,
            "symmetry_expansion_tested": not bool(fixed),
            "minimum_doubled_cd": optimum,
            "histogram": {str(k): len(v) for k, v in sorted(oracle.items())},
            "queries": queries, "passed": all(q["passed"] for q in queries),
            "elapsed_seconds": time.perf_counter() - started}


def source_hashes():
    paths = [Path(__file__), Path("Experiment/Synister/worked_oracle.py")]
    paths.extend(sorted(Path("synkit/Chem/Mapper").rglob("*.py")))
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def select_real_cases(paths, count=20, max_maps=100_000):
    """Selection uses inputs only, never results or previous solver completion."""
    candidates, rejected, seen = [], [], set()
    for path in paths:
        for item in json.loads(path.read_text()):
            reaction = item["reaction"]
            key = sha256(reaction.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            try:
                r, p = parse_reaction(reaction)
            except ValueError as exc:
                rejected.append({"source": str(path), "case_id": item["case_id"], "reason": str(exc)})
                continue
            size = compatible_count(r, p)
            if 0 < size <= max_maps:
                candidates.append({"source": str(path), "case_id": item["case_id"],
                                   "reaction_id": item.get("reaction_id"), "reaction": reaction,
                                   "reaction_sha256": key, "compatible_maps": size})
    candidates.sort(key=lambda x: x["reaction_sha256"])
    return {"rule": "Unique reaction strings; compatible maps <= threshold; ascending reaction SHA256",
            "requested": count, "max_compatible_maps": max_maps,
            "unique_inputs_examined": len(seen), "eligible": len(candidates),
            "selected": candidates[:count], "unsupported": rejected,
            "input_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}}


def run(output, *, binary_pairs=4096, weighted=128, worked=True, real_inputs=(), real_count=20):
    """Create new records only. No automatic restart or historical overwrite."""
    output.mkdir(parents=True, exist_ok=False)
    selection = select_real_cases(real_inputs, real_count)
    (output / "real_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    sources = source_hashes()
    (output / "sources.json").write_text(json.dumps({p: Path(p).read_text() for p in sources}, indent=2) + "\n")
    manifest = {"schema": "synister.all-distance-oracle.v1", "seed": SEED,
                "binary_pairs_requested": binary_pairs, "weighted_pairs_requested": weighted,
                "worked_reaction_requested": worked, "python": sys.version,
                "source_sha256": sources,
                "dependencies": {name: version(name) for name in ("numpy", "scipy", "networkx", "rdkit")},
                "real_selection_sha256": sha256((output / "real_selection.json").read_bytes()).hexdigest(),
                "scope": "Complete-set checks; not a matched speed or structural-classification benchmark"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    def cases():
        for index in range(binary_pairs):
            a, b = divmod(index, 64)
            yield f"binary_{a:02d}_{b:02d}", binary_endpoint(a), binary_endpoint(b), True
        for case_id, r, p in weighted_cases(weighted):
            yield case_id, r, p, False
        if worked:
            r, p = parse_reaction(REACTION)
            yield "worked_84_1", r, p, False
        for index, item in enumerate(selection["selected"]):
            r, p = parse_reaction(item["reaction"])
            yield f"real_{index:03d}", r, p, False
    started = time.perf_counter()
    summary = {"cases": 0, "queries": 0, "compatible_maps": 0, "empty_queries": 0,
               "failed_cases": [], "failed_queries": 0, "groups": {}}
    with (output / "cases.jsonl").open("x") as handle:
        for case_id, r, p, binary in cases():
            result = check_case(case_id, r, p, binary=binary)
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()
            summary["cases"] += 1
            summary["queries"] += len(result["queries"])
            summary["compatible_maps"] += result["compatible_maps"]
            summary["empty_queries"] += sum(q["expected_count"] == 0 for q in result["queries"])
            failures = sum(not q["passed"] for q in result["queries"])
            summary["failed_queries"] += failures
            if failures:
                summary["failed_cases"].append(case_id)
            group = ("binary" if binary else "worked" if case_id.startswith("worked")
                     else "real" if case_id.startswith("real") else "weighted")
            summary["groups"][group] = summary["groups"].get(group, 0) + 1
            if summary["cases"] % 64 == 0 or failures:
                print(json.dumps({"cases": summary["cases"], "queries": summary["queries"],
                                  "failed_queries": summary["failed_queries"],
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    summary.update(all_passed=summary["failed_queries"] == 0,
                   elapsed_seconds=time.perf_counter() - started,
                   records_sha256=sha256((output / "cases.jsonl").read_bytes()).hexdigest(),
                   manifest_sha256=sha256((output / "manifest.json").read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-pairs", type=int, default=4096)
    parser.add_argument("--weighted", type=int, default=128)
    parser.add_argument("--skip-worked", action="store_true")
    parser.add_argument("--real-input", type=Path, action="append", default=[])
    parser.add_argument("--real-count", type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.binary_pairs <= 4096 or args.weighted < 0 or args.real_count < 0:
        parser.error("Require 0 <= binary pairs <= 4096 and nonnegative case counts")
    result = run(args.output, binary_pairs=args.binary_pairs, weighted=args.weighted,
                 worked=not args.skip_worked, real_inputs=args.real_input, real_count=args.real_count)
    raise SystemExit(0 if result["all_passed"] else 1)
