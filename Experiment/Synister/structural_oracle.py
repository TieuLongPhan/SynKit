"""Independent pairwise attributed-graph checks of full ITS class equality."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import networkx as nx
import numpy as np

from Experiment.Synister.all_distance_oracle import binary_endpoint, literal_sets, weighted_cases
from synkit.Chem.Mapper.spectrum import exact_its_and_template_codes


def independent_its(r, p, mapping):
    """Construct paired endpoints independently; no production ITS builder."""
    graph = nx.Graph()
    rb = {(i, j): w for i, j, w in r.bonds}
    pb = {(i, j): w for i, j, w in p.bonds}
    for i, image in enumerate(mapping):
        graph.add_node(i, attributes=(r.atomic_numbers[i], r.charges[i], p.charges[image],
                                      r.hcounts[i], p.hcounts[image]))
    for i, j in combinations(range(len(mapping)), 2):
        before = rb.get((i, j), 0)
        after = pb.get(tuple(sorted((mapping[i], mapping[j]))), 0)
        if before or after:
            graph.add_edge(i, j, attributes=(before, after))
    return graph


def isomorphic(a, b):
    return nx.is_isomorphic(a, b,
                            node_match=lambda x, y: x["attributes"] == y["attributes"],
                            edge_match=lambda x, y: x["attributes"] == y["attributes"])


def check_case(name, r, p, code_function=exact_its_and_template_codes):
    n = len(r.atomic_numbers)
    matrices = [np.zeros((n, n)), np.zeros((n, n))]
    for endpoint, matrix in zip((r, p), matrices):
        for i, j, order in endpoint.bonds:
            matrix[i, j] = matrix[j, i] = order / 2
    properties = {"charges": (r.charges, p.charges), "hcounts": (r.hcounts, p.hcounts)}
    results = []
    for distance, mappings in sorted(literal_sets(r, p).items()):
        mappings = sorted(mappings)
        graphs = [independent_its(r, p, m) for m in mappings]
        codes = [code_function(*matrices, r.atomic_numbers, properties, m,
                               timeout_seconds=2, max_search_nodes=1_000_000)
                 for m in mappings]
        unfinished = sum(code[0] is None or code[2] is not None for code in codes)
        differences, equal_pairs = [], 0
        for i, j in combinations(range(len(mappings)), 2):
            equal = isomorphic(graphs[i], graphs[j])
            equal_pairs += equal
            if codes[i][0] is None or codes[j][0] is None or equal != (codes[i][0] == codes[j][0]):
                differences.append({"first": mappings[i], "second": mappings[j], "isomorphic": equal})
        representatives = []
        for graph in graphs:
            if not any(isomorphic(graph, representative) for representative in representatives):
                representatives.append(graph)
        observed_count = len({repr(code[0]) for code in codes if code[0] is not None})
        results.append({"doubled_cd": distance, "maps": len(mappings),
                        "pairs_checked": len(mappings) * (len(mappings) - 1) // 2,
                        "isomorphic_pairs": equal_pairs, "independent_classes": len(representatives),
                        "production_classes": observed_count, "unfinished": unfinished,
                        "differences": differences, "passed": not unfinished and not differences
                        and observed_count == len(representatives)})
    return {"case_id": name, "reactant": asdict(r), "product": asdict(p),
            "distances": results, "passed": all(x["passed"] for x in results)}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = [(f"binary_{a}_{b}", binary_endpoint(a), binary_endpoint(b)) for a, b in
             ((0, 0), (0, 63), (1, 1), (3, 7), (7, 11), (12, 31), (30, 45), (63, 63))]
    small = [case for case in weighted_cases(128) if len(case[1].atomic_numbers) <= 5]
    cases.extend(small[:24])
    records = [check_case(*case) for case in cases]
    (output / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    paths = [Path(__file__), Path("Experiment/Synister/all_distance_oracle.py")]
    paths.extend(sorted(Path("synkit/Chem/Mapper").rglob("*.py")))
    sources = {str(path): path.read_text() for path in paths}
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    summary = {"schema": "synister.structural-oracle.v1", "cases": len(records),
               "distances": sum(len(r["distances"]) for r in records),
               "mapping_pairs": sum(d["pairs_checked"] for r in records for d in r["distances"]),
               "isomorphic_pairs": sum(d["isomorphic_pairs"] for r in records for d in r["distances"]),
               "all_passed": all(r["passed"] for r in records),
               "cases_sha256": sha256((output / "cases.json").read_bytes()).hexdigest(),
               "sources_sha256": sha256((output / "sources.json").read_bytes()).hexdigest(),
               "scope": "Full attributed ITS equality, not template equivalence or chemical plausibility"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output)["all_passed"] else 1)
