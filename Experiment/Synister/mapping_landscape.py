"""Complete small-example CD distributions with independently checked ITS classes.

Literal element-compatible permutations determine all indexed maps. Complete
product automorphisms determine their orbits. Production canonical identities
are checked against a separate paired-graph/isomorphism partition at every CD.
This is a correctness and illustration experiment, not a runtime benchmark.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np

from Experiment.Synister.all_distance_oracle import compatible_count, literal_sets, map_digest
from Experiment.Synister.enumeration_benchmark import encode
from Experiment.Synister.structural_oracle import independent_its, isomorphic
from Experiment.Synister.worked_oracle import REACTION
from synkit.Chem.Mapper.identifiability import Endpoint, parse_reaction
from synkit.Chem.Mapper.spectrum import exact_its_and_template_codes


def toy_endpoints():
    def endpoint(edges):
        return Endpoint((6,)*5, (0,)*5, (0,)*5, tuple((i, j, 2) for i, j in edges))
    return (endpoint(((0, 1), (1, 2), (2, 3), (3, 4))),
            endpoint(((0, 1), (1, 2), (1, 3), (3, 4))))


def product_automorphisms(endpoint, limit=10000):
    graph = nx.Graph()
    for i, z in enumerate(endpoint.atomic_numbers):
        graph.add_node(i, attributes=(z, endpoint.charges[i], endpoint.hcounts[i]))
    for i, j, order in endpoint.bonds:
        graph.add_edge(i, j, order=order)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, graph, node_match=lambda a, b: a["attributes"] == b["attributes"],
        edge_match=lambda a, b: a["order"] == b["order"])
    result = []
    for image in matcher.isomorphisms_iter():
        if len(result) >= limit:
            raise ValueError("Full automorphism group exceeds the illustration limit")
        result.append(tuple(image[i] for i in range(len(graph))))
    return sorted(result)


def partition_orbits(mappings, group):
    orbits = {}
    for mapping in sorted(mappings):
        members = {tuple(g[j] for j in mapping) for g in group}
        if len(members) != len(group) or not members <= mappings:
            raise ValueError("Product group action is not free or does not preserve the CD set")
        key = min(members)
        orbits.setdefault(key, sorted(members))
    if sum(map(len, orbits.values())) != len(mappings):
        raise ValueError("Product orbits do not partition indexed maps")
    return orbits


def fingerprint(graph):
    """An isomorphism invariant used only to avoid impossible comparisons.

    Equal fingerprints do not establish graph equality: every candidate match
    still receives an exact attributed-graph isomorphism test.
    """
    return tuple(sorted((graph.nodes[i]["attributes"],
                         tuple(sorted((graph.edges[i, j]["attributes"], graph.nodes[j]["attributes"])
                                      for j in graph.neighbors(i)))) for i in graph))


def analyze(name, r, p, code_function=exact_its_and_template_codes):
    if compatible_count(r, p) > 100000 or len(r.atomic_numbers) > 16:
        raise ValueError("This illustration experiment requires a small literal mapping space")
    started = time.perf_counter()
    oracle = literal_sets(r, p)
    group = product_automorphisms(p)
    n = len(r.atomic_numbers)
    matrices = [np.zeros((n, n)), np.zeros((n, n))]
    for endpoint, matrix in zip((r, p), matrices):
        for i, j, order in endpoint.bonds:
            matrix[i, j] = matrix[j, i] = order/2
    properties = {"charges": (r.charges, p.charges), "hcounts": (r.hcounts, p.hcounts)}
    rows, outputs, comparisons = [], [], 0
    for distance in range(max(oracle)+3):
        mappings = oracle.get(distance, set())
        orbits = partition_orbits(mappings, group)
        buckets, classes = defaultdict(list), []
        code_to_class, class_to_code, failures = {}, {}, []
        for representative, members in orbits.items():
            graph = independent_its(r, p, representative)
            key = fingerprint(graph)
            match = None
            for cid, candidate in buckets[key]:
                comparisons += 1
                if isomorphic(graph, candidate):
                    match = cid
                    break
            if match is None:
                match = len(classes)
                classes.append(graph)
                buckets[key].append((match, graph))
            # Check the orbit interpretation with independent attributed graphs.
            for member in members:
                if not isomorphic(graph, independent_its(r, p, member)):
                    raise ValueError("Product orbit includes a distinct attributed ITS")
            code, _, reason = code_function(*matrices, r.atomic_numbers, properties,
                                           representative, tolerance=0,
                                           timeout_seconds=5, max_search_nodes=1000000)
            code_string = repr(code) if code is not None else None
            if code is None or reason is not None:
                failures.append({"mapping": representative, "reason": reason})
            else:
                if (code_string in code_to_class and code_to_class[code_string] != match
                        or match in class_to_code and class_to_code[match] != code_string):
                    raise ValueError("Canonical identities disagree with independent full-ITS partition")
                code_to_class[code_string] = match
                class_to_code[match] = code_string
            outputs.append({"doubled_cd": distance, "representative": representative,
                            "members": members, "independent_its_class": match,
                            "canonical_code": code_string})
        rows.append({"doubled_cd": distance, "indexed_maps": len(mappings),
                     "product_orbits": len(orbits),
                     "its_classes": len(classes) if not failures else None,
                     "independent_its_classes": len(classes),
                     "classification_complete": not failures, "failures": failures,
                     "maps_sha256": map_digest(mappings)})
    return {"case_id": name, "reactant": asdict(r), "product": asdict(p),
            "compatible_maps": compatible_count(r, p), "product_group": group,
            "product_group_order": len(group), "minimum_doubled_cd": min(oracle),
            "rows": rows, "orbits": outputs,
            "all_classifications_complete": all(row["classification_complete"] for row in rows),
            "independent_class_comparisons": comparisons,
            "elapsed_seconds": time.perf_counter()-started}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    paths += sorted(Path("synkit/Graph/Canon").rglob("*.py"))
    paths += [Path(__file__), Path("Experiment/Synister/all_distance_oracle.py"),
              Path("Experiment/Synister/structural_oracle.py"), Path("Experiment/Synister/worked_oracle.py"),
              Path("Experiment/Synister/enumeration_benchmark.py")]
    (output/"sources.json").write_text(encode({str(path): path.read_text() for path in paths}))
    reports = []
    for name, (r, p) in (("synthetic_five_vertices", toy_endpoints()),
                         ("worked_84_1", parse_reaction(REACTION))):
        record = analyze(name, r, p)
        (output/f"{name}.json").write_text(encode(record))
        minimum = next(row for row in record["rows"] if row["doubled_cd"] == record["minimum_doubled_cd"])
        reports.append({"case_id": name, "compatible_maps": record["compatible_maps"],
                        "minimum": minimum, "product_group_order": record["product_group_order"],
                        "all_classifications_complete": record["all_classifications_complete"],
                        "independent_class_comparisons": record["independent_class_comparisons"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "record_sha256": sha256((output/f"{name}.json").read_bytes()).hexdigest()})
        print(encode(reports[-1]), flush=True)
    summary = {"schema": "synister.complete-small-landscape.v1", "cases": reports,
               "all_passed": all(r["all_classifications_complete"] for r in reports),
               "sources_sha256": sha256((output/"sources.json").read_bytes()).hexdigest(),
               "scope": "Complete product groups and full attributed ITS counts; not probabilities or a timing comparison"}
    (output/"summary.json").write_text(encode(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output)["all_passed"] else 1)
