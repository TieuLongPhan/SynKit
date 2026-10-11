"""Compare endpoint-overlap search with independently constructed common worlds."""

import argparse
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph  # noqa: E402
from synkit.Rule.Compose import enumerate_overlaps  # noqa: E402

SCHEMA = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))


def catalog(size):
    pairs = tuple(combinations(range(size), 2))
    for states in product((0, 1), repeat=size):
        for bonds in product((None, 0, 1), repeat=len(pairs)):
            graph = nx.Graph()
            for node, state in enumerate(states):
                graph.add_node(node, kind="X", x=state)
            for pair, bond in zip(pairs, bonds):
                if bond is not None:
                    graph.add_edge(*pair, b=bond)
            yield graph


def partial_bijections(first, second):
    # Enumerate complete subsets and permutations without the search recursion.
    for size in range(min(len(first), len(second)) + 1):
        for domain in combinations(first, size):
            for targets in permutations(second, size):
                yield dict(zip(domain, targets))


def common_world(first, second, mapping):
    # Construct the atom quotient and insert every endpoint coordinate and bond.
    # A clash on a quotient owner rejects the whole common-world construction.
    inverse = {target: source for source, target in mapping.items()}
    union = nx.Graph()
    images = (
        {node: ("first", node) for node in first},
        {
            node: ("first", inverse[node]) if node in inverse else ("second", node)
            for node in second
        },
    )
    for graph, image in zip((first, second), images):
        for node, labels in graph.nodes(data=True):
            owner = image[node]
            if owner in union and dict(union.nodes[owner]) != dict(labels):
                return None
            union.add_node(owner, **labels)
        for left, right, labels in graph.edges(data=True):
            pair = (image[left], image[right])
            if union.has_edge(*pair) and dict(union.edges[pair]) != dict(labels):
                return None
            union.add_edge(*pair, **labels)
    # Check the union embeddings directly, including exact atom intersection.
    assert set(images[0].values()) & set(images[1].values()) == {
        ("first", node) for node in mapping
    }
    for graph, image in zip((first, second), images):
        assert len(set(image.values())) == len(graph)
        for node, labels in graph.nodes(data=True):
            assert dict(union.nodes[image[node]]) == dict(labels)
        for left, right, labels in graph.edges(data=True):
            assert dict(union.edges[image[left], image[right]]) == dict(labels)
    return union


def check_pair(first, second):
    expected = set()
    examined = 0
    for mapping in partial_bijections(first, second):
        examined += 1
        if common_world(first, second, mapping) is not None:
            expected.add(tuple(sorted(mapping.items())))
    endpoints = tuple(
        LewisLabelledGraph.from_networkx(g, SCHEMA) for g in (first, second)
    )
    overlaps, _ = enumerate_overlaps(*endpoints)
    actual = set()
    for overlap in overlaps:
        mapping = {
            overlap.first_arm.mapping[node]: overlap.second_arm.mapping[node]
            for node in overlap.interface.node_ids
        }
        actual.add(tuple(sorted(mapping.items())))
        # The actual intersection contains every bond present in both patterns.
        common = {
            frozenset(pair)
            for pair in first.edges
            if set(pair) <= set(mapping)
            and second.has_edge(*(mapping[n] for n in pair))
        }
        supplied = {
            frozenset(overlap.first_arm.mapping[n] for n in edge)
            for edge in overlap.interface.edge_keys
        }
        assert supplied == common
        assert not overlap.interface.partial
    assert len(actual) == len(overlaps), "Duplicate concrete overlap mapping"
    assert actual == expected, {
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
    }
    return examined, len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    small = [graph for size in range(3) for graph in catalog(size)]
    large = list(catalog(3))
    probes = [g for g in catalog(2) if g.nodes[0]["x"] == 0]
    panels = (
        ("exhaustive_through_two_atoms", product(small, repeat=2)),
        ("three_atom_to_two_atom_probes", product(large, probes)),
        ("two_atom_to_three_atom_probes", product(probes, large)),
    )
    results = {}
    for name, pairs in panels:
        graph_pairs = examined = admitted = 0
        for first, second in pairs:
            count, accepted = check_pair(first, second)
            graph_pairs += 1
            examined += count
            admitted += accepted
        results[name] = {
            "graph_pairs": graph_pairs,
            "partial_bijections": examined,
            "common_worlds": admitted,
        }
        print(name, json.dumps(results[name]), flush=True)
    report = {
        "schema": "synkit.endpoint-overlap-common-world-audit/1",
        "status": "PASS",
        "catalog": {
            "through_two_atoms": len(small),
            "three_atoms": len(large),
            "two_atom_probes": len(probes),
        },
        "panels": results,
        "summary": {
            key: sum(row[key] for row in results.values())
            for key in next(iter(results.values()))
        },
        "all_concrete_mapping_sets_equal": True,
        "all_shared_bonds_in_intersection": True,
        "sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "Experiment/MTG/overlap_union_audit.py",
                "synkit/Rule/Compose/search.py",
                "synkit/Rule/Compose/composition.py",
                "synkit/Graph/Morphism/llg.py",
            )
        },
        "limits": (
            "Endpoint geometry only; not sequential rule admission, "
            "guard synthesis, copy inference or physical feasibility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
