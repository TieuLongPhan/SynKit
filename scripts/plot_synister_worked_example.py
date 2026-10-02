#!/usr/bin/env python3
"""Reproduce and draw the genuinely unmapped FlowER 84:1 worked example.

Normal builds verify and draw the frozen record. --recompute repeats both
Synister searches and an independent exhaustive oracle from unmapped SMILES.
--dataset additionally checks those endpoints against the original FlowER row.
No reference atom mapping is supplied to either search or the oracle.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import gzip
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synkit.Chem.Mapper.analysis import _property_vectors  # noqa: E402
from synkit.Chem.Mapper.exact.hybrid import (  # noqa: E402
    enumerate_hybrid_distance_mappings,
)  # noqa: E402
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph  # noqa: E402
from synkit.Chem.Mapper.spectrum import exact_its_and_template_codes  # noqa: E402

PAPER = ROOT / "paper/synister"
RECORD = PAPER / "evidence/worked_unmapped_flower84_v1/record.json"
SMILES = ["CO.NCC(O)C(=O)O.O=S(Cl)Cl", "COC(=O)C(O)CN.Cl.Cl.O=S=O"]
DATASET_SHA = "e40647847169a7fc98af1aab44fa81c7a64deb70b77085ae3deb3925e39642b0"
REACTION_SHA = "d8992069dde70b5894943449190df35c1acf4b6c6d87adc2e501a295e03e5f36"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def endpoints():
    graphs, matrices, inventories = [], [], []
    for smiles in SMILES:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None and not any(a.GetAtomMapNum() for a in mol.GetAtoms())
        matrix = Chem.GetAdjacencyMatrix(mol, useBO=True).astype(int)
        elements = [a.GetAtomicNum() for a in mol.GetAtoms()]
        graph = LabeledGraph(
            {
                i: {j: int(v) for j, v in enumerate(row) if v}
                for i, row in enumerate(matrix)
            },
            elements,
        )
        graph.set_prop("atomic numbers", elements)
        graph.set_prop("hcounts", [a.GetTotalNumHs() for a in mol.GetAtoms()])
        graph.set_prop("charges", [a.GetFormalCharge() for a in mol.GetAtoms()])
        graphs.append(graph)
        matrices.append(matrix)
        inventories.append(
            [
                {
                    "index": a.GetIdx(),
                    "element": a.GetSymbol(),
                    "hydrogens": a.GetTotalNumHs(),
                    "charge": a.GetFormalCharge(),
                }
                for a in mol.GetAtoms()
            ]
        )
    return graphs, matrices, inventories


def paired_graph(a, b, inventories, mapping):
    """Independent NetworkX representation: full paired atom/bond attributes."""
    g = nx.Graph()
    r, p = inventories
    for i, j in enumerate(mapping):
        g.add_node(
            i,
            state=(
                r[i]["element"],
                r[i]["hydrogens"],
                p[j]["hydrogens"],
                r[i]["charge"],
                p[j]["charge"],
            ),
        )
    for i, j in combinations(range(len(mapping)), 2):
        state = (int(a[i, j]), int(b[mapping[i], mapping[j]]))
        if state != (0, 0):
            g.add_edge(i, j, state=state)
    return g


def source_provenance(dataset):
    provenance = {
        "reaction_id": "84:1",
        "source_line": 109,
        "dataset_sha256": DATASET_SHA,
        "mapped_reaction_sha256": REACTION_SHA,
    }
    if dataset is None:
        previous = read_record()
        return previous["provenance"]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == DATASET_SHA
    with gzip.open(dataset, "rt", newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if int(r["source_line"]) == 109)
    assert row["reaction_id"] == "84:1"
    reaction = row["mapped_reaction"].split("|", 1)[0]
    assert hashlib.sha256(reaction.encode()).hexdigest() == REACTION_SHA
    unmapped = []
    for side in reaction.split(">>"):
        mol = Chem.MolFromSmiles(side)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        unmapped.append(Chem.MolToSmiles(mol))
    assert unmapped == SMILES
    provenance["unmapping_verified_against_source"] = True
    return provenance


def recompute(dataset):
    provenance = source_provenance(dataset)
    graphs, (a, b), inventories = endpoints()
    n = len(a)
    buckets = defaultdict(list)
    for atom in inventories[0]:
        buckets[atom["element"]].append(atom["index"])
    p_buckets = defaultdict(list)
    for atom in inventories[1]:
        p_buckets[atom["element"]].append(atom["index"])
    histogram, best, optimal = Counter(), None, []
    # Independent literal loop: no solver bounds, seeds or symmetry pruning.
    for choices in product(*(permutations(p_buckets[e]) for e in buckets)):
        mapping = [None] * n
        for indices, images in zip(buckets.values(), choices):
            for i, j in zip(indices, images):
                mapping[i] = j
        cost = sum(
            abs(int(a[i, j]) - int(b[mapping[i], mapping[j]]))
            for i, j in combinations(range(n), 2)
        )
        histogram[cost] += 1
        if best is None or cost < best:
            best, optimal = cost, []
        if cost == best:
            optimal.append(mapping)

    options = dict(
        CD="minimal",
        binary=False,
        max_bijections=None,
        time_limit_seconds=60,
        max_mappings=10000,
        collect_mappings=True,
        compute_minimum_cost=True,
        initial_mapping=None,
        fixed_mapping=None,
        symmetry_node_properties=("hcounts", "charges"),
    )
    labeled = enumerate_hybrid_distance_mappings(
        graphs, symmetry_pruning=False, **options
    )
    quotient = enumerate_hybrid_distance_mappings(
        graphs, symmetry_pruning=True, expand_symmetry=False, **options
    )
    assert (
        labeled.complete and quotient.complete and quotient.symmetry_quotient_complete
    )
    assert labeled.minimum_cost == quotient.minimum_cost == best == 6
    assert sorted(map(tuple, labeled.mappings)) == sorted(map(tuple, optimal))

    # Independent product automorphisms and explicit action on every minimum map.
    pg = nx.Graph()
    for atom in inventories[1]:
        pg.add_node(
            atom["index"], state=(atom["element"], atom["hydrogens"], atom["charge"])
        )
    for i, j in combinations(range(n), 2):
        if b[i, j]:
            pg.add_edge(i, j, state=int(b[i, j]))

    def match(x, y):
        return x["state"] == y["state"]

    automorphisms = list(
        nx.algorithms.isomorphism.GraphMatcher(
            pg, pg, node_match=match, edge_match=match
        ).isomorphisms_iter()
    )

    def orbit_key(mapping):
        return min(tuple(h[j] for j in mapping) for h in automorphisms)

    orbits = {orbit_key(m) for m in optimal}
    assert {orbit_key(m) for m in quotient.mappings} == orbits
    assert len(quotient.mappings) == len(orbits) == 2
    assert quotient.symmetry_group_order == len(automorphisms) == 4

    properties = _property_vectors(graphs, ("hcounts", "charges"))
    classes = defaultdict(list)
    for mapping in sorted(optimal):
        code, _, reason = exact_its_and_template_codes(
            a, b, graphs[0].labels, properties, mapping, timeout_seconds=5
        )
        assert reason is None
        classes[code].append(mapping)
    # Compare every pair, including both same-class and different-class pairs.
    code_by_map = {tuple(m): code for code, maps in classes.items() for m in maps}
    for m, other in combinations(optimal, 2):
        iso = nx.is_isomorphic(
            paired_graph(a, b, inventories, m),
            paired_graph(a, b, inventories, other),
            node_match=match,
            edge_match=match,
        )
        assert iso == (code_by_map[tuple(m)] == code_by_map[tuple(other)])
    assert len(classes) == 2 and sorted(map(len, classes.values())) == [4, 4]
    assert sum(histogram.values()) == 5760
    class_records = []
    for code, maps in classes.items():
        mapping = maps[0]
        edits = [
            [i, j, int(a[i, j]), int(b[mapping[i], mapping[j]])]
            for i, j in combinations(range(n), 2)
            if a[i, j] != b[mapping[i], mapping[j]]
        ]
        class_records.append(
            {
                "its_code_sha256": hashlib.sha256(repr(code).encode()).hexdigest(),
                "labeled_maps": maps,
                "representative": mapping,
                "bond_edits": edits,
                "methanol_oxygen_destination": (
                    "ester" if mapping[1] == 1 else "sulfur_dioxide"
                ),
            }
        )
    class_records.sort(key=lambda c: c["methanol_oxygen_destination"])
    for i, cls in enumerate(class_records):
        cls["display_name"] = chr(65 + i)
    payload = {
        "schema_version": 1,
        "kind": "unmapped_real_reaction_worked_example",
        "provenance": provenance,
        "unmapped_smiles": SMILES,
        "input_atom_maps_present": False,
        "reference_seed_supplied": False,
        "objective": "heavy_atom_bond_order_cd",
        "index_base": 0,
        "endpoint_inventories": inventories,
        "endpoint_adjacency": [a.tolist(), b.tolist()],
        "search_options": options,
        "minimum_cd": best,
        "oracle": {
            "compatible_maps": sum(histogram.values()),
            "cd_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "all_minimum_maps": sorted(optimal),
            "pairwise_its_isomorphism_checks": 28,
            "product_automorphisms": [[h[i] for i in range(n)] for h in automorphisms],
        },
        "labeled_search": asdict(labeled),
        "quotient_search": asdict(quotient),
        "classes": class_records,
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "networkx_version": nx.__version__,
    }
    paths = sorted((ROOT / "synkit/Chem/Mapper").rglob("*.py"))
    paths += sorted((ROOT / "synkit/Graph/Canon").rglob("*.py")) + [Path(__file__)]
    payload["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    payload["record_sha256"] = digest(payload)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def read_record():
    payload = json.loads(RECORD.read_text())
    body = {k: v for k, v in payload.items() if k != "record_sha256"}
    if digest(body) != payload["record_sha256"]:
        raise ValueError("Worked-example record digest mismatch")
    if payload["unmapped_smiles"] != SMILES:
        raise ValueError("Worked-example input changed")
    return payload


from synister_diagrams import draw_input, draw_classes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    if args.dataset and not args.recompute:
        parser.error("--dataset requires --recompute")
    record = recompute(args.dataset) if args.recompute else read_record()
    draw_input(record)
    draw_classes(record)
    print(
        "FlowER 84:1: 5,760 maps; minimum CD 6; 8 optimal maps; 2 orbits; 2 ITS classes."
    )


if __name__ == "__main__":
    main()
