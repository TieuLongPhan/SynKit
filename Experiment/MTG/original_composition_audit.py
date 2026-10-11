#!/usr/bin/env python
"""Finite combinatorial witnesses for the original MTG construction repair.

These are copy-binding and occurrence-order checks, not chemical mechanisms
or a substitute for the separate proofs in formal note 12.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from math import comb, factorial
from pathlib import Path

import networkx as nx


def binding_count(p, q, automorphisms):
    r = min(p, q)
    count = 0
    for left in itertools.combinations(range(p), r):
        for right in itertools.combinations(range(q), r):
            for ordering in itertools.permutations(right):
                pairs = tuple(zip(left, ordering))
                assert len({a for a, _ in pairs}) == r
                assert len({b for _, b in pairs}) == r
                count += sum(
                    1 for _ in itertools.product(range(automorphisms), repeat=r)
                )
    expected = comb(p, r) * comb(q, r) * factorial(r) * automorphisms**r
    assert count == expected
    return count


def ideals(graph):
    events = tuple(graph)
    return [
        set(chosen)
        for n in range(len(events) + 1)
        for chosen in itertools.combinations(events, n)
        if all(set(nx.ancestors(graph, e)) <= set(chosen) for e in chosen)
    ]


def choice(producer):
    graph = nx.DiGraph()
    graph.add_nodes_from(("f", "t", "j"))
    graph.add_edges_from((("f", "t"), (producer, "j")))
    cuts = ideals(graph)
    for e, f in itertools.product(graph, repeat=2):
        recovered = all(e in cut for cut in cuts if f in cut)
        assert recovered == (e == f or nx.has_path(graph, e, f))
    for first, second in itertools.product(cuts, repeat=2):
        assert first & second in cuts
        assert first | second in cuts
    # Same species incidence on either copy assignment; copy IDs are retained.
    materials = {
        "D": ("f", "t", "D"),
        "g_f": ("f", "j" if producer == "f" else None, "G"),
        "g_t": ("t", "j" if producer == "t" else None, "G"),
        "S": (None, "j", "S"),
        "P": ("j", None, "P"),
    }

    def active(cut):
        return {
            m
            for m, (p, c, _) in materials.items()
            if (p is None or p in cut) and (c is None or c not in cut)
        }

    final = sorted(materials[m][2] for m in active(set(graph)))
    coexist = any({"P", "D"} <= active(cut) for cut in cuts)
    return {
        "join_consumes": "g_" + producer,
        "causal_pairs": sorted(
            [list(e) for e in nx.transitive_closure_dag(graph).edges]
        ),
        "schedules": list(nx.all_topological_sorts(graph)),
        "ideals": sorted([sorted(cut) for cut in cuts], key=lambda x: (len(x), x)),
        "final_species": final,
        "product_and_D_coexist": coexist,
    }


def run():
    counts = [
        {"p": p, "q": q, "automorphisms": a, "bindings": binding_count(p, q, a)}
        for p, q, a in itertools.product(range(5), range(5), (1, 2))
    ]
    rows = [choice("f"), choice("t")]
    assert rows[0]["final_species"] == rows[1]["final_species"] == ["G", "P"]
    assert [len(r["schedules"]) for r in rows] == [2, 1]
    assert [r["product_and_D_coexist"] for r in rows] == [True, False]
    return {
        "schema": "synkit.mtg-original-composition-audit/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "binding_cases": counts,
        "copy_choices": rows,
        "summary": {"binding_counts_checked": len(counts), "copy_choices": 2},
        "claim_limits": (
            "Finite structural abstractions of the original copy-choice argument. "
            "No chemical rates, atom mapping truth or general proof by enumeration."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
