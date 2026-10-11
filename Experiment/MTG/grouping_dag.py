#!/usr/bin/env python
"""Count contiguous arrow groupings using a DAG of certified prefix states.

This prototype applies only to fixed polar arrow lists with passing ordered
replay, constant additive resource deltas, and no stereo or kinetic claim.
Interval admission reuses a private replay API; it is a research experiment,
not a new production validator or a proof of the general MTG theorems.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.synepd_construction import (  # noqa: E402
    _hydrogen_duet_excess,
    _replay,
    _states,
)
from synkit.Mechanism.adapters import group_from_legacy_epd  # noqa: E402
from synkit.Mechanism.replay import MechanismReplayer  # noqa: E402


def _signature(graph):
    keys = (
        "element",
        "isotope",
        "aromatic",
        "hcount",
        "radical",
        "lone_pairs",
        "valence_electrons",
        "charge",
    )
    atoms = tuple(
        sorted(
            (a["atom_map"], tuple(a.get(k) for k in keys))
            for _, a in graph.nodes(data=True)
        )
    )
    bonds = tuple(
        sorted(
            (
                min(graph.nodes[a]["atom_map"], graph.nodes[b]["atom_map"]),
                max(graph.nodes[a]["atom_map"], graph.nodes[b]["atom_map"]),
                e["sigma_order"],
                e["pi_order"],
            )
            for a, b, e in graph.edges(data=True)
        )
    )
    return atoms, bonds


def _paths(edges, size, allowed=None):
    allowed = set(range(size + 1)) if allowed is None else set(allowed)
    ways = [0] * (size + 1)
    minimum, maximum = [None] * (size + 1), [None] * (size + 1)
    if 0 in allowed:
        ways[0], minimum[0], maximum[0] = 1, 0, 0
    for end in range(1, size + 1):
        if end not in allowed:
            continue
        for start in range(end):
            if (start, end) in edges and ways[start]:
                ways[end] += ways[start]
                value = minimum[start] + 1
                minimum[end] = (
                    value if minimum[end] is None else min(minimum[end], value)
                )
                value = maximum[start] + 1
                maximum[end] = (
                    value if maximum[end] is None else max(maximum[end], value)
                )
    return {
        "count": ways[size],
        "minimum_groups": minimum[size],
        "maximum_groups": maximum[size],
    }


def _cutsets(edges, size):
    result = set()

    def visit(start, cuts):
        if start == size:
            result.add(cuts)
            return
        for end in range(start + 1, size + 1):
            if (start, end) in edges:
                visit(end, cuts + ((end,) if end < size else ()))

    visit(0, ())
    return result


def run(source_path, oracle_path):
    raw = source_path.read_bytes()
    source = json.loads(raw)
    oracle = json.loads(oracle_path.read_text())
    if sha256(raw).hexdigest() != oracle["source"]["sha256"]:
        raise ValueError("Source and exhaustive oracle have different checksums.")
    baseline = {r["id"]: r for r in oracle["records"]}
    if set(baseline) != {r["id"] for r in source["records"]}:
        raise ValueError("The oracle must cover the complete source population.")
    engine = MechanismReplayer()
    summary, rows = Counter(), []
    started = perf_counter()
    for index, record in enumerate(source["records"], 1):
        n = len(record["epd"])
        profile, replay = _replay(record, tuple(range(1, n)))
        if profile["status"] != "VALID":
            raise ValueError(
                f"Record {record['id']} is outside the passing-prefix domain."
            )
        states = _states(replay)
        signatures = [_signature(g) for g in states]
        edges = set()
        for start in range(n):
            for end in range(start + 1, n + 1):
                group = group_from_legacy_epd(
                    record["epd"][start:end], group_id=f"g{start}:{end}"
                )
                result, report = engine._apply_group(
                    states[start], group, step_id="interval"
                )
                summary["interval_checks"] += 1
                if report.status == "VALID":
                    if _signature(result) != signatures[end]:
                        raise ValueError(
                            f"Non-additive prefix state at {record['id']}:{start}:{end}"
                        )
                    edges.add((start, end))
                    summary["admitted_intervals"] += 1
        all_paths = _paths(edges, n)
        permitted = [
            i for i, state in enumerate(states) if not _hydrogen_duet_excess(state)
        ]
        duet_paths = _paths(edges, n, permitted)
        exhaustive = baseline[record["id"]]["contiguous_partitions"]
        if exhaustive["complete_in_declared_family"]:
            expected = {tuple(cuts) for cuts in exhaustive["valid_cuts"]}
            if _cutsets(edges, n) != expected:
                raise ValueError(f"DAG/exhaustive mismatch at record {record['id']}")
            allowed = set(permitted)
            filtered = sum(set(cuts) <= allowed for cuts in expected)
            if 0 not in allowed or n not in allowed:
                filtered = 0
            if filtered != duet_paths["count"]:
                raise ValueError(
                    f"DAG/exhaustive hydrogen-boundary query mismatch at record {record['id']}"
                )
            summary["records_verified_against_exhaustive_partition_sets"] += 1
        summary["records"] += 1
        summary["records_with_multiple_valid_groupings"] += all_paths["count"] > 1
        summary["records_with_multiple_possible_group_counts"] += (
            all_paths["minimum_groups"] != all_paths["maximum_groups"]
        )
        summary["records_with_no_hydrogen_duet_admitted_grouping"] += (
            duet_paths["count"] == 0
        )
        summary["records_with_multiple_hydrogen_duet_admitted_groupings"] += (
            duet_paths["count"] > 1
        )
        summary["total_admitted_groupings"] += all_paths["count"]
        may_violate = duet_paths["count"] < all_paths["count"]
        must_violate = duet_paths["count"] == 0
        summary["records_with_possible_hydrogen_boundary_violation"] += may_violate
        summary["records_with_unavoidable_hydrogen_boundary_violation"] += must_violate
        row = {
            "id": record["id"],
            "arrows": n,
            "valid_intervals": len(edges),
            "groupings": all_paths,
            "hydrogen_duet_groupings": duet_paths,
            "hydrogen_boundary_query": {
                "may_violate": may_violate,
                "must_violate": must_violate,
            },
        }
        if record["id"] in (1, 67, 475):
            row["edges"] = sorted(edges)
        rows.append(row)
        if index % 100 == 0:
            print(
                f"Counted {index}/{len(source['records'])} grouping families",
                file=sys.stderr,
                flush=True,
            )
    return {
        "schema": "synkit.mtg-grouping-dag-research/1",
        "source": oracle["source"],
        "environment": oracle["environment"],
        "experiment_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Fixed-order contiguous polar grouping with constant additive deltas "
            "and passing ordered prefixes; explicit source endpoint policies "
            "retained; stereo off."
        ),
        "summary": dict(sorted(summary.items())),
        "examples": [r for r in rows if r["id"] in (1, 67, 475)],
        "most_groupings": sorted(
            rows, key=lambda r: r["groupings"]["count"], reverse=True
        )[:5],
        "elapsed_seconds": perf_counter() - started,
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    report = run(args.source, args.oracle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
