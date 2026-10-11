"""Independently compare complete cut tables and a diamond additivity criterion."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "Experiment/MTG/observer_additivity_protocol.json"
SOURCE = ROOT / "Experiment/MTG/evidence/component_observation.json"
OUTPUT = ROOT / "Experiment/MTG/evidence/observer_additivity.json"


def orders(size):
    """Enumerate distinct closures without using the source order implementation."""
    pairs = list(combinations(range(size), 2))
    seen = set()
    for mask in range(1 << len(pairs)):
        reach = {(a, b) for k, (a, b) in enumerate(pairs) if mask & (1 << k)}
        for middle in range(size):
            reach |= {(a, b) for a in range(size) for b in range(size) if (a, middle) in reach and (middle, b) in reach}
        key = frozenset(reach)
        if key not in seen:
            seen.add(key)
            yield key


def ideal_domain(size, relation):
    predecessors = [sum(1 << a for a, b in relation if b == e) for e in range(size)]
    ideals = [
        mask
        for mask in range(1 << size)
        if all(not mask & (1 << e) or mask & predecessors[e] == predecessors[e] for e in range(size))
    ]
    index = {mask: k for k, mask in enumerate(ideals)}
    diamonds = []
    for mask in ideals:
        enabled = [e for e in range(size) if not mask & (1 << e) and mask & predecessors[e] == predecessors[e]]
        for a, b in combinations(enabled, 2):
            diamonds.append(
                tuple(index[m] for m in (mask, mask | (1 << a), mask | (1 << b), mask | (1 << a) | (1 << b)))
            )
    weight_pairs = [(index[p | (1 << e)], index[p]) for e, p in enumerate(predecessors)]
    return ideals, diamonds, weight_pairs


def classify(values, ideals, diamonds, weight_pairs):
    weights = [values[upper] - values[lower] for upper, lower in weight_pairs]
    fixed = all(
        value == values[0] + sum(weight for e, weight in enumerate(weights) if mask & (1 << e))
        for mask, value in zip(ideals, values)
    )
    cancelled = all(
        values[joint] - values[left] - values[right] + values[base] == 0 for base, left, right, joint in diamonds
    )
    assert fixed == cancelled, (ideals, values, weights)
    return fixed, weights


def table_panel(states, event_names, relation):
    size = len(event_names)
    bits = {name: 1 << k for k, name in enumerate(event_names)}
    table = {sum(bits[e] for e in row["ideal"]): row["inventory"] for row in states}
    ideals, diamonds, weight_pairs = ideal_domain(size, relation)
    assert set(table) == set(ideals), "the table must cover the entire ideal domain"
    answer = {}
    for observation in ("total", "labelled"):
        values = tuple(table[mask][observation] for mask in ideals)
        additive, weights = classify(values, ideals, diamonds, weight_pairs)
        witness = next(
            (
                {
                    "ideals": [ideals[k] for k in face],
                    "values": [values[k] for k in face],
                    "difference": values[face[3]] - values[face[1]] - values[face[2]] + values[face[0]],
                }
                for face in diamonds
                if values[face[3]] - values[face[1]] - values[face[2]] + values[face[0]]
            ),
            None,
        )
        answer[observation] = {
            "fixed_ledger": additive,
            "weights": dict(zip(event_names, weights)),
            "nonzero_diamond": witness,
        }
    return {"events": size, "cuts": len(ideals), "diamonds": len(diamonds), "observations": answer}


def run():
    protocol = json.loads(PROTOCOL.read_text())
    assert not protocol["new_chemical_execution"] and not protocol["new_timing_claim"]
    rows = []
    for size in range(5):
        row = {
            "events": size,
            "posets": 0,
            "binary_tables": 0,
            "binary_additive": 0,
            "ternary_tables": 0,
            "ternary_additive": 0,
        }
        for relation in orders(size):
            row["posets"] += 1
            ideals, diamonds, weight_pairs = ideal_domain(size, relation)
            for width, kind in ((2, "binary"), (3, "ternary")):
                if width == 3 and size == 4:
                    continue
                for values in product(range(width), repeat=len(ideals)):
                    additive, _ = classify(values, ideals, diamonds, weight_pairs)
                    row[kind + "_tables"] += 1
                    row[kind + "_additive"] += int(additive)
        rows.append(row)
    original = json.loads(SOURCE.read_text())
    assert original["status"] == "PASS"
    synthetic = table_panel(original["synthetic"]["states"], [f"add{k}" for k in range(6)], set())
    assert not synthetic["observations"]["labelled"]["fixed_ledger"]
    phosphate = []
    names = ["copy0:capture", "copy0:beta", "copy0:gamma"]
    for case in original["chemical"]:
        panel = table_panel(case["results"]["states"], names, {(0, 1), (0, 2)})
        assert all(v["fixed_ledger"] for v in panel["observations"].values())
        phosphate.append({"retention_control": case["retention_control"], "labels": case["labels"], **panel})
    result = {
        "schema": "synkit.mtg-observer-additivity-audit/1",
        "status": "PASS",
        "finite_tables": rows,
        "summary": {
            "distinct_posets": sum(r["posets"] for r in rows),
            "binary_tables": sum(r["binary_tables"] for r in rows),
            "ternary_tables": sum(r["ternary_tables"] for r in rows),
            "source_table_cuts": synthetic["cuts"] + sum(p["cuts"] for p in phosphate),
        },
        "synthetic": synthetic,
        "phosphate": phosphate,
        "prior_source_executions_reused": True,
        "new_source_execution": False,
        "new_measured_chemical_case": False,
        "novelty_established": False,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), PROTOCOL, SOURCE)
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **result["summary"], "report": str(OUTPUT)}))


if __name__ == "__main__":
    run()
