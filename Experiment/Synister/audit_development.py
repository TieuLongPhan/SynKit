"""Replay saved development labels and scores without running either mapper.

This verifies artifact consistency and attained score witnesses, not an
independent exhaustive proof of every optimal-shell closure assertion.
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import networkx as nx

from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.prediction_adapter import align_mapped_prediction


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(r, p, mapping):
    n = len(r.atomic_numbers)
    assert sorted(mapping) == list(range(n))
    assert all(r.atomic_numbers[i] == p.atomic_numbers[mapping[i]] for i in range(n))
    a = {(i, j): w for i, j, w in r.bonds}
    b = {(i, j): w for i, j, w in p.bonds}
    edits = []
    for i in range(n):
        for j in range(i + 1, n):
            before = a.get((i, j), 0)
            after = b.get(tuple(sorted((mapping[i], mapping[j]))), 0)
            if before != after:
                edits.append([i, j, before, after])
    unary = sorted([i, name, getattr(r, name)[i], getattr(p, name)[mapping[i]]]
                   for name in ("charges", "hcounts") for i in range(n)
                   if getattr(r, name)[i] != getattr(p, name)[mapping[i]])
    return {"typed_bond_edits": edits, "unary_changes": unary}


def bonds(value):
    return frozenset(tuple(edit[:2]) for edit in value["typed_bond_edits"])


def automorphisms(r):
    graph = nx.Graph()
    for i, z in enumerate(r.atomic_numbers):
        graph.add_node(i, z=z, q=r.charges[i], h=r.hcounts[i])
    for i, j, order in r.bonds:
        graph.add_edge(i, j, order=order)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, graph, node_match=lambda a, b: a == b, edge_match=lambda a, b: a == b)
    return [tuple(m[i] for i in range(len(graph))) for m in matcher.isomorphisms_iter()]


def transformed(value, permutation):
    return frozenset(tuple(sorted((permutation[i], permutation[j]))) for i, j in value)


def f1(a, b):
    return Fraction(2 * len(a & b), len(a) + len(b)) if a or b else Fraction(1)


def is_automorphism(r, mapping):
    n = len(r.atomic_numbers)
    if len(mapping) != n or sorted(mapping) != list(range(n)):
        return False
    if any(getattr(r, name)[i] != getattr(r, name)[mapping[i]]
           for name in ("atomic_numbers", "charges", "hcounts") for i in range(n)):
        return False
    before = {(i, j): w for i, j, w in r.bonds}
    after = {tuple(sorted((mapping[i], mapping[j]))): w for i, j, w in r.bonds}
    return before == after


def audit(directory):
    read = lambda name: json.loads((directory / name).read_text())
    manifest, inputs, summary = read("manifest.json"), read("inputs.json"), read("summary.json")
    assert sha(directory / "inputs.json") == manifest["inputs_sha256"]
    assert sha(directory / "source_snapshot.json") == manifest["source_snapshot_sha256"]
    freeze = read("prediction_freeze.json")
    counts = Counter()
    lows, highs, positive = [], [], []
    rows = []
    for case in inputs:
        if case["status"] != "eligible":
            counts["input_rejected"] += 1
            continue
        counts["eligible"] += 1
        key = case["case_id"]
        r, p = parse_reaction(case["reaction"])
        predictions = []
        for method in ("slap", "rxnmapper"):
            path = f"cases/{key}.{method}.json"
            record = read(path)
            assert sha(directory / path) == freeze[f"{key}.{method}"]
            counts[f"{method}:{record['status']}"] += 1
            if record["status"] == "valid":
                mapping = record["prediction"]["mapping"]
                assert label(r, p, mapping) == record["label"]
                if method == "rxnmapper":
                    aligned = align_mapped_prediction(case["reaction"], record["raw_prediction"]["mapped_rxn"])
                    assert list(aligned.mapping) == mapping
                    assert list(aligned.reactant_to_output) == record["prediction"]["reactant_to_output"]
                    assert list(aligned.product_to_output) == record["prediction"]["product_to_output"]
            predictions.append(record)
        common = all(x["status"] == "valid" for x in predictions)
        counts["common_valid_predictions"] += common
        exact = read(f"cases/{key}.exact.json")
        counts[f"exact:{exact['status']}"] += 1
        if exact.get("initial_mapping") is not None:
            seed_label = label(r, p, exact["initial_mapping"])
            seed_cost = Fraction(sum(abs(x[3]-x[2]) for x in seed_label["typed_bond_edits"]), 2)
            assert seed_cost == Fraction(exact["initial_cost"])
        if exact["status"] != "complete":
            continue
        assert exact["minimum_proved"] and exact["enumeration_complete"] and exact["labels_complete"]
        candidate_labels = []
        for candidate in exact["labels"]:
            value = label(r, p, candidate["mapping"])
            assert value == candidate["label"]
            assert Fraction(sum(abs(x[3] - x[2]) for x in value["typed_bond_edits"]), 2) == exact["minimum"]
            candidate_labels.append(bonds(value))
        assert candidate_labels and len(set(candidate_labels)) == len(candidate_labels)
        if not common:
            continue
        score = read(f"cases/{key}.score.json")
        counts[f"score:{score['status']}"] += 1
        if score["status"] != "complete":
            continue
        backend = score.get("symmetry_backend", "full-group")
        counts[f"audit_orbit_engine:{backend}"] += 1
        if backend == "support-stabilizer":
            from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator
            replay = SupportOrbitEvaluator(r, time_limit_seconds=120)
            orbits = {}
            for y in candidate_labels:
                images = replay.orbit(y)
                for image, g in images.items():
                    assert is_automorphism(r, g) and transformed(y, g) == image
                orbits[y] = set(images)
            assert score["reactant_automorphisms"] is None
        else:
            assert backend == "full-group"
            autos = automorphisms(r)
            orbits = {y: {transformed(y, g) for g in autos} for y in candidate_labels}
            assert len(autos) == score["reactant_automorphisms"]
        a, b = (bonds(x["label"]) for x in predictions)
        differences = []
        orbit_keys = set()
        for y in candidate_labels:
            orbit = orbits[y]
            orbit_keys.add(min(tuple(sorted(z)) for z in orbit))
            differences.append(max(f1(a, z) for z in orbit) - max(f1(b, z) for z in orbit))
        lo, hi = min(differences), max(differences)
        assert len(candidate_labels) == score["fixed_bond_labels"]
        assert len(orbit_keys) == score["bond_label_orbits"]
        assert Fraction(score["width"]) == hi - lo
        for name, value in (("lower", lo), ("upper", hi)):
            witness = score[name]
            assert Fraction(witness["difference"]) == value
            y = frozenset(tuple(pair) for pair in witness["label"])
            assert y in candidate_labels
            for method, prediction in (("a", a), ("b", b)):
                g = tuple(witness[f"{method}_transporter"])
                assert is_automorphism(r, g)
                observed = f1(prediction, transformed(y, g))
                assert Fraction(witness[f"{method}_score"]) == observed
                assert observed == max(f1(prediction, image) for image in orbits[y])
            assert Fraction(witness["a_score"]) - Fraction(witness["b_score"]) == value
        lows.append(lo)
        highs.append(hi)
        if hi > lo:
            positive.append(case["reaction_id"])
        rows.append({"reaction_id": case["reaction_id"], "atoms": len(r.atomic_numbers),
                     "lower": str(lo), "upper": str(hi), "width": str(hi-lo),
                     "bond_label_orbits": len(orbit_keys)})
    n, resolved = counts["common_valid_predictions"], len(lows)
    lower_sum, upper_sum = sum(lows, Fraction()), sum(highs, Fraction())
    assert summary["selected"] == len(inputs)
    assert summary["eligible"] == counts["eligible"]
    assert summary["common_valid_predictions"] == n
    assert summary["resolved_comparisons"] == resolved
    assert summary["exact_searches_closed"] == counts["exact:complete"]
    assert summary["positive_paired_width_cases"] == len(positive)
    if resolved:
        assert summary["resolved_conditional_envelope"] == [str(lower_sum/resolved), str(upper_sum/resolved)]
    if n:
        unknown = n-resolved
        assert summary["common_valid_outer_envelope"] == [str((lower_sum-unknown)/n), str((upper_sum+unknown)/n)]
        assert summary["unresolved_weight"] == str(Fraction(unknown,n))
    return {"scope": "artifact replay; solver closure not independently reproved; support backend reuses orbit engine with independent transporter checks",
            "manifest_sha256": sha(directory / "manifest.json"),
            "summary_sha256": sha(directory / "summary.json"),
            "counts": dict(counts), "positive_width_reaction_ids": positive,
            "resolved_rows": rows, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
