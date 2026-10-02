"""Independent label/transport witness checks for annotation archives.

This audit does not reprove search closure, orbit maximality, or canonical
minimality. Those distinctions are explicit in its output. It checks saved
attainment witnesses and compatibility with the earlier primary archive.
"""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from Experiment.Synister.audit_development import label, bonds, f1, is_automorphism, sha
from synkit.Chem.Mapper.identifiability import parse_reaction


METRICS = {"bond_f1": "bond", "atom_f1": "atom", "typed_f1": "typed",
           "bond_exact": "bond", "joint_exact": "joint"}


def tokens(value, kind):
    edits, unary = value["typed_bond_edits"], value["unary_changes"]
    if kind == "bond":
        return frozenset(("bond", *x[:2]) for x in edits)
    if kind == "atom":
        atoms = {i for x in edits for i in x[:2]} | {x[0] for x in unary}
        return frozenset(("atom", i) for i in atoms)
    typed = frozenset(("bond", *x) for x in edits)
    return typed if kind == "typed" else typed | frozenset(("unary", *x) for x in unary)


def transport(value, permutation):
    output = []
    for x in value:
        if x[0] == "bond":
            output.append((x[0], *sorted((permutation[x[1]], permutation[x[2]])), *x[3:]))
        else:
            output.append((x[0], permutation[x[1]], *x[2:]))
    return frozenset(output)


def check_witness(r, predictions, candidates, witness, *, exact=False):
    y = frozenset(tuple(x) for x in witness["label"])
    assert y in candidates, "Extremum label absent from exported candidates"
    scores = []
    for method, prediction in zip(("a", "b"), predictions):
        g = witness[f"{method}_transporter"]
        assert is_automorphism(r, g), "Invalid full-reactant transporter"
        image = transport(y, g)
        value = Fraction(prediction == image) if exact else f1(prediction, image)
        assert value == Fraction(witness[f"{method}_score"]), "Witness score mismatch"
        scores.append(value)
    assert scores[0] - scores[1] == Fraction(witness["difference"])


def audit(directory, parent):
    read = lambda root, name: json.loads((root / name).read_text())
    manifest = read(directory, "manifest.json")
    assert sha(parent / "manifest.json") == manifest["parent_manifest_sha256"]
    assert sha(directory / "inputs.json") == manifest["inputs_sha256"]
    assert sha(directory / "source_snapshot.json") == manifest["source_snapshot_sha256"]
    assert (directory / "inputs.json").read_bytes() == (parent / "inputs.json").read_bytes()
    freeze = read(directory, "prediction_freeze.json")
    assert freeze == read(parent, "prediction_freeze.json")
    inputs = read(directory, "inputs.json")
    summary = read(directory, "summary.json")
    counts, statuses = Counter(), {name: Counter() for name in METRICS}
    intervals = {name: [] for name in METRICS}
    structural, policies = Counter(), Counter()
    policy_values = {name: [] for name in ("canonical_its", "nearest_a", "nearest_b")}
    for case in inputs:
        if case["status"] != "eligible":
            continue
        key = case["case_id"]
        r, p = parse_reaction(case["reaction"])
        predictions = []
        for method in ("slap", "rxnmapper"):
            filename = f"cases/{key}.{method}.json"
            assert sha(directory / filename) == freeze[f"{key}.{method}"]
            record = read(directory, filename)
            assert record["status"] == "valid", "This development audit requires common-valid predictions"
            value = label(r, p, record["prediction"]["mapping"])
            assert value == record["label"]
            predictions.append(value)
        exact = read(directory, f"cases/{key}.exact.json")
        previous = read(parent, f"cases/{key}.exact.json")
        assert exact["status"] == "complete" and exact["joint_labels_complete"]
        assert exact["minimum_proved"] and exact["enumeration_complete"]
        assert exact["minimum"] == previous["minimum"]
        candidates = []
        mappings = set()
        for row in exact["joint_labels"]:
            value = label(r, p, row["mapping"])
            assert value == row["label"]
            cost = Fraction(sum(abs(x[3]-x[2]) for x in value["typed_bond_edits"]), 2)
            assert cost == exact["minimum"]
            candidates.append(value)
            mappings.add(tuple(row["mapping"]))
        assert candidates
        assert {bonds(x) for x in candidates} == {bonds(x["label"]) for x in previous["labels"]}
        record = read(directory, f"cases/{key}.annotations.json")
        counts[record["status"]] += 1
        if record["status"] != "evaluated":
            assert record["status"] in {"hard_timeout", "worker_failed", "invalid_worker_output", "error"}
            continue
        for name, kind in METRICS.items():
            metric = record["metrics"][name]
            statuses[name][metric["status"]] += 1
            if metric["status"] != "complete":
                continue
            ys = {tokens(x, kind) for x in candidates}
            assert len(ys) == metric["fixed_labels"]
            assert 1 <= metric["label_orbits"] <= len(ys)
            for end in ("lower", "upper"):
                check_witness(r, [tokens(x, kind) for x in predictions], ys,
                              metric[end], exact=name.endswith("exact"))
            lo, hi = [Fraction(metric[x]["difference"]) for x in ("lower", "upper")]
            assert lo <= hi and hi-lo == Fraction(metric["width"])
            intervals[name].append((lo, hi))
            if name == "bond_f1":
                old = read(parent, f"cases/{key}.score.json")
                if old["status"] == "complete":
                    assert [lo, hi] == [Fraction(old[x]["difference"]) for x in ("lower", "upper")]
                    assert metric["label_orbits"] == old["bond_label_orbits"]
        structure = record["structure"]
        structural[structure["status"]] += 1
        if structure["status"] == "complete":
            codes = structure["codes"]
            assert {tuple(x["mapping"]) for x in codes} == mappings
            assert structure["its_classes"] == len({x["its_code"] for x in codes})
            assert structure["template_classes"] == len({x["template_code"] for x in codes})
            for name, policy in record["policies"].items():
                assert any(tuple(x["mapping"]) == tuple(policy["mapping"]) and
                           x["its_code"] == policy["its_code"] for x in codes)
                if name == "canonical_its":
                    assert policy["its_code"] == min(x["its_code"] for x in codes)
                diff = Fraction(policy["a_score"])-Fraction(policy["b_score"])
                assert diff == Fraction(policy["difference"])
                policies[name] += 1
                policy_values[name].append(diff)
    assert summary["selected"] == len(inputs)
    assert summary["annotation_status"] == dict(counts)
    assert summary["structure_status"] == dict(structural)
    for name in METRICS:
        assert summary["metric_status"][name] == dict(statuses[name])
    n = sum(counts.values())
    reports = {}
    for name, values in intervals.items():
        k = len(values)
        lo = sum((x[0] for x in values), Fraction())
        hi = sum((x[1] for x in values), Fraction())
        reports[name] = {"resolved": k, "positive_width": sum(a < b for a, b in values),
                         "local_sign_reversal": sum(a < 0 < b for a, b in values),
                         "conditional_envelope": [str(lo/k), str(hi/k)] if k else None,
                         "outer_envelope": [str((lo-(n-k))/n), str((hi+(n-k))/n)] if n else None}
    return {"scope": "witness/artifact audit, not independent closure, orbit-maximality or canonical-minimality proof; policy scores and reference alignment not replayed",
            "analysis_scope": manifest.get("scope"), "attempts": n,
            "whole_process_status": dict(counts),
            "manifest_sha256": sha(directory / "manifest.json"),
            "summary_sha256": sha(directory / "summary.json"), "metrics": reports,
            "policy_record_counts": dict(policies),
            "policy_reported_conditional_means": {name: str(sum(v, Fraction())/len(v)) if v else None
                                                   for name, v in policy_values.items()},
            "policy_reported_outer_bounds": {
                name: [str((sum(v, Fraction())-(n-len(v)))/n),
                       str((sum(v, Fraction())+(n-len(v)))/n)] if n else None
                for name, v in policy_values.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.parent)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
