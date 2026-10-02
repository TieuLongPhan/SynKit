"""Literal heavy-bijection oracle for the declared combined pendant-H model."""
import argparse
from collections import Counter
from itertools import permutations, product
from math import factorial, prod
from pathlib import Path
import json

from Experiment.Synister.audit_development import label, sha
from Experiment.Synister.confirmation_contract import source_contents
from Experiment.Synister.development import digest, encoded, save
from Experiment.Synister.freeze_environment import closure, ROOTS
from synkit.Chem.Mapper.identifiability import parse_reaction


def compatible_count(endpoint):
    return prod(factorial(n) for n in Counter(endpoint.atomic_numbers).values())


def mappings(r, p):
    elements = sorted(set(r.atomic_numbers))
    source = {z: [i for i, x in enumerate(r.atomic_numbers) if x == z] for z in elements}
    target = {z: [i for i, x in enumerate(p.atomic_numbers) if x == z] for z in elements}
    if Counter(r.atomic_numbers) != Counter(p.atomic_numbers):
        raise ValueError("Unbalanced heavy inventory")
    for choices in product(*(permutations(target[z]) for z in elements)):
        mapping = [0] * len(r.atomic_numbers)
        for z, targets in zip(elements, choices):
            for i, j in zip(source[z], targets):
                mapping[i] = j
        yield tuple(mapping)


def literal_h_cost(rh, ph, mapping):
    """Literal labeled H assignment, only for tiny test controls (<=8 H)."""
    if sum(rh) != sum(ph) or sum(rh) > 8:
        raise ValueError("Tiny balanced H control required")
    source = [i for i, n in enumerate(rh) for _ in range(n)]
    target = [j for j, n in enumerate(ph) for _ in range(n)]
    return min(2 * sum(mapping[i] != target[j] for i, j in zip(source, perm))
               for perm in permutations(range(len(target))))


def oracle(r, p):
    count = compatible_count(r)
    if count > 100000 or sum(r.hcounts) != sum(p.hcounts):
        raise ValueError("Outside prespecified oracle domain")
    rb = {(i, j): w for i, j, w in r.bonds}
    pb = {(i, j): w for i, j, w in p.bonds}
    minima = {"heavy": None, "combined": None}
    optimizers = {name: [] for name in minima}
    tested = 0
    for mapping in mappings(r, p):
        heavy = sum(abs(rb.get((i, j), 0) - pb.get(tuple(sorted((mapping[i], mapping[j]))), 0))
                    for i in range(len(mapping)) for j in range(i+1, len(mapping)))
        hydrogen = sum(abs(r.hcounts[i]-p.hcounts[mapping[i]]) for i in range(len(mapping)))
        costs = {"heavy": heavy, "combined": heavy+2*hydrogen}
        for name, value in costs.items():
            if minima[name] is None or value < minima[name]:
                minima[name], optimizers[name] = value, []
            if value == minima[name]:
                optimizers[name].append(mapping)
        tested += 1
    assert tested == count
    labels = {name: {encoded(label(r, p, m)).decode() for m in maps}
              for name, maps in optimizers.items()}
    bonds = {name: {tuple(tuple(e[:2]) for e in json.loads(x)["typed_bond_edits"])
                   for x in values} for name, values in labels.items()}
    return {"status": "complete", "compatible_maps_tested": tested,
            "minimum_doubled_cost": minima,
            "optimizers": {k: [list(m) for m in v] for k, v in optimizers.items()},
            "joint_labels": {k: [json.loads(s) for s in sorted(v)] for k, v in labels.items()},
            "joint_label_sets_equal": labels["heavy"] == labels["combined"],
            "bond_label_sets_equal": bonds["heavy"] == bonds["combined"],
            "optimizer_sets_equal": set(optimizers["heavy"]) == set(optimizers["combined"])}


def run(selection, protocol, output):
    assert sha(protocol) == "7e3bd15154ce6eabba80277a32fb5875bf72821bcf0831d86691e0453c00f9be"
    assert sha(selection) == "d01c56e6eaab47232cdb2feefe45fac20c894683b739a66b2d8467e1423bf363"
    rows = json.loads(selection.read_text())
    assert len(rows) == 1000
    accounting, eligible = [], []
    for row in rows:
        r, p = parse_reaction(row["reaction"])
        count = compatible_count(r)
        balanced = sum(r.hcounts) == sum(p.hcounts)
        ok = balanced and count <= 100000
        accounting.append({"original_id": row["original_id"], "compatible_maps": count,
                           "balanced_h": balanced, "eligible": ok})
        if ok:
            eligible.append(row)
    rank = lambda row: (digest(("synister-r1-hydrogen-v1\0"+row["original_id"]).encode()), row["original_id"])
    selected = sorted(eligible, key=rank)[:30]
    output.mkdir(parents=True, exist_ok=False)
    save(output / "accounting.json", accounting)
    save(output / "selected.json", selected)
    save(output / "all_sources.json", source_contents())
    source_hash = sha(output / "all_sources.json")
    save(output / "manifest.json", {"scope": "R1 finite exhaustive combined-pendant-H oracle",
         "selection_sha256": sha(selection), "protocol_sha256": sha(protocol),
         "accounting_sha256": sha(output / "accounting.json"), "selected_sha256": sha(output / "selected.json"),
         "all_sources_sha256": source_hash, "packages": closure(ROOTS),
         "eligible": len(eligible), "selected": len(selected),
         "conventions": "All retained H are indistinguishable ordinary singly bound pendant H. No H-H, bridging, isotopic or reservoir terms. Charges affect labels, not the distance objective or element-only compatibility. Costs use doubled integer bond orders; combined=heavy+2*sum_abs_parent_H_difference. All compatible heavy maps are enumerated, not only heavy minima."})
    results = [{"original_id": row["original_id"], "reaction": row["reaction"],
                **oracle(*parse_reaction(row["reaction"]))} for row in selected]
    assert digest(encoded(source_contents())) == source_hash
    save(output / "results.json", results)
    summary = {"attempts": len(results), "complete": len(results),
               "total_heavy_maps": sum(x["compatible_maps_tested"] for x in results),
               "different_optimizer_sets": sum(not x["optimizer_sets_equal"] for x in results),
               "different_joint_label_sets": sum(not x["joint_label_sets_equal"] for x in results),
               "different_bond_label_sets": sum(not x["bond_label_sets_equal"] for x in results)}
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("selection", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.selection, args.protocol, args.output)
