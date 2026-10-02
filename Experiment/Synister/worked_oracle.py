"""Literal compatible-bijection oracle, independent of search pruning.

Run: python -m Experiment.Synister.worked_oracle
Emits JSON to stdout; does not alter historical manuscript evidence.
"""

import json
from collections import defaultdict
from itertools import permutations, product

from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction
from synkit.Chem.Mapper.evaluation import ExactBondEvaluator


REACTION = "CO.NCC(O)C(=O)O.O=S(Cl)Cl>>COC(=O)C(O)CN.Cl.Cl.O=S=O"


def run():
    r, p = parse_reaction(REACTION)
    ri, pi = defaultdict(list), defaultdict(list)
    for i, z in enumerate(r.atomic_numbers):
        ri[z].append(i)
    for i, z in enumerate(p.atomic_numbers):
        pi[z].append(i)
    elements = sorted(ri)
    minimum, maps, labels, tested = None, [], [], 0
    # Exact integer doubled-order arithmetic; not the enumerator's objective.
    rb = {(i, j): w for i, j, w in r.bonds}
    pb = {(i, j): w for i, j, w in p.bonds}
    for choices in product(*(permutations(pi[z]) for z in elements)):
        mapping = [None] * len(r.atomic_numbers)
        for z, targets in zip(elements, choices):
            for i, j in zip(ri[z], targets):
                mapping[i] = j
        cost = sum(abs(rb.get((i, j), 0) - pb.get(tuple(sorted((mapping[i], mapping[j]))), 0))
                   for i in range(len(mapping)) for j in range(i + 1, len(mapping)))
        tested += 1
        if minimum is None or cost < minimum:
            minimum, maps, labels = cost, [], []
        if cost == minimum:
            label = extract_label(r, p, mapping)
            assert 2 * label.weighted_distance == cost
            maps.append(mapping)
            labels.append(label)
    scorer = ExactBondEvaluator(r)
    result = {
        "schema": "synister.worked-oracle.v1",
        "status": "development_oracle_not_mapper_comparison",
        "reaction": REACTION,
        "compatible_maps_tested": tested,
        "minimum_doubled_distance": minimum,
        "minimizing_maps": maps,
        "fixed_bond_labels": len({x.changed_bonds for x in labels}),
        "bond_label_orbits": len({scorer.orbit_key(x.changed_bonds) for x in labels}),
        "reactant_automorphisms": len(scorer.automorphisms),
    }
    assert tested == 5760 and minimum == 12 and len(maps) == 8
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
