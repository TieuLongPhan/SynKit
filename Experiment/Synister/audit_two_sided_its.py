"""Literal endpoint-group action versus independent paired-graph isomorphism.

Small attributed-graph controls, not mechanistic validation or an asymptotic
performance benchmark. No production automorphism or canonicalization code is
used to construct the two-sided classes.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations, permutations
import json
from pathlib import Path

from Experiment.Synister.all_distance_oracle import binary_endpoint
from Experiment.Synister.structural_oracle import independent_its, isomorphic
from synkit.Chem.Mapper.identifiability import Endpoint


def compatible_maps(r, p):
    return [m for m in permutations(range(len(r.atomic_numbers)))
            if all(r.atomic_numbers[i] == p.atomic_numbers[j] for i, j in enumerate(m))]


def automorphisms(endpoint):
    bonds = set(endpoint.bonds)
    return [m for m in compatible_maps(endpoint, endpoint)
            if all(values[i] == values[j] for values in (endpoint.charges, endpoint.hcounts)
                   for i, j in enumerate(m))
            and {(*sorted((m[i], m[j])), w) for i, j, w in bonds} == bonds]


def transform(mapping, reactant_permutation, product_permutation):
    result = [None] * len(mapping)
    for i, image in enumerate(mapping):
        result[reactant_permutation[i]] = product_permutation[image]
    return tuple(result)


def doubled_cd(r, p, mapping):
    rb = {(i, j): w for i, j, w in r.bonds}
    pb = {(i, j): w for i, j, w in p.bonds}
    return sum(abs(rb.get((i, j), 0) - pb.get(tuple(sorted((mapping[i], mapping[j]))), 0))
               for i, j in combinations(range(len(mapping)), 2))


def cases():
    yield 'empty_three', Endpoint((6,)*3, (0,)*3, (0,)*3, ()), Endpoint((6,)*3, (0,)*3, (0,)*3, ())
    yield 'product_only_insufficient', Endpoint((6,)*3, (0,)*3, (0,)*3, ()), Endpoint((6,)*3, (0,)*3, (0,)*3, ((0, 1, 2),))
    for a, b in ((0, 63), (1, 1), (3, 7), (7, 11), (12, 31), (30, 45)):
        yield f'binary_{a}_{b}', binary_endpoint(a), binary_endpoint(b)
    r = Endpoint((6, 6), (0, 1), (4, 3), ())
    yield 'paired_unary_attributes', r, r
    yield 'weighted_colours', Endpoint((6, 6, 8, 8), (0,)*4, (0, 1, 0, 1), ((0, 2, 3), (1, 3, 2))), Endpoint((6, 8, 6, 8), (0,)*4, (1, 0, 0, 1), ((0, 1, 3), (2, 3, 4)))


def check_case(name, r, p):
    maps = compatible_maps(r, p)
    gr, gp = automorphisms(r), automorphisms(p)
    allowed = set(maps)
    orbits = [{transform(m, hr, hp) for hr in gr for hp in gp} for m in maps]
    graphs = [independent_its(r, p, m) for m in maps]
    comparisons = 0
    for i, m in enumerate(maps):
        if not orbits[i] <= allowed or any(doubled_cd(r, p, b) != doubled_cd(r, p, m) for b in orbits[i]):
            raise ValueError('Two-sided action changes compatibility or CD')
        for j in range(i, len(maps)):
            comparisons += 1
            if (maps[j] in orbits[i]) != isomorphic(graphs[i], graphs[j]):
                raise ValueError('Two-sided action differs from full ITS isomorphism')
    return {'case_id': name, 'reactant': asdict(r), 'product': asdict(p),
            'maps': len(maps), 'reactant_group_order': len(gr), 'product_group_order': len(gp),
            'product_orbits': len(maps)//len(gp),
            'two_sided_classes': len({frozenset(orbit) for orbit in orbits}),
            'isomorphism_comparisons': comparisons, 'all_passed': True}


def audit():
    records = [check_case(*case) for case in cases()]
    paths = [Path(__file__), Path('Experiment/Synister/structural_oracle.py'),
             Path('Experiment/Synister/all_distance_oracle.py'),
             Path('synkit/Chem/Mapper/identifiability.py')]
    return {'schema': 'synister.two-sided-its-control.v1', 'all_passed': True,
            'scope': __doc__, 'cases': records,
            'comparisons': sum(r['isomorphism_comparisons'] for r in records),
            'source_sha256': {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output is not None:
        with args.output.open('x') as handle:
            handle.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'all_passed': report['all_passed'], 'cases': len(report['cases']),
                      'comparisons': report['comparisons']}))
