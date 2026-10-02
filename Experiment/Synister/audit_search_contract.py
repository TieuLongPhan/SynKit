"""Bounded independent checks supporting the written search-contract review.

This is not a proof checker for Python. It tests every prefix of declared
small compatible mapping spaces, and binds the manually reviewed operations
and experiment settings to their actual source snapshots.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import inspect
from itertools import combinations, permutations
import json
from pathlib import Path
import random
import time

import numpy as np

from Experiment.Synister.all_distance_oracle import literal_sets
from synkit.Chem.Mapper.identifiability import Endpoint
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.distance_bounds import (
    ResidualBondMass, assignment_edge_lower_bounds, atom_profile_costs,
    blocked_assignment_extreme, conditioned_profile_assignment_lower_bound)
from synkit.Chem.Mapper.exact import symmetry


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    'synkit/Chem/Mapper/exact/distance.py',
    'synkit/Chem/Mapper/exact/distance_bounds.py',
    'synkit/Chem/Mapper/exact/assignment_certificate.py',
    'synkit/Chem/Mapper/exact/symmetry.py',
    'synkit/Chem/Mapper/graph/automorphism.py',
    'synkit/Chem/Mapper/slap/lap.py',
    'Experiment/Synister/all_distance_oracle.py',
    'Experiment/Synister/audit_search_contract.py',
)


def cases():
    rng = random.Random(2026092026)
    for index in range(12):
        colors = [6]*5 if index < 6 else [6, 6, 8, 8, 7]
        def endpoint():
            atoms = colors.copy()
            rng.shuffle(atoms)
            bonds = tuple((i, j, weight) for i, j in combinations(range(5), 2)
                          if (weight := rng.choice((0, 2, 3, 4, 6))))
            return Endpoint(tuple(atoms), (0,)*5, (0,)*5, bonds)
        yield f'prefix_{index:02d}', endpoint(), endpoint()


def matrix(endpoint):
    result = np.zeros((len(endpoint.atomic_numbers),)*2)
    for i, j, weight in endpoint.bonds:
        result[i, j] = result[j, i] = weight/2
    return result


def prefix_checks(name, r, p):
    oracle = literal_sets(r, p)
    mapping_costs = {m: cost/2 for cost, maps in oracle.items() for m in maps}
    a, b = matrix(r), matrix(p)
    order = [2, 0, 4, 1, 3]
    profiles = atom_profile_costs(a, b, r.atomic_numbers, p.atomic_numbers)
    prefixes = defaultdict(list)
    for mapping, cost in mapping_costs.items():
        for depth in range(6):
            prefixes[tuple(mapping[i] for i in order[:depth])].append((mapping, cost))
    checks, forced_checks, zero_row_checks = 0, 0, 0
    for images, completions in prefixes.items():
        depth = len(images)
        fixed = dict(zip(order[:depth], images))
        rows, columns = order[depth:], [j for j in range(5) if j not in images]
        # Independent integer doubled bond dictionary, not the search delta.
        rb = {(i, j): w for i, j, w in r.bonds}
        pb = {(i, j): w for i, j, w in p.bonds}
        committed = sum(abs(rb.get(tuple(sorted((i, j))), 0)
                            - pb.get(tuple(sorted((fixed[i], fixed[j]))), 0))
                        for i, j in combinations(fixed, 2))/2
        cross = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                cross[i, j] = sum(abs(rb.get(tuple(sorted((i, k))), 0)
                                      - pb.get(tuple(sorted((j, image))), 0))
                                  for k, image in fixed.items())/2
        residual = ResidualBondMass(a, b, order)
        saved = [residual.remove(image) for image in images]
        lower, upper = residual.interval(depth)
        cross_min = blocked_assignment_extreme(cross, rows, columns, r.atomic_numbers, p.atomic_numbers)
        cross_max = blocked_assignment_extreme(cross, rows, columns, r.atomic_numbers, p.atomic_numbers, maximize=True)
        profile_prefix = sum(profiles[i, j] for i, j in fixed.items())
        profile_lower = profile_prefix+blocked_assignment_extreme(profiles, rows, columns,
                                                                 r.atomic_numbers, p.atomic_numbers)
        conditioned, costs = conditioned_profile_assignment_lower_bound(
            a, b, cross, rows, columns, r.atomic_numbers, p.atomic_numbers, return_costs=True)
        actual_min, actual_max = min(c for _, c in completions), max(c for _, c in completions)
        if not (committed+cross_min+lower <= actual_min <= actual_max <= committed+cross_max+upper
                and profile_lower <= actual_min and committed+conditioned <= actual_min):
            raise AssertionError('A production pruning bound excludes a literal completion')
        # Verify exact forced LAP costs, not just a weaker admissibility test.
        if rows:
            row_colors = [r.atomic_numbers[i] for i in rows]
            col_colors = [p.atomic_numbers[j] for j in columns]
            forced = assignment_edge_lower_bounds(costs, row_colors, col_colors)
            lap = [(perm, sum(costs[i, j] for i, j in enumerate(perm)))
                   for perm in permutations(range(len(rows)))
                   if all(row_colors[i] == col_colors[j] for i, j in enumerate(perm))]
            for i, color in enumerate(row_colors):
                for j, other in enumerate(col_colors):
                    expected = min((v for perm, v in lap if perm[i] == j), default=float('inf'))
                    if forced[i, j] != expected:
                        raise AssertionError('Forced-edge cost differs from literal LAP')
                    forced_checks += 1
                    if color == other:
                        possible = [cost for m, cost in completions if m[rows[i]] == columns[j]]
                        if committed+forced[i, j] > min(possible):
                            raise AssertionError('Forced-edge pruning excludes a literal map')
        for image, delta in reversed(list(zip(images, saved))):
            residual.restore(image, delta)
        if residual.interval(0) != (abs(a.sum()-b.sum())/2, (a.sum()+b.sum())/2):
            raise AssertionError('Residual state is not exactly restored')
        checks += 1
    global_profile = blocked_assignment_extreme(profiles, range(5), range(5),
                                                r.atomic_numbers, p.atomic_numbers)
    for m, cost in mapping_costs.items():
        if cost != global_profile:
            continue
        for i, j in enumerate(m):
            if profiles[i, j] == 0:
                if any(a[i, k] != b[j, m[k]] for k in range(5)):
                    raise AssertionError('Attained zero-profile row has a bond mismatch')
                zero_row_checks += 1
    return {'case_id': name, 'reactant': asdict(r), 'product': asdict(p),
            'compatible_maps': len(mapping_costs), 'prefixes': checks,
            'forced_assignment_entries': forced_checks,
            'attained_zero_profile_rows': zero_row_checks, 'passed': True}


def query_checks():
    # A symmetric single-bond pair exercises attained-profile propagation,
    # all numeric half-step targets, nontrivial expansion and fixed subspaces.
    r = Endpoint((6,)*5, (0,)*5, (0,)*5, ((0, 1, 2),))
    p = Endpoint((6,)*5, (0,)*5, (0,)*5, ((2, 3, 2),))
    graph = [r.graph(), p.graph()]
    records = []
    for fixed in (None, {0: 2}):
        oracle = literal_sets(r, p, fixed)
        minimum = min(oracle)
        seed = max((m for maps in oracle.values() for m in maps), key=lambda m: (next(c for c, maps in oracle.items() if m in maps), m))
        for tolerance in (0, 1e-9, np.nextafter(.25, 0)):
            for target in ('minimal', *range(7)):
                expected = oracle.get(minimum if target == 'minimal' else target, set())
                for expand in ((False,) if fixed else (False, True)):
                    emitted = []
                    result = enumerate_distance_mappings(
                        graph, CD='minimal' if target == 'minimal' else target/2,
                        binary=False, max_bijections=None, tolerance=tolerance,
                        initial_mapping=seed, fixed_mapping=fixed,
                        collect_mappings=False,
                        mapping_callback=lambda mapping, cost: emitted.append(tuple(mapping)),
                        compute_minimum_cost=target == 'minimal',
                        symmetry_pruning=expand, expand_symmetry=expand)
                    if not result.complete or set(emitted) != expected or len(emitted) != len(expected):
                        raise AssertionError('Streamed supported-domain query differs from literal map set')
                    if expand and not result.symmetry_quotient_complete:
                        raise AssertionError('Default cyclic expansion has an incomplete stabilizer chain')
                    records.append({'fixed': fixed, 'tolerance': float(tolerance), 'target_doubled_cd': target,
                                    'expanded': expand, 'maps': len(emitted),
                                    'dynamic_row_order': result.backend_statistics.get('search', {}).get('dynamic_profile_order')
                                        if result.backend_statistics else None})
    return records


def cyclic_settings():
    maximum = inspect.signature(enumerate_distance_mappings).parameters['max_symmetry_automorphisms'].default
    if not (maximum == 256 and maximum-1 <= symmetry._MAX_STABILIZER_GENERATORS
            and maximum*(maximum-1) <= symmetry._MAX_SCHREIER_WORK):
        raise AssertionError('Default expansion settings no longer satisfy the written stabilizer bound')
    permutations_ = tuple(tuple((i+j) % maximum for i in range(maximum)) for j in range(maximum))
    remaining, complete = symmetry.point_stabilizer_generators_checked(permutations_[1:], 0)
    if not complete or remaining:
        raise AssertionError('Regular cyclic boundary control failed')
    return {'default_maximum_order': maximum,
            'maximum_orbit_generator_applications': maximum*(maximum-1),
            'work_limit': symmetry._MAX_SCHREIER_WORK,
            'maximum_distinct_nonidentity_generators': maximum-1,
            'generator_limit': symmetry._MAX_STABILIZER_GENERATORS,
            'regular_cyclic_boundary_order': maximum, 'boundary_check_passed': True}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = [prefix_checks(*case) for case in cases()]
    # A nonrandom attained-bound control ensures zero-row propagation is tested.
    endpoint = Endpoint((6,)*5, (0,)*5, (0,)*5, ((0, 1, 2),))
    rows.append(prefix_checks('attained_zero_rows', endpoint, endpoint))
    queries, settings = query_checks(), cyclic_settings()
    sources = {path: (ROOT/path).read_text() for path in SOURCE_PATHS}
    documents = {'cases.json': rows, 'queries.json': queries, 'sources.json': sources}
    for name, value in documents.items():
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    summary = {'schema': 'synister.search-contract-controls.v1',
               'cases': len(rows), 'prefixes': sum(row['prefixes'] for row in rows),
               'forced_assignment_entries': sum(row['forced_assignment_entries'] for row in rows),
               'attained_zero_profile_rows': sum(row['attained_zero_profile_rows'] for row in rows),
               'streamed_queries': len(queries), 'cyclic_settings': settings,
               'all_passed': True, 'seconds': time.perf_counter()-started,
               'file_sha256': {name: sha256((output/name).read_bytes()).hexdigest() for name in documents},
               'scope': 'Bounded controls and source binding for a manual proof-to-implementation review, not formal verification of Python or a benchmark.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
