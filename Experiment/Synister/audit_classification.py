"""Independent graph-isomorphism validation of saved full-ITS assignments.

This auditor does not call the production ITS builder or canonicalizer. A
necessary graph signature only removes impossible comparisons; class equality
always requires attributed NetworkX graph isomorphism. Budget exits are not
successful verification.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import resource
import sys
import time


class AuditTimeLimit(Exception):
    pass


def audit_details(r, p, mappings, result, details, target, *, seconds=60):
    import networkx as nx
    from Experiment.Synister.classification_worker import verify_cyclic_group
    from Experiment.Synister.structural_oracle import independent_its

    started = time.perf_counter()
    deadline = started+seconds
    def check_time():
        if time.perf_counter() >= deadline:
            raise AuditTimeLimit()

    class Matcher(nx.algorithms.isomorphism.GraphMatcher):
        def syntactic_feasibility(self, first, second):
            check_time()
            return super().syntactic_feasibility(first, second)

    def equivalent(first, second):
        check_time()
        return Matcher(first, second,
                       node_match=lambda a, b: a['attributes'] == b['attributes'],
                       edge_match=lambda a, b: a['attributes'] == b['attributes']).is_isomorphic()

    def fingerprint(graph):
        return (tuple(sorted(data['attributes'] for _, data in graph.nodes(data=True))),
                tuple(sorted(data['attributes'] for _, _, data in graph.edges(data=True))),
                tuple(sorted((graph.nodes[i]['attributes'], graph.degree(i)) for i in graph)))

    maps = set(map(tuple, mappings))
    if len(maps) != len(mappings) or len(maps) != result['indexed_maps']:
        raise ValueError('Indexed input count or uniqueness mismatch')
    group = verify_cyclic_group(p, result['group_permutations'])
    h = len(group)
    if result['group_order'] != h or len(maps) != h*result['product_orbits']:
        raise ValueError('Product-orbit cardinality mismatch')
    codes = {item['class_id']: item['canonical_code'] for item in details['class_codes']}
    if (len(codes) != len(details['class_codes']) or set(codes) != set(range(len(codes)))
            or len(set(codes.values())) != len(codes)):
        raise ValueError('Invalid canonical class inventory')
    seen, classes, buckets = set(), {}, defaultdict(list)
    bond_counts, joint_counts = Counter(), Counter()
    coded, pairs, checked, failures = 0, 0, 0, []
    try:
        for record in details['representatives']:
            check_time()
            mapping = tuple(record['mapping'])
            n = len(r.atomic_numbers)
            if (len(mapping) != n or sorted(mapping) != list(range(n))
                    or any(r.atomic_numbers[i] != p.atomic_numbers[j] for i, j in enumerate(mapping))):
                raise ValueError('Invalid representative bijection')
            members = {tuple(g[j] for j in mapping) for g in group}
            if len(members) != h or not members <= maps or members & seen or record['multiplicity'] != h:
                raise ValueError('Invalid or repeated product orbit')
            seen.update(members)
            graph = independent_its(r, p, mapping)
            edits = tuple((i, j, *graph.edges[i, j]['attributes']) for i, j in combinations(range(n), 2)
                          if graph.has_edge(i, j) and graph.edges[i, j]['attributes'][0] != graph.edges[i, j]['attributes'][1])
            unary = []
            for i, data in graph.nodes(data=True):
                _, qr, qp, hr, hp = data['attributes']
                if qr != qp:
                    unary.append((i, 'charges', qr, qp))
                if hr != hp:
                    unary.append((i, 'hcounts', hr, hp))
            if sum(abs(after-before) for _, _, before, after in edits) != target:
                raise ValueError('Independent ITS has incorrect CD')
            bond_key = repr(tuple((i, j) for i, j, _, _ in edits))
            joint_key = repr((edits, tuple(sorted(unary))))
            if record['bond_pattern'] != bond_key or record['joint_pattern'] != joint_key:
                raise ValueError('Incorrect fixed-coordinate change pattern')
            bond_counts[bond_key] += h
            joint_counts[joint_key] += h
            cid = record['its_class']
            if cid is None:
                failures.append(mapping)
            else:
                if cid not in codes:
                    raise ValueError('Unknown ITS class')
                key = fingerprint(graph)
                if cid in classes:
                    pairs += 1
                    if not equivalent(graph, classes[cid]):
                        raise ValueError('Canonical identity merged nonisomorphic ITS graphs')
                else:
                    for other in buckets[key]:
                        pairs += 1
                        if equivalent(graph, classes[other]):
                            raise ValueError('Canonical identities split isomorphic ITS graphs')
                    classes[cid] = graph
                    buckets[key].append(cid)
                coded += 1
            checked += 1
    except AuditTimeLimit:
        return {'verified': False, 'consistent': None, 'termination': 'audit_time_limit',
                'checked_orbits': checked, 'isomorphism_comparisons': pairs,
                'elapsed_seconds': time.perf_counter()-started}
    partition_complete = seen == maps
    complete = partition_complete and not failures
    if set(classes) != set(codes) or result['observed_canonical_classes'] != len(classes):
        raise ValueError('Reported class inventory differs from independently checked assignments')
    if (result['processed_orbits'] != checked or result['processed_indexed_maps'] != len(seen)
            or result['classified_orbits'] != coded):
        raise ValueError('Processed output denominator mismatch')
    if sorted(tuple(item['representative']) for item in details['failures']) != sorted(failures):
        raise ValueError('Canonicalization failures were lost')
    for field, expected in (('orbit_partition_complete', partition_complete),
                            ('complete', complete), ('classification_complete', complete),
                            ('its_classes', len(classes) if complete else None),
                            ('its_class_lower_bound', max(int(bool(maps)), len(classes))),
                            ('its_class_upper_bound', min(result['product_orbits'], len(classes)+result['product_orbits']-coded))):
        if result[field] != expected:
            raise ValueError(f'Incorrect classification claim: {field}')
    for name, counts in (('bond', bond_counts), ('joint', joint_counts)):
        if (details[f'{name}_pattern_frequencies'] != dict(counts)
                or result[f'{name}_pattern_lower_bound'] != len(counts)
                or result[f'{name}_pattern_count'] != (len(counts) if partition_complete else None)):
            raise ValueError('Change-pattern frequency or completion mismatch')
    return {'verified': True, 'consistent': True, 'termination': 'verified',
            'checked_orbits': checked, 'isomorphism_comparisons': pairs,
            'independent_observed_classes': len(classes),
            'elapsed_seconds': time.perf_counter()-started,
            'scope': 'All saved class assignments and separations, independent full ITS graphs; no canonical-code replay'}


def perform(task):
    from synkit.Chem.Mapper.identifiability import parse_reaction
    map_bytes = Path(task['map_path']).read_bytes()
    detail_bytes = Path(task['detail_path']).read_bytes()
    if (sha256(map_bytes).hexdigest() != task['mapping_sha256']
            or sha256(detail_bytes).hexdigest() != task['result']['detail_sha256']):
        raise ValueError('Changed mapping or classification detail file')
    r, p = parse_reaction(task['reaction'])
    return audit_details(r, p, json.loads(map_bytes), task['result'], json.loads(detail_bytes),
                         task['target_doubled_cd'], seconds=task['audit_seconds'])


if __name__ == '__main__':
    task = json.load(sys.stdin)
    limit = int(task['memory_gib']*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        report = perform(task)
    except MemoryError:
        report = {'verified': False, 'consistent': None, 'termination': 'audit_memory_limit'}
    except Exception as exc:
        report = {'verified': False, 'consistent': False, 'termination': 'audit_error',
                  'error_type': type(exc).__name__, 'error': str(exc)}
    print(json.dumps(report, allow_nan=False))
