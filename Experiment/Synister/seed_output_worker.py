"""E2 worker: native matched outputs and explicitly staged output diagnostics."""

from hashlib import sha256
import importlib
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch


def diagnostic(r, p, seed, output, seconds, max_maps, verified_group=None):
    """Same verified cyclic subgroup for all three diagnostic output requests.

    The group provider is replaced only within this diagnostic process. Native
    indexed benchmark calls do not use this wrapper. Expansion is an explicit
    separately timed post-process; a complete representative set is required
    to establish a complete expanded set.
    """
    from Experiment.Synister.classification_worker import verify_cyclic_group
    module = importlib.import_module('synkit.Chem.Mapper.exact.distance')
    started = time.perf_counter()
    deadline = started+seconds
    graphs = [r.graph(), p.graph()]
    before = time.perf_counter()
    discovered = None
    if verified_group is None:
        permutations, discovered = module.bounded_automorphism_permutations(
            graphs[1], False, limit=256, timeout_seconds=min(.25, max(0, deadline-time.perf_counter())),
            max_search_nodes=10000, node_properties=('charges', 'hcounts'))
        verified_group = module.largest_cyclic_subgroup(permutations)
    group = verify_cyclic_group(p, verified_group)
    group_seconds = time.perf_counter()-before
    result = {'complete': False, 'minimum_proved': False, 'minimum_doubled_cd': None,
        'mappings': [], 'termination': 'minimum_time_limit', 'group': group,
        'group_order': len(group), 'full_group_discovery_finished': discovered,
        'group_seconds': group_seconds, 'proof_seconds': None, 'enumeration_seconds': None,
        'expansion_seconds': None, 'output_unit': {'proof': 'minimum_value_and_witness',
            'representatives': 'verified_cyclic_subgroup_representatives', 'indexed': 'all_indexed_atom_maps'}[output]}
    options = dict(binary=False, max_bijections=None, tolerance=0, symmetry_pruning=True,
                   symmetry_node_properties=('charges', 'hcounts'), initial_mapping=seed)
    # False explicitly avoids claiming that this subgroup is the full group.
    with patch.object(module, 'bounded_automorphism_permutations', return_value=(group, False)):
        before = time.perf_counter()
        proof = module.enumerate_distance_mappings(graphs, CD='minimal', _optimization_only=True,
            time_limit_seconds=max(0, deadline-before), **options)
        result.update(proof_seconds=time.perf_counter()-before, proof_nodes=proof.visited_nodes,
                      proof_statistics=proof.backend_statistics)
        if not proof.complete or proof.minimum_cost is None:
            return result
        result.update(minimum_proved=True, minimum_doubled_cd=int(2*proof.minimum_cost))
        if output == 'proof':
            result.update(complete=True, termination='minimum_proved', mappings=proof.mappings)
            return result
        maps = []
        options['initial_mapping'] = proof.mappings[0]
        before = time.perf_counter()
        raw = module.enumerate_distance_mappings(graphs, CD=proof.minimum_cost,
            compute_minimum_cost=False, collect_mappings=False,
            mapping_callback=lambda mapping, cost: maps.append(tuple(mapping)),
            max_mappings=max_maps if output == 'representatives' else (max_maps+len(group)-1)//len(group),
            time_limit_seconds=max(0, deadline-before), **options)
        result.update(enumeration_seconds=time.perf_counter()-before, enumeration_nodes=raw.visited_nodes,
                      enumeration_statistics=raw.backend_statistics, representative_count=len(maps),
                      quotient_complete=raw.symmetry_quotient_complete)
    if raw.symmetry_group_order != len(group):
        raise ValueError('Diagnostic search used a different symmetry group')
    result.update(complete=raw.complete and raw.symmetry_quotient_complete,
                  termination=raw.truncation_reason or raw.status, mappings=maps)
    if raw.complete and not raw.symmetry_quotient_complete:
        result['termination'] = 'quotient_incomplete'
    if output == 'indexed':
        before = time.perf_counter()
        expanded = []
        for mapping in maps:
            for g in group:
                if len(expanded) >= max_maps or time.perf_counter() >= deadline:
                    result.update(complete=False, termination='expansion_output_limit' if len(expanded) >= max_maps else 'expansion_time_limit')
                    break
                expanded.append(tuple(g[j] for j in mapping))
            else:
                continue
            break
        result.update(expansion_seconds=time.perf_counter()-before, mappings=expanded)
    return result


def perform(task):
    started = time.perf_counter()
    from Experiment.Synister.global_milp import doubled_distance, enumerate_milp
    from synkit.Chem.Mapper.identifiability import parse_reaction
    imported = time.perf_counter()
    r, p = parse_reaction(task['reaction'])
    parsed = time.perf_counter()
    if task['stage'] == 'group':
        from Experiment.Synister.classification_worker import verify_cyclic_group
        module = importlib.import_module('synkit.Chem.Mapper.exact.distance')
        permutations, discovered = module.bounded_automorphism_permutations(p.graph(), False,
            limit=256, timeout_seconds=.25, max_search_nodes=10000, node_properties=('charges', 'hcounts'))
        group = verify_cyclic_group(p, module.largest_cyclic_subgroup(permutations))
        return {'complete': True, 'termination': 'verified_group', 'group': group, 'group_order': len(group),
                'full_group_discovery_finished': discovered, 'group_seconds': time.perf_counter()-parsed,
                'end_to_end_seconds': time.perf_counter()-started}
    if task['stage'] == 'seed':
        from synkit.Chem.Mapper.prediction_adapter import predict_slap
        before = time.perf_counter()
        prediction = predict_slap(task['reaction'])
        cost = doubled_distance(r, p, prediction['mapping'])
        return {'complete': True, 'termination': 'valid_prediction', 'prediction': prediction,
                'seed_doubled_cd': cost, 'seed_calculation_seconds': time.perf_counter()-before,
                'parse_seconds': parsed-imported, 'import_seconds': imported-started,
                'end_to_end_seconds': time.perf_counter()-started}
    seed = task.get('initial_mapping')
    if task['seed_condition'] == 'slap' and seed is None:
        return {'complete': False, 'minimum_proved': False, 'termination': 'seed_unavailable'}
    seed_cost = doubled_distance(r, p, seed) if seed is not None else None
    before = time.perf_counter()
    remaining = max(0, task['seconds']-(before-parsed))
    # Equal solver budgets exclude independently recorded prediction preparation.
    if task['stage'] == 'diagnostic':
        result = diagnostic(r, p, seed, task['output'], remaining, task['max_maps'], task['diagnostic_group'])
    elif task['method'] == 'milp':
        result = enumerate_milp(r, p, seconds=remaining, max_maps=task['max_maps'], initial_mapping=seed)
        result['output_unit'] = 'all_indexed_atom_maps'
        result['proof_seconds'] = result['minimum_proof_seconds']
        result['enumeration_seconds'] = (result['elapsed_seconds']-result['minimum_proof_seconds']
                                         if result['minimum_proof_seconds'] is not None else None)
        result['expansion_seconds'] = None
    else:
        from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
        maps = []
        raw = enumerate_distance_mappings([r.graph(), p.graph()], CD='minimal', binary=False,
            max_bijections=None, max_mappings=task['max_maps'], tolerance=0,
            time_limit_seconds=remaining, initial_mapping=seed,
            symmetry_pruning=True, expand_symmetry=True, symmetry_node_properties=('charges', 'hcounts'),
            collect_mappings=False, mapping_callback=lambda mapping, cost: maps.append(tuple(mapping)))
        elapsed = time.perf_counter()-before
        stats = (raw.backend_statistics or {}).get('search', {})
        proof_time = elapsed if stats.get('phase') == 'minimum_proof' else stats.get('minimum_proof_seconds')
        result = {'complete': raw.complete, 'minimum_proved': raw.minimum_cost is not None,
                  'minimum_doubled_cd': None if raw.minimum_cost is None else int(2*raw.minimum_cost),
                  'termination': raw.truncation_reason or raw.status, 'mappings': maps,
                  'proof_seconds': proof_time,
                  'enumeration_seconds': elapsed-proof_time if raw.minimum_cost is not None and proof_time is not None else None,
                  'expansion_seconds': None, 'output_unit': 'all_indexed_atom_maps',
                  'backend_statistics': raw.backend_statistics, 'visited_nodes': raw.visited_nodes,
                  'group_order': raw.symmetry_group_order}
    finished = time.perf_counter()
    maps = [tuple(m) for m in result.pop('mappings')]
    if len(set(maps)) != len(maps):
        raise ValueError('Duplicate requested output')
    if any(doubled_distance(r, p, m) != result['minimum_doubled_cd'] for m in maps):
        raise ValueError('Output witness differs from proved minimum')
    checked = time.perf_counter()
    encoded = json.dumps(sorted(maps), separators=(',', ':')).encode()
    with Path(task['map_path']).open('xb') as stream:
        stream.write(encoded)
    result.update(seed_doubled_cd=seed_cost, mapping_count=len(maps), mapping_sha256=sha256(encoded).hexdigest(),
        output_bytes=len(encoded), import_seconds=imported-started, parse_seconds=parsed-imported,
        search_seconds=finished-before, checking_seconds=checked-finished,
        writing_seconds=time.perf_counter()-checked, end_to_end_seconds=time.perf_counter()-started,
        timing_scope='Native enumeration includes expansion; separate expansion timing exists only in explicitly staged diagnostics')
    return result


if __name__ == '__main__':
    task = json.load(sys.stdin)
    limit = task['memory_gib']*1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    try:
        result = perform(task)
    except Exception as exc:
        result = {'complete': False, 'minimum_proved': False, 'termination': 'worker_error',
                  'error_type': type(exc).__name__, 'error': str(exc)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(worker_seconds=time.perf_counter()-started, cpu_seconds=usage.ru_utime+usage.ru_stime,
                  peak_rss_kib=usage.ru_maxrss, execution_root=str(Path(__file__).resolve().parents[2]),
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(result, allow_nan=False))
