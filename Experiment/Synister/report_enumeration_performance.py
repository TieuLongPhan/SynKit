"""Descriptive resource accounting from terminal, audited enumeration studies.

No search is run here. Missing measurements remain missing, and incomplete
queries never contribute a completion event or a completed-output size.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'paper/synister'
METRICS = ('parent_seconds', 'worker_seconds', 'cpu_seconds', 'peak_rss_kib',
           'parse_seconds', 'endpoint_construction_seconds', 'seed_seconds',
           'minimum_proof_seconds', 'first_map_seconds', 'search_finished_seconds',
           'search_seconds', 'checking_seconds', 'export_seconds',
           'end_to_end_seconds', 'output_bytes', 'mapping_count', 'visited_nodes',
           'minimum_proof_nodes', 'pruned_branches')
SUCCESS = {'matched': 'all_output_comparisons_consistent',
           'ablation': 'all_output_comparisons_consistent',
           'scaling': 'all_verified', 'repeats': 'all_repeat_comparisons_consistent'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_study(directory, kind):
    """Verify the audit's inputs and every attempt/output before aggregation."""
    directory = Path(directory)
    manifest, summary, audit = [json.loads((directory/name).read_text())
                                for name in ('manifest.json', 'summary.json', 'audit.json')]
    if audit.get(SUCCESS[kind]) is not True:
        raise ValueError('Study lacks a successful terminal audit')
    if kind == 'repeats' and audit.get('all_output_comparisons_consistent') is not True:
        raise ValueError('Repeated study has inconsistent method outputs')
    if audit['summary_sha256'] != digest(directory/'summary.json'):
        raise ValueError('Summary changed after audit')
    if 'manifest_sha256' in audit and audit['manifest_sha256'] != digest(directory/'manifest.json'):
        raise ValueError('Manifest changed after audit')
    expected_files = dict(manifest.get('file_sha256', {}))
    for filename, key in (('inputs.json', 'inputs_sha256'), ('sources.json', 'sources_sha256')):
        if key in manifest:
            expected_files[filename] = manifest[key]
    for filename, value in expected_files.items():
        if digest(directory/filename) != value:
            raise ValueError('Recorded input or source snapshot changed')
    tasks = (json.loads((directory/'tasks.json').read_text()) if kind in ('ablation', 'scaling')
             else json.loads((directory/'minimum_tasks.json').read_text())
             + json.loads((directory/'numeric_tasks.json').read_text()))
    expected = {task['task_id']: task for task in tasks}
    paths = sorted((directory/'cases').glob('*.json'))
    if len(expected) != len(tasks) or {p.stem for p in paths} != set(expected):
        raise ValueError('Missing, extra or duplicate attempts')
    if set(summary['all_record_hashes']) != {p.name for p in paths}:
        raise ValueError('Attempt hash inventory differs')
    records = []
    for path in paths:
        if digest(path) != summary['all_record_hashes'][path.name]:
            raise ValueError('Attempt changed after audit')
        record = json.loads(path.read_text())
        if {k: record['task'].get(k) for k in expected[path.stem]} != expected[path.stem]:
            raise ValueError('Attempt differs from planned task')
        if any(record['task'].get(key) != manifest[key]
               for key in ('seconds', 'memory_gib', 'max_maps') if key in manifest):
            raise ValueError('Attempt resource limit differs from manifest')
        if 'mapping_sha256' in record:
            saved = directory/'maps'/path.name
            if digest(saved) != record['mapping_sha256'] or saved.stat().st_size != record['output_bytes']:
                raise ValueError('Saved mapping output changed')
        elif record['complete']:
            raise ValueError('Completed query lacks saved output')
        records.append(record)
    inputs = json.loads((directory/'inputs.json').read_text())
    if len(records) != summary['attempts'] or len(records) != audit['attempts']:
        raise ValueError('Attempt denominator differs from audited study')
    if len(inputs) != summary['selected'] or len(inputs) != audit['selected']:
        raise ValueError('Reaction denominator differs from audited study')
    return {'manifest': manifest, 'records': records, 'inputs': inputs,
            'provenance': {'directory': str(directory),
                           'sha256': {name: digest(directory/name) for name in
                                      ('manifest.json', 'summary.json', 'audit.json')},
                           'attempt_hashes': summary['all_record_hashes']}}


def describe(values, denominator):
    """No imputation of missing observations, including parent-killed workers."""
    observed = [value for value in values if value is not None]
    if any(not isinstance(v, (float, int)) or not isfinite(v) or v < 0 for v in observed):
        raise ValueError('Invalid nonnegative resource measurement')
    return {'observed': len(observed), 'missing': denominator-len(observed),
            'min': min(observed) if observed else None,
            'median': median(observed) if observed else None,
            'max': max(observed) if observed else None}


def summarize(records):
    completed = [r for r in records if r['complete']]
    precheck = [r for r in records if r.get('termination') == 'proved_empty_precheck']
    if any(not r['complete'] or r.get('mapping_count') != 0 for r in precheck):
        raise ValueError('Invalid elementary empty-target result')
    minimum = [r for r in records if r['task']['query'] == 'minimum']
    return {'attempts': len(records), 'complete': len(completed),
            'incomplete': len(records)-len(completed),
            'minimum_attempts': len(minimum),
            'minimum_proved': sum(bool(r.get('minimum_proved')) for r in minimum),
            'proved_empty': sum(r.get('mapping_count') == 0 for r in completed),
            'precheck_empty': len(precheck),
            'terminations': dict(sorted(Counter(r['termination'] for r in records).items())),
            'completion_events_parent_seconds': sorted(r['parent_seconds'] for r in completed),
            'search_completion_events_parent_seconds': sorted(r['parent_seconds'] for r in completed
                if r.get('termination') != 'proved_empty_precheck'),
            'search_attempts': len(records)-len(precheck),
            'resources_all_attempts': {key: describe([measurement(r, key) for r in records], len(records))
                                       for key in METRICS},
            'resources_completed_only': {key: describe([measurement(r, key) for r in completed], len(completed))
                                         for key in METRICS}}


def measurement(record, key):
    # The implementation initializes this timer to zero even when optimization
    # terminates without proving an optimum. That zero is not a proof time.
    if key == 'minimum_proof_seconds' and (record['task']['query'] != 'minimum'
                                           or not record.get('minimum_proved')):
        return None
    return record.get(key)


def compact(record):
    task = record['task']
    row = {key: task.get(key) for key in ('task_id', 'benchmark_id', 'query', 'repeat',
                                         'method', 'variant', 'target_doubled_cd')}
    row.update({key: measurement(record, key) for key in (*METRICS, 'complete', 'minimum_proved',
                                                'minimum_doubled_cd', 'termination')})
    row['raw_minimum_proof_seconds'] = record.get('minimum_proof_seconds')
    row['search_statistics'] = (record.get('backend_statistics') or {}).get('search', {})
    return row


def group_summary(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[record['task'][field], record['task']['query']].append(record)
    return [{'configuration': config, 'query': query, **summarize(rows)}
            for (config, query), rows in sorted(groups.items())]


def conditional_ratios(records, field, reference, other):
    """Paired completed queries only; never a global speedup estimate."""
    groups = defaultdict(dict)
    for row in records:
        task = row['task']
        key = task['benchmark_id'], task['query'], task.get('repeat', 0)
        name = task[field]
        if name in groups[key]:
            raise ValueError('Duplicate configuration in paired comparison')
        groups[key][name] = row
    if any(reference not in g or other not in g for g in groups.values()):
        raise ValueError('Unmatched configurations')
    pairs = [(g[reference], g[other]) for g in groups.values()]
    joint = [(a, b) for a, b in pairs if a['complete'] and b['complete']]
    ratios = [b['parent_seconds']/a['parent_seconds'] for a, b in joint]
    nodes = [b['visited_nodes']/a['visited_nodes'] for a, b in joint
             if a.get('visited_nodes', 0) and b.get('visited_nodes') is not None]
    return {'reference': reference, 'other': other, 'paired_attempts': len(pairs),
            'jointly_complete': len(joint),
            'reference_only_complete': sum(a['complete'] and not b['complete'] for a, b in pairs),
            'other_only_complete': sum(b['complete'] and not a['complete'] for a, b in pairs),
            'neither_complete': sum(not a['complete'] and not b['complete'] for a, b in pairs),
            'other_over_reference_parent_seconds': describe(ratios, len(pairs)),
            'other_over_reference_visited_nodes': describe(nodes, len(pairs)),
            'scope': 'Conditional on jointly completed attempts, not an all-attempt speedup'}


def summarize_study(study, kind):
    records = study['records']
    field = 'method' if kind in ('matched', 'repeats') else 'variant'
    configurations = sorted({r['task'][field] for r in records})
    reference = 'synister' if field == 'method' else 'full'
    result = {'selected': len(study['inputs']), 'attempts': len(records),
              'manifest': study['manifest'], 'provenance': study['provenance'],
              'by_configuration_query': group_summary(records, field),
              'by_configuration': {v: summarize([r for r in records if r['task'][field] == v])
                                   for v in configurations},
              'conditional_pairs': [conditional_ratios(records, field, reference, v)
                                    for v in configurations if v != reference],
              'conditional_pairs_by_query': {query: [conditional_ratios(
                    [r for r in records if r['task']['query'] == query], field, reference, v)
                    for v in configurations if v != reference]
                    for query in sorted({r['task']['query'] for r in records})},
              'attempt_measurements': [compact(r) for r in records]}
    if kind in ('matched', 'repeats'):
        result['completion_curves'] = {query: {method: summarize([r for r in records
                    if r['task']['method'] == method and (r['task']['query'] == 'minimum') == (query == 'minimum')])
                    for method in configurations} for query in ('minimum', 'specified_cd')}
    if kind == 'ablation':
        result['planned_queries_per_configuration'] = (len(study['inputs'])*3*study['manifest']['repeats'])
        result['unavailable_queries_per_configuration'] = (
            result['planned_queries_per_configuration']-len(records)//len(configurations))
    if kind == 'scaling':
        lookup = {row['benchmark_id']: row for row in study['inputs']}
        for row in result['attempt_measurements']:
            example = lookup[row['benchmark_id']]
            row.update(family=example['family'], parameter=example['parameter'],
                       atoms=len(example['reactant']['atomic_numbers']),
                       expected_indexed_maps=example['expected_indexed_maps'])
            count = row['mapping_count']
            if count is not None and (count > example['expected_indexed_maps'] or
                    row['complete'] and count != example['expected_indexed_maps']):
                raise ValueError('Output-size observation differs from analytic count')
    return result


def build(directories):
    return {'schema': 'synister.enumeration-performance-report.v1',
            'reporter_sha256': digest(Path(__file__)),
            'semantics': {
                'completion_curves': 'Recorded terminal completion times from one bounded attempt, not independent shorter-budget reruns; all attempted queries remain in the denominator.',
                'comparison_clock': 'parent_seconds includes process startup/import, parsing, search, checking, serialization and writing.',
                'missing_measurements': 'A killed worker may have a parent time but no phase, output or RSS observation; missing is never zero.',
                'peak_rss_kib': 'Linux whole-process high-water RSS, including imports and retained outputs; not isolated search memory.',
                'phase_clocks': 'Cumulative and duration measurements overlap and must not be summed. Internal proof/first-map timers have method-specific origins.',
                'proof_timer_applicability': 'Minimum-proof time is summarized only where minimum_proved is true for a minimum query; raw default zeros from interrupted optimization are retained separately, not interpreted as instant proofs.',
                'synister_minimum_proof_seconds': 'Duration of recursive minimum optimization; excludes outer setup and subsequent full-set traversal.',
                'milp_minimum_proof_seconds': 'Elapsed model construction, minimum solve and integer check within enumerate_milp; excludes parsing.',
                'synister_first_map_seconds': 'From worker parse start to first returned minimum/numeric map, not first feasible optimization incumbent.',
                'milp_first_map_seconds': 'From enumerate_milp start after parsing to first returned map, not first feasible optimization incumbent.',
                'ablation_first_map_seconds': 'From variant start, including seed construction, to first returned map; excludes parsing/import.',
                'classification': 'Separate downstream study; not included in indexed-map enumeration times.',
                'repeat_variation': 'All repeat attempts retained; no fastest-repeat selection or population confidence interval.',
                'scaling': 'Mathematical graph stress tests with analytic indexed-map counts, not chemically representative reactions.'},
            'studies': {kind: summarize_study(load_study(directory, kind), kind)
                        for kind, directory in directories.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched', type=Path, required=True)
    for name in ('ablation', 'scaling', 'repeats'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build({key: getattr(args, key) for key in SUCCESS if getattr(args, key) is not None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: {'selected': value['selected'], 'attempts': value['attempts']}
                      for key, value in result['studies'].items()}, indent=2))
