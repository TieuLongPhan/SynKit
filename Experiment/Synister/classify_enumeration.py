"""Classify every available complete set in a finished matched-CD study.

Equal reaction/CD queries share structural work, while every original solver
attempt and its original completion state remain in the saved query aliases.
An incomplete enumeration is never promoted by this post-processing step.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def encode(value):
    return json.dumps(value, indent=2, allow_nan=False)+'\n'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare(matched, *, seconds, audit_seconds, memory_gib):
    manifest = json.loads((matched/'manifest.json').read_text())
    summary = json.loads((matched/'summary.json').read_text())
    audit = json.loads((matched/'audit.json').read_text())
    if (not audit['all_output_comparisons_consistent']
            or audit['manifest_sha256'] != digest(matched/'manifest.json')
            or audit['summary_sha256'] != digest(matched/'summary.json')
            or manifest['inputs_sha256'] != digest(matched/'inputs.json')
            or manifest['sources_sha256'] != digest(matched/'sources.json')):
        raise ValueError('Upstream study audit or identity mismatch')
    rows = json.loads((matched/'inputs.json').read_text())
    inputs = {r['benchmark_id']: r for r in rows}
    tasks = json.loads((matched/'minimum_tasks.json').read_text())+json.loads((matched/'numeric_tasks.json').read_text())
    paths = sorted((matched/'cases').glob('*.json'))
    if (len(tasks) != summary['attempts'] or {p.stem for p in paths} != {t['task_id'] for t in tasks}
            or len(inputs) != summary['selected'] or len({t['task_id'] for t in tasks}) != len(tasks)):
        raise ValueError('Upstream attempt or input denominator mismatch')
    records, proofs = {}, defaultdict(set)
    expected = {t['task_id']: t for t in tasks}
    for path in paths:
        if digest(path) != summary['all_record_hashes'][path.name]:
            raise ValueError('Upstream record changed')
        record = json.loads(path.read_text())
        task = record['task']
        if ({k: task[k] for k in expected[path.stem]} != expected[path.stem]
                or task['reaction'] != inputs[task['benchmark_id']]['reaction']):
            raise ValueError('Upstream task changed')
        records[path.stem] = record
        if task['query'] == 'minimum' and record.get('minimum_proved'):
            proofs[task['benchmark_id']].add(record['minimum_doubled_cd'])
    if any(len(values) != 1 for values in proofs.values()):
        raise ValueError('Conflicting minimum proofs')
    groups = defaultdict(list)
    for key, record in records.items():
        task = record['task']
        target = task['target_doubled_cd']
        if target == 'minimal':
            target = next(iter(proofs[task['benchmark_id']]), None)
        groups[(task['benchmark_id'], target)].append(key)
    prepared = []
    for (bid, target), ids in sorted(groups.items(), key=lambda pair: (pair[0][0], -1 if pair[0][1] is None else pair[0][1])):
        aliases = [{'task_id': key, 'query': records[key]['task']['query'],
                    'method': records[key]['task']['method'], 'complete': records[key]['complete'],
                    'termination': records[key]['termination'],
                    'minimum_proved': records[key].get('minimum_proved', False),
                    'mapping_count': records[key].get('mapping_count'),
                    'mapping_sha256': records[key].get('mapping_sha256'),
                    'record_sha256': summary['all_record_hashes'][key+'.json']} for key in sorted(ids)]
        complete = [key for key in ids if records[key]['complete']]
        identities = {(records[key]['mapping_count'], records[key]['mapping_sha256']) for key in complete}
        if len(identities) > 1:
            raise ValueError('Complete sets at the same reaction/CD differ across query aliases')
        chosen = min(complete, key=lambda key: (records[key]['task']['method'] != 'synister', key)) if complete else None
        item = {'task_id': f'{bid}.cd_{target}' if target is not None else f'{bid}.minimum_unproved',
                'benchmark_id': bid, 'reaction': inputs[bid]['reaction'],
                'source': inputs[bid]['source'], 'atoms': inputs[bid]['atoms'],
                'size_bin': inputs[bid]['size_bin'], 'target_doubled_cd': target,
                'query_aliases': aliases, 'complete_indexed_set_available': chosen is not None,
                'source_task_id': chosen, 'seconds': seconds, 'audit_seconds': audit_seconds,
                'memory_gib': memory_gib,
                'partial_mapping_lower_bound': max((records[key].get('mapping_count', 0) for key in ids), default=0)}
        if chosen:
            record = records[chosen]
            map_path = matched/'maps'/f'{chosen}.json'
            if digest(map_path) != record['mapping_sha256'] or map_path.stat().st_size != record['output_bytes']:
                raise ValueError('Chosen complete mapping output changed')
            item.update(mapping_count=record['mapping_count'], mapping_sha256=record['mapping_sha256'])
        prepared.append(item)
    return rows, prepared


def isolated(module, task, seconds):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONHASHSEED='0')
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, '-m', module], input=encode(task),
                                 capture_output=True, text=True, env=env, timeout=seconds+15)
        record = json.loads(process.stdout) if process.returncode == 0 else {
            'complete': False, 'termination': 'process_failure', 'returncode': process.returncode}
        record['stderr'] = process.stderr[-4000:]
    except subprocess.TimeoutExpired:
        record = {'complete': False, 'termination': 'external_time_limit'}
    record['parent_seconds'] = time.perf_counter()-started
    return record


def execute(task, matched, output):
    path = output/'cases'/f'{task["task_id"]}.json'
    if path.exists():
        raise FileExistsError(path)
    if not task['complete_indexed_set_available']:
        result = {'complete': False, 'classification_complete': False,
                  'termination': 'not_classified_incomplete_enumeration',
                  'indexed_maps': None, 'product_orbits': None, 'its_classes': None,
                  'structural_audit': {'verified': False, 'consistent': None, 'termination': 'not_applicable'}}
    else:
        detail_path = output/'details'/path.name
        runtime = {**task, 'map_path': str((matched/'maps'/f'{task["source_task_id"]}.json').resolve()),
                   'detail_path': str(detail_path.resolve())}
        result = isolated('Experiment.Synister.classification_worker', runtime, task['seconds'])
        if 'detail_sha256' in result:
            result['structural_audit'] = isolated('Experiment.Synister.audit_classification',
                                                 {**runtime, 'result': result}, task['audit_seconds'])
        else:
            result['structural_audit'] = {'verified': False, 'consistent': None,
                                          'termination': 'no_classification_output'}
    record = {**result, 'task': task}
    with path.open('x') as stream:
        stream.write(encode(record))
    print(json.dumps({'task': task['task_id'], 'termination': result['termination'],
                      'L': result.get('indexed_maps'), 'Q': result.get('product_orbits'),
                      'K': result.get('its_classes'), 'verified': result['structural_audit'].get('verified', False)}), flush=True)
    return record


def audit(output, matched):
    """Reconcile the saved plans, outputs and independent structural checks.

    This integrity/accounting audit does not rerun timed isomorphism checks.
    Their completion and contradiction states are reported separately.
    """
    manifest = json.loads((output/'manifest.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        if digest(output/name) != expected:
            raise ValueError('Classification input, task or source snapshot changed')
    for name, expected in manifest['upstream_sha256'].items():
        if digest(matched/name) != expected:
            raise ValueError('Upstream provenance changed')
    rows, planned = prepare(matched, seconds=manifest['seconds'], audit_seconds=manifest['audit_seconds'],
                            memory_gib=manifest['memory_gib'])
    if rows != json.loads((output/'inputs.json').read_text()) or planned != json.loads((output/'tasks.json').read_text()):
        raise ValueError('Saved classification selection differs from upstream queries')
    paths = sorted((output/'cases').glob('*.json'))
    if {p.stem for p in paths} != {t['task_id'] for t in planned}:
        raise ValueError('Missing or unexpected classification record')
    tasks = {t['task_id']: t for t in planned}
    records = []
    for path in paths:
        if digest(path) != summary['all_record_hashes'][path.name]:
            raise ValueError('Classification record changed')
        record = json.loads(path.read_text())
        if record['task'] != tasks[path.stem]:
            raise ValueError('Executed classification task differs from plan')
        if 'detail_sha256' in record:
            detail_path = output/'details'/path.name
            if digest(detail_path) != record['detail_sha256'] or detail_path.stat().st_size != record['output_bytes']:
                raise ValueError('Structural classification details changed')
        elif record.get('complete'):
            raise ValueError('Complete classification without detail output')
        records.append(record)
    available = [r for r in records if r['task']['complete_indexed_set_available']]
    detailed = [r for r in available if 'detail_sha256' in r]
    aliases = [a for t in planned for a in t['query_aliases']]
    if (summary['selected_reactions'] != len(rows) or summary['unique_queries'] != len(planned)
            or summary['original_attempts'] != len(aliases)):
        raise ValueError('Classification denominator mismatch')
    return {'schema': 'synister.classification-audit.v1', 'accounting_verified': True,
            'selected_reactions': len(rows), 'original_attempts': len(aliases),
            'original_paired_queries': len({a['task_id'].rsplit('.', 1)[0] for a in aliases}),
            'unique_queries': len(planned), 'complete_indexed_sets_available': len(available),
            'no_complete_indexed_set': len(records)-len(available),
            'classification_details_saved': len(detailed),
            'complete_classifications': sum(r.get('classification_complete', False) for r in detailed),
            'independently_verified_classifications': sum(r['structural_audit'].get('verified', False) for r in detailed),
            'independently_verified_complete_classifications': sum(r.get('classification_complete', False)
                and r['structural_audit'].get('verified', False) for r in detailed),
            'structural_contradictions': sum(r['structural_audit'].get('consistent') is False for r in records),
            'all_saved_classifications_independently_verified': all(r['structural_audit'].get('verified', False) for r in detailed),
            'termination_counts': dict(Counter(r['termination'] for r in records)),
            'structural_audit_termination_counts': dict(Counter(r['structural_audit']['termination'] for r in records)),
            'manifest_sha256': digest(output/'manifest.json'), 'summary_sha256': digest(output/'summary.json'),
            'auditor_sha256': digest(Path(__file__))}


def run(args):
    if args.after is not None and not json.loads((args.after/'audit.json').read_text())['all_verified']:
        raise ValueError('Preceding binary-backend comparison failed its audit')
    rows, tasks = prepare(args.matched, seconds=args.seconds, audit_seconds=args.audit_seconds,
                           memory_gib=args.memory_gib)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'cases').mkdir()
    (args.output/'details').mkdir()
    paths = sorted(Path('synkit/Chem/Mapper').rglob('*.py'))+sorted(Path('synkit/Graph/Canon').rglob('*.py'))
    paths += [Path(__file__), Path('Experiment/Synister/classification_worker.py'),
              Path('Experiment/Synister/audit_classification.py'), Path('Experiment/Synister/structural_oracle.py'),
              Path('Experiment/Synister/all_distance_oracle.py'), Path('Experiment/Synister/worked_oracle.py')]
    saved = {'inputs.json': rows, 'tasks.json': tasks, 'sources.json': {str(p): p.read_text() for p in paths}}
    for name, value in saved.items():
        (args.output/name).write_text(encode(value))
    manifest = {'schema': 'synister.enumeration-classification.v1', 'seconds': args.seconds,
                'audit_seconds': args.audit_seconds, 'external_allowance_seconds': 15,
                'workers': args.workers, 'threads_per_worker': 1, 'memory_gib': args.memory_gib,
                'python': sys.version, 'platform': platform.platform(),
                'dependencies': {name: version(name) for name in ('numpy', 'networkx', 'rdkit')},
                'matched_directory': str(args.matched),
                'upstream_sha256': {name: digest(args.matched/name) for name in
                    ('manifest.json', 'summary.json', 'audit.json', 'inputs.json', 'minimum_tasks.json', 'numeric_tasks.json')},
                'file_sha256': {name: digest(args.output/name) for name in saved},
                'selection': 'All reaction/CD groups; equal targets share classification, original attempts unchanged',
                'group_scope': 'Newly declared verified cyclic product subgroup, not necessarily the search subgroup or full group',
                'scope': 'Post-processing only; timings are not part of or substituted for enumeration timings'}
    (args.output/'manifest.json').write_text(encode(manifest))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda task: execute(task, args.matched, args.output), tasks))
    summary = {'selected_reactions': len(rows), 'unique_queries': len(tasks),
               'original_attempts': sum(len(t['query_aliases']) for t in tasks),
               'all_record_hashes': {p.name: digest(p) for p in sorted((args.output/'cases').glob('*.json'))}}
    (args.output/'summary.json').write_text(encode(summary))
    checked = audit(args.output, args.matched)
    (args.output/'audit.json').write_text(encode(checked))
    print(encode(checked))
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--audit-seconds', type=float, default=60)
    parser.add_argument('--memory-gib', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if min(args.seconds, args.audit_seconds, args.memory_gib, args.workers) <= 0:
        parser.error('Resource limits must be positive')
    raise SystemExit(1 if run(args)['structural_contradictions'] else 0)
