"""Prespecified E2 pilot; isolated source snapshots keep long runs reproducible."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from Experiment.Synister.classify_enumeration import digest, encode
from Experiment.Synister.benchmark_inputs import selection


PROTOCOL = {
    'schema': 'synister.seed-output-pilot.v1',
    'selection': 'Two smallest input hashes per source/size bin of the original 100 reactions; 20 inputs, no outcome selection',
    'seed': 'Fresh SLAP heavy-atom prediction from unmapped endpoints; one shared verified map per input. Never a reference or supplied optimum.',
    'seed_conditions': ['none', 'slap'],
    'matched_methods': ['synister', 'milp'],
    'matched_output': 'All indexed minimum maps; native production Synister expansion and independent all-solutions MILP',
    'milp_seed': 'Feasible non-strict cost upper-bound inequality; not a native warm start, no fixed atom assignments',
    'synister_seed': 'Feasible incumbent and traversal preference, never fixed atom assignments',
    'diagnostics': 'Three fresh Synister runs per seed condition: proof only, representatives, externally expanded indexed set. One independently prepared verified cyclic group per input, identical across six runs; scoped group-provider replacement, not native performance data.',
    'accounting': '60s search excludes prediction preparation; 75s search-process parent cap. Charge full seed-preparation parent time in seeded end-to-end scenarios. Report prediction-already-available separately. Group preparation charged to diagnostic end-to-end. Classification and writing separate. Nested timings are not summed twice.',
    'seed_limits': '75s parent, 6 GiB, one thread; failures retained without substitution',
    'search_seconds': 60, 'parent_seconds': 75, 'memory_gib': 6, 'max_maps': 100000,
    'classification_seconds': 30, 'audit_seconds': 30, 'workers': 4, 'threads_per_worker': 1,
    'matched_attempts': 80, 'diagnostic_attempts': 120,
    'extension': 'After interface/accounting gate, 80 additional preselected inputs give 400 matched attempts with pilot; pilot diagnostics not repeated',
    'difficult_followup': 'Before outcomes: five largest inputs per source in each of bins 61-80 and >80, hash ties; 20 inputs. Fresh equal 300s searches for both methods and seed conditions; do not extend only successes.',
    'classification': 'One independent ITS classification and audit per unique complete indexed set, reused across identical task outputs with explicit aliases and measured cost; no effect on original enumeration completion',
    'scope': 'Explain input/seed/output effects and failures; no broad superiority or chemical-accuracy claim',
}


def save(path, value):
    with path.open('x') as stream:
        stream.write(encode(value))


def freeze(output, inputs_path):
    inputs = json.loads(inputs_path.read_text())
    pilot, _ = selection(inputs)
    chosen = {r['benchmark_id'] for r in pilot}
    extension = [r for r in inputs if r['benchmark_id'] not in chosen]
    difficult = []
    for source in ('FlowER', 'Rhea'):
        for size in (3, 4):
            difficult += sorted([r for r in inputs if r['source'] == source and r['size_bin'] == size],
                                key=lambda r: (-r['atoms'], r['selection_sha256']))[:5]
    if (len(pilot), len(extension), len(difficult)) != (20, 80, 20):
        raise ValueError('Prespecified selection denominators differ')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('cases', 'maps', 'preparation', 'classification', 'details', 'frozen_source'):
        (output/name).mkdir()
    save(output/'protocol.json', PROTOCOL)
    save(output/'inputs.json', pilot)
    save(output/'extension_inputs.json', extension)
    save(output/'difficult_inputs.json', difficult)
    paths = sorted(Path('synkit').rglob('*.py'))+sorted(Path('Experiment/Synister').glob('*.py'))
    for name in ('Experiment/__init__.py',):
        if Path(name).exists():
            paths.append(Path(name))
    sources = {str(path): path.read_text() for path in paths}
    save(output/'sources.json', sources)
    for name, content in sources.items():
        path = output/'frozen_source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    save(output/'manifest.json', {'schema': PROTOCOL['schema'], 'python': sys.version, 'platform': platform.platform(),
        'dependencies': {name: version(name) for name in ('numpy', 'scipy', 'rdkit', 'networkx')},
        'inputs_source': str(inputs_path), 'inputs_source_sha256': digest(inputs_path),
        'file_sha256': {name: digest(output/name) for name in ('protocol.json', 'inputs.json', 'extension_inputs.json', 'difficult_inputs.json', 'sources.json')},
        'execution': 'Workers execute with cwd at the saved source tree, independent of later worktree edits'})
    return pilot


def isolated(output, module, task, parent_seconds):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYTHONHASHSEED='0', PYTHONPATH=str((output/'frozen_source').resolve()))
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, '-m', 'Experiment.Synister.'+module], input=encode(task),
            capture_output=True, text=True, env=env, cwd=output/'frozen_source', timeout=parent_seconds)
        record = json.loads(process.stdout) if process.returncode == 0 else {
            'complete': False, 'minimum_proved': False, 'termination': 'process_failure', 'returncode': process.returncode}
        record['stderr'] = process.stderr[-4000:]
    except subprocess.TimeoutExpired:
        record = {'complete': False, 'minimum_proved': False, 'termination': 'external_time_limit'}
    record['parent_seconds'] = time.perf_counter()-started
    return record


def execute(output, task, *, preparation=False):
    runtime = dict(task)
    if not preparation:
        runtime['map_path'] = str((output/'maps'/f"{task['task_id']}.json").resolve())
    result = isolated(output, 'seed_output_worker', runtime, 75 if preparation else task['seconds']+15)
    result['task'] = task
    if not preparation:
        seed = task.get('seed_preparation_parent_seconds', 0)
        group = task.get('group_preparation_parent_seconds', 0)
        result.update(prediction_already_available_parent_seconds=result['parent_seconds']+group,
                      seed_inclusive_parent_seconds=result['parent_seconds']+seed+group)
    save(output/('preparation' if preparation else 'cases')/f"{task['task_id']}.json", result)
    print(json.dumps({'task': task['task_id'], 'termination': result['termination'],
                      'complete': result['complete'], 'parent_seconds': result['parent_seconds']}), flush=True)
    return result


def classify(output, task):
    key = task['classification_id']
    runtime = {**task, 'detail_path': str((output/'details'/f'{key}.json').resolve()),
               'seconds': 30, 'audit_seconds': 30, 'memory_gib': 6}
    result = isolated(output, 'classification_worker', runtime, 45)
    if 'detail_sha256' in result:
        result['structural_audit'] = isolated(output, 'audit_classification', {**runtime, 'result': result}, 45)
    result['task'] = task
    save(output/'classification'/f'{key}.json', result)
    print(json.dumps({'classification': key, 'termination': result['termination']}), flush=True)
    return result


def run(output, inputs_path):
    rows = freeze(output, inputs_path)
    preparations = [{'task_id': f"{r['benchmark_id']}.{stage}", 'benchmark_id': r['benchmark_id'],
                     'reaction': r['reaction'], 'stage': stage, 'memory_gib': 6}
                    for r in rows for stage in ('seed', 'group')]
    save(output/'preparation_tasks.json', preparations)
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(lambda t: execute(output, t, preparation=True), preparations))
    by_preparation = {r['task']['task_id']: r for r in prepared}
    tasks = []
    for index, row in enumerate(rows):
        bid = row['benchmark_id']
        seed_record, group_record = (by_preparation[f'{bid}.{name}'] for name in ('seed', 'group'))
        for condition in (('none', 'slap') if index % 2 == 0 else ('slap', 'none')):
            base = {'benchmark_id': bid, 'reaction': row['reaction'], 'seed_condition': condition,
                    'initial_mapping': seed_record.get('prediction', {}).get('mapping') if condition == 'slap' else None,
                    'seed_preparation_parent_seconds': seed_record['parent_seconds'] if condition == 'slap' else 0,
                    'seconds': 60, 'max_maps': 100000, 'memory_gib': 6}
            for method in (('synister', 'milp') if index % 2 == 0 else ('milp', 'synister')):
                tasks.append({**base, 'stage': 'matched', 'method': method, 'output': 'indexed',
                              'task_id': f'{bid}.{condition}.{method}.indexed'})
            if not group_record.get('complete'):
                raise ValueError('Group preparation failed; keep outputs and investigate before search')
            for unit in ('proof', 'representatives', 'indexed'):
                tasks.append({**base, 'stage': 'diagnostic', 'method': 'synister', 'output': unit,
                    'diagnostic_group': group_record['group'], 'group_preparation_parent_seconds': group_record['parent_seconds'],
                    'task_id': f'{bid}.{condition}.diagnostic.{unit}'})
    save(output/'tasks.json', tasks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda t: execute(output, t), tasks))
    grouped = defaultdict(list)
    for record in records:
        if record['complete'] and record.get('output_unit') == 'all_indexed_atom_maps':
            grouped[record['task']['benchmark_id'], record['mapping_sha256']].append(record)
    classification_tasks = []
    for (bid, identity), aliases in sorted(grouped.items()):
        chosen = min(aliases, key=lambda r: r['task']['task_id'])
        classification_tasks.append({'classification_id': bid+'.'+identity[:16], 'reaction': chosen['task']['reaction'],
            'benchmark_id': bid, 'mapping_count': chosen['mapping_count'], 'mapping_sha256': identity,
            'target_doubled_cd': chosen['minimum_doubled_cd'],
            'map_path': str((output/'maps'/f"{chosen['task']['task_id']}.json").resolve()),
            'source_task_id': chosen['task']['task_id'], 'aliases': [r['task']['task_id'] for r in aliases]})
    save(output/'classification_tasks.json', classification_tasks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        classifications = list(pool.map(lambda t: classify(output, t), classification_tasks))
    summary = {'selected': len(rows), 'preparation_attempts': len(prepared), 'attempts': len(records),
        'matched_attempts': sum(r['task']['stage'] == 'matched' for r in records),
        'diagnostic_attempts': sum(r['task']['stage'] == 'diagnostic' for r in records),
        'seed_preparation': dict(Counter(r['termination'] for r in prepared if r['task']['stage'] == 'seed')),
        'outcomes': {f'{stage}.{condition}.{method}.{unit}': dict(Counter(r['termination'] for r in records
            if (r['task']['stage'], r['task']['seed_condition'], r['task']['method'], r['task']['output']) == (stage, condition, method, unit)))
            for stage, condition, method, unit in sorted({(r['task']['stage'], r['task']['seed_condition'], r['task']['method'], r['task']['output']) for r in records})},
        'classification_terminations': dict(Counter(r['termination'] for r in classifications)),
        'structural_contradictions': sum(r.get('structural_audit', {}).get('consistent') is False for r in classifications),
        'files_sha256': {str(path.relative_to(output)): digest(path) for folder in ('cases','maps','preparation','classification','details')
                         for path in sorted((output/folder).glob('*.json'))},
        'plans_sha256': {name: digest(output/name) for name in ('preparation_tasks.json','tasks.json','classification_tasks.json')}}
    save(output/'summary.json', summary)
    print(encode({k: v for k, v in summary.items() if k not in ('files_sha256','plans_sha256')}), flush=True)
    return summary


def extend(output, parent, *, followup=False, main_extension=None):
    """Continue the input-fixed study using exactly the pilot worker sources."""
    audited = json.loads((parent/'audit.json').read_text())
    if (not audited['all_output_comparisons_consistent'] or audited['attempts'] != 200
            or audited['summary_sha256'] != digest(parent/'summary.json')):
        raise ValueError('Pilot output/accounting gate not satisfied')
    if followup:
        if main_extension is None:
            raise ValueError('300-second follow-up requires the completed 60-second extension')
        main_audit = json.loads((main_extension/'audit.json').read_text())
        if (not main_audit['all_output_comparisons_consistent'] or main_audit['attempts'] != 320
                or main_audit['summary_sha256'] != digest(main_extension/'summary.json')):
            raise ValueError('100-input 60-second study is not audited and complete')
    phase = 'followup' if followup else 'extension'
    seconds = 300 if followup else 60
    selection_name = 'difficult_inputs.json' if followup else 'extension_inputs.json'
    rows = json.loads((parent/selection_name).read_text())
    expected_count = 20 if followup else 80
    if len(rows) != expected_count:
        raise ValueError('Preselected extension denominator differs')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('cases', 'maps', 'preparation', 'classification', 'details', 'frozen_source'):
        (output/name).mkdir()
    protocol = {**PROTOCOL, 'schema': 'synister.seed-output-'+phase+'.v1', 'phase': phase,
        'selection': 'Exactly '+selection_name+' saved before pilot outcomes; no new selection',
        'search_seconds': seconds, 'parent_seconds': seconds+15,
        'accounting': f'{seconds}s search excludes prediction preparation; {seconds+15}s search-process parent cap. Charge full seed-preparation parent time in seeded end-to-end scenarios. Report prediction-already-available separately. Classification and writing separate. Nested timings are not summed twice.',
        'diagnostics': 'Native indexed-output comparison only; the separate output-request diagnostics belong to the twenty-input pilot.',
        'matched_attempts': expected_count*4, 'diagnostic_attempts': 0,
        'pilot_gate': 'All 200 pilot attempts accounted and complete/partial set comparisons consistent; incomplete classifications retained'}
    save(output/'protocol.json', protocol)
    save(output/'inputs.json', rows)
    sources = json.loads((parent/'sources.json').read_text())
    save(output/'sources.json', sources)
    for name, content in sources.items():
        path = output/'frozen_source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    save(output/'orchestrator_source.json', {str(Path(__file__)): Path(__file__).read_text()})
    old_manifest = json.loads((parent/'manifest.json').read_text())
    manifest = {**old_manifest, 'schema': protocol['schema'], 'phase': phase,
        'parent': str(parent), 'parent_sha256': {name: digest(parent/name) for name in ('summary.json','audit.json',selection_name,'sources.json')},
        'file_sha256': {name: digest(output/name) for name in ('protocol.json','inputs.json','sources.json','orchestrator_source.json')}}
    if followup:
        manifest.update(main_extension=str(main_extension), main_extension_audit_sha256=digest(main_extension/'audit.json'))
    save(output/'manifest.json', manifest)
    preparations = [{'task_id': f"{r['benchmark_id']}.seed", 'benchmark_id': r['benchmark_id'],
                     'reaction': r['reaction'], 'stage': 'seed', 'memory_gib': 6} for r in rows]
    save(output/'preparation_tasks.json', preparations)
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(lambda t: execute(output, t, preparation=True), preparations))
    seeds = {r['task']['benchmark_id']: r for r in prepared}
    tasks = []
    for index, row in enumerate(rows):
        bid = row['benchmark_id']
        seed = seeds[bid]
        for condition in (('none','slap') if index % 2 == 0 else ('slap','none')):
            for method in (('synister','milp') if index % 2 == 0 else ('milp','synister')):
                tasks.append({'task_id': f'{bid}.{condition}.{method}.indexed', 'benchmark_id': bid,
                    'reaction': row['reaction'], 'stage': 'matched', 'method': method, 'output': 'indexed',
                    'seed_condition': condition, 'initial_mapping': seed.get('prediction',{}).get('mapping') if condition == 'slap' else None,
                    'seed_preparation_parent_seconds': seed['parent_seconds'] if condition == 'slap' else 0,
                    'seconds': seconds, 'max_maps': 100000, 'memory_gib': 6})
    save(output/'tasks.json', tasks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda t: execute(output, t), tasks))
    grouped = defaultdict(list)
    for record in records:
        if record['complete']:
            grouped[record['task']['benchmark_id'], record['mapping_sha256']].append(record)
    classification_tasks = []
    for (bid, identity), aliases in sorted(grouped.items()):
        chosen = min(aliases, key=lambda r: r['task']['task_id'])
        classification_tasks.append({'classification_id': bid+'.'+identity[:16], 'reaction': chosen['task']['reaction'],
            'benchmark_id': bid, 'mapping_count': chosen['mapping_count'], 'mapping_sha256': identity,
            'target_doubled_cd': chosen['minimum_doubled_cd'],
            'map_path': str((output/'maps'/f"{chosen['task']['task_id']}.json").resolve()),
            'source_task_id': chosen['task']['task_id'], 'aliases': [r['task']['task_id'] for r in aliases]})
    save(output/'classification_tasks.json', classification_tasks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        classifications = list(pool.map(lambda t: classify(output, t), classification_tasks))
    summary = {'phase': phase, 'selected': len(rows), 'preparation_attempts': len(prepared),
        'attempts': len(records), 'matched_attempts': len(records), 'diagnostic_attempts': 0,
        'seed_preparation': dict(Counter(r['termination'] for r in prepared)),
        'classification_terminations': dict(Counter(r['termination'] for r in classifications)),
        'structural_contradictions': sum(r.get('structural_audit',{}).get('consistent') is False for r in classifications),
        'files_sha256': {str(path.relative_to(output)): digest(path) for folder in ('cases','maps','preparation','classification','details')
                         for path in sorted((output/folder).glob('*.json'))},
        'plans_sha256': {name: digest(output/name) for name in ('preparation_tasks.json','tasks.json','classification_tasks.json')}}
    save(output/'summary.json', summary)
    print(encode({k:v for k,v in summary.items() if k not in ('files_sha256','plans_sha256')}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=Path('paper/synister/evidence/enumeration_main_v1/inputs.json'))
    parser.add_argument('--phase', choices=('pilot','extension','followup'), default='pilot')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--main-extension', type=Path)
    args = parser.parse_args()
    if args.phase == 'pilot':
        run(args.output, args.inputs)
    else:
        if args.parent is None:
            parser.error('Extension/follow-up requires --parent pilot directory')
        extend(args.output, args.parent, followup=args.phase == 'followup', main_extension=args.main_extension)
