"""Independent E2 indexed-set, subgroup-expansion and timing-accounting audit."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from Experiment.Synister.mapping_check import independent_map
from Experiment.Synister.classification_worker import verify_cyclic_group
from Experiment.Synister.classify_enumeration import digest, encode
from Experiment.Synister.benchmark_inputs import selection
from synkit.Chem.Mapper.identifiability import parse_reaction


def audit(directory, controls):
    from Experiment.Synister.audit_seed_output_recovery import verify_continuation
    continuation = verify_continuation(directory)
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in {**manifest['file_sha256'], **summary['files_sha256'], **summary['plans_sha256']}.items():
        if digest(directory/name) != expected:
            raise ValueError('E2 artifact changed: '+name)
    control = json.loads(controls.read_text())
    sources = json.loads((directory/'sources.json').read_text())
    if not control['all_passed']:
        raise ValueError('E2 controls failed')
    for name, expected in control['source_sha256'].items():
        if '/tests/' not in name and (name not in sources or sha256(sources[name].encode()).hexdigest() != expected):
            raise ValueError('Executed E2 source differs from small controls')
    if digest(Path(manifest['inputs_source'])) != manifest['inputs_source_sha256']:
        raise ValueError('Original 100-input selection changed')
    phase = manifest.get('phase', 'pilot')
    if phase == 'pilot':
        selected, _ = selection(json.loads(Path(manifest['inputs_source']).read_text()))
    else:
        parent = Path(manifest['parent'])
        for name, value in manifest['parent_sha256'].items():
            if digest(parent/name) != value:
                raise ValueError('Pilot provenance changed')
        selected = json.loads((parent/('difficult_inputs.json' if phase == 'followup' else 'extension_inputs.json')).read_text())
        if phase == 'followup':
            main = Path(manifest['main_extension'])
            if digest(main/'audit.json') != manifest['main_extension_audit_sha256']:
                raise ValueError('Completed 60-second extension provenance changed')
            main_audit = json.loads((main/'audit.json').read_text())
            if (main_audit['attempts'] != 320 or not main_audit['all_output_comparisons_consistent']
                    or main_audit['summary_sha256'] != digest(main/'summary.json')):
                raise ValueError('Follow-up lacks a completed audited extension')
    inputs = json.loads((directory/'inputs.json').read_text())
    if inputs != selected:
        raise ValueError('E2 pilot differs from the source/size input-only rule')
    by_input = {r['benchmark_id']: r for r in inputs}
    endpoints = {bid: parse_reaction(row['reaction']) for bid, row in by_input.items()}
    preparations = json.loads((directory/'preparation_tasks.json').read_text())
    prepared = {}
    for task in preparations:
        record = json.loads((directory/'preparation'/f"{task['task_id']}.json").read_text())
        if record['task'] != task or task['reaction'] != by_input[task['benchmark_id']]['reaction']:
            raise ValueError('Preparation task identity differs')
        if record.get('complete'):
            r, p = endpoints[task['benchmark_id']]
            if task['stage'] == 'seed':
                if independent_map(r, p, record['prediction']['mapping'])[0] != record['seed_doubled_cd']:
                    raise ValueError('Independently rescored seed differs')
            else:
                verify_cyclic_group(p, record['group'])
        prepared[task['task_id']] = record
    tasks = json.loads((directory/'tasks.json').read_text())
    paths = list((directory/'cases').glob('*.json'))
    attempts = 200 if phase == 'pilot' else len(inputs)*4
    if len(tasks) != attempts or len(paths) != attempts or {p.stem for p in paths} != {t['task_id'] for t in tasks}:
        raise ValueError('Missing, extra or duplicate E2 attempts')
    records, proofs, complete_sets, subsets = [], defaultdict(set), defaultdict(list), defaultdict(list)
    maps_checked = 0
    for task in tasks:
        key, bid = task['task_id'], task['benchmark_id']
        record = json.loads((directory/'cases'/f'{key}.json').read_text())
        if record['task'] != task or task['reaction'] != by_input[bid]['reaction']:
            raise ValueError('E2 task identity differs')
        if 'worker_sha256' in record and record['worker_sha256'] != sha256(sources['Experiment/Synister/seed_output_worker.py'].encode()).hexdigest():
            raise ValueError('Worker executed a different source version')
        seed = prepared[f'{bid}.seed']
        expected_seed = seed.get('prediction', {}).get('mapping') if task['seed_condition'] == 'slap' else None
        if task['initial_mapping'] != expected_seed:
            raise ValueError('Methods received different seed information')
        expected_seed_seconds = seed['parent_seconds'] if task['seed_condition'] == 'slap' else 0
        group_seconds = prepared[f'{bid}.group']['parent_seconds'] if task['stage'] == 'diagnostic' else 0
        if (task['seed_preparation_parent_seconds'] != expected_seed_seconds
                or record['seed_inclusive_parent_seconds'] != record['parent_seconds']+expected_seed_seconds+group_seconds
                or record['prediction_already_available_parent_seconds'] != record['parent_seconds']+group_seconds):
            raise ValueError('Preparation cost missing or counted incorrectly')
        if task['stage'] == 'diagnostic' and task['diagnostic_group'] != prepared[f'{bid}.group']['group']:
            raise ValueError('Output diagnostic group changed between requests')
        maps = set()
        if 'mapping_sha256' in record:
            path = directory/'maps'/f'{key}.json'
            data = path.read_bytes()
            saved = json.loads(data)
            maps = {tuple(m) for m in saved}
            if (sha256(data).hexdigest() != record['mapping_sha256'] or len(saved) != len(maps)
                    or len(maps) != record['mapping_count'] or len(data) != record['output_bytes']):
                raise ValueError('Requested output identity or count differs')
            r, p = endpoints[bid]
            if any(independent_map(r, p, m)[0] != record['minimum_doubled_cd'] for m in maps):
                raise ValueError('Independent minimum witness rescoring differs')
            maps_checked += len(maps)
        elif record['complete']:
            raise ValueError('Complete E2 attempt lacks output')
        if record.get('minimum_proved'):
            proofs[bid].add(record['minimum_doubled_cd'])
        if record.get('output_unit') == 'verified_cyclic_subgroup_representatives':
            group = verify_cyclic_group(endpoints[bid][1], record['group'])
            expanded = {tuple(g[j] for j in m) for m in maps for g in group}
            if len(expanded) != len(maps)*len(group):
                raise ValueError('Repeated representative or incomplete subgroup orbit')
            maps = expanded
        if record['complete'] and record.get('output_unit') != 'minimum_value_and_witness':
            complete_sets[bid].append((key, maps))
        subsets[bid].append((key, maps))
        records.append(record)
    if any(len(values) != 1 for values in proofs.values()):
        raise ValueError('Proved minima disagree between seeds/methods/output tasks')
    comparisons = 0
    for bid, outputs in complete_sets.items():
        baseline = outputs[0][1]
        if any(maps != baseline for _, maps in outputs):
            raise ValueError('Complete indexed sets disagree across methods/seeds/expanded diagnostics')
        for key, maps in subsets[bid]:
            if not maps <= baseline:
                raise ValueError('Partial output is outside the complete minimum set: '+key)
            comparisons += 1
    classifications = []
    by_task = {r['task']['task_id']: r for r in records}
    expected_aliases = {key for key, r in by_task.items()
                        if r['complete'] and r.get('output_unit') == 'all_indexed_atom_maps'}
    seen_aliases = set()
    for task in json.loads((directory/'classification_tasks.json').read_text()):
        record = json.loads((directory/'classification'/f"{task['classification_id']}.json").read_text())
        if record['task'] != task:
            raise ValueError('Classification task changed')
        if record.get('structural_audit', {}).get('consistent') is False:
            raise ValueError('Structural classification contradiction')
        if task['source_task_id'] not in task['aliases'] or not task['aliases']:
            raise ValueError('Classification lacks its source alias')
        for key in task['aliases']:
            if key not in expected_aliases or key in seen_aliases:
                raise ValueError('Missing, repeated or incomplete classification alias')
            original = by_task[key]
            if (original['mapping_sha256'] != task['mapping_sha256']
                    or original['mapping_count'] != task['mapping_count']
                    or original['minimum_doubled_cd'] != task['target_doubled_cd']
                    or original['task']['reaction'] != task['reaction']
                    or original['task']['benchmark_id'] != task['benchmark_id']):
                raise ValueError('Classification is attached to different output')
            seen_aliases.add(key)
        if 'detail_sha256' in record and digest(directory/'details'/f"{task['classification_id']}.json") != record['detail_sha256']:
            raise ValueError('Classification details changed')
        classifications.append(record)
    if seen_aliases != expected_aliases:
        raise ValueError('Complete indexed output lacks classification disposition')
    result = {'schema': 'synister.seed-output-audit.v1', 'all_output_comparisons_consistent': True,
        'selected': len(inputs), 'attempts': len(records), 'matched_attempts': sum(r['task']['stage'] == 'matched' for r in records),
        'diagnostic_attempts': sum(r['task']['stage'] == 'diagnostic' for r in records),
        'minimum_proved_inputs': len(proofs), 'inputs_with_complete_indexed_set': len(complete_sets),
        'independently_rescored_maps': maps_checked, 'complete_partial_comparisons': comparisons,
        'classifications': len(classifications), 'independently_verified_classifications': sum(r.get('structural_audit', {}).get('verified') is True for r in classifications),
        'classification_terminations': dict(Counter(r['termination'] for r in classifications)),
        'manifest_sha256': digest(directory/'manifest.json'), 'summary_sha256': digest(directory/'summary.json'),
        'controls_sha256': digest(controls), 'auditor_sha256': digest(Path(__file__))}
    if continuation is not None:
        result['continuation'] = continuation
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--controls', type=Path, default=Path('paper/synister/evidence/seed_output_controls_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.study, args.controls)
    with args.output.open('x') as stream:
        stream.write(encode(result))
    print(encode(result))
