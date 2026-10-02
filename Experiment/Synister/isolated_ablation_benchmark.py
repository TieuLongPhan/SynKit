"""Forty input-fixed E4 component interventions, with separate instrumentation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

from Experiment.Synister.ablation_benchmark import derive_tasks, compare_records
from Experiment.Synister.classify_enumeration import digest, encode
from Experiment.Synister.isolated_ablation import compile_variant, VARIANTS
from Experiment.Synister.seed_output_benchmark import isolated


def validate_controls(controls):
    summary = json.loads((controls/'summary.json').read_text())
    if not summary['all_passed'] or summary['queries'] != 2814:
        raise ValueError('Isolated-ablation literal gate did not pass')
    for name, expected in summary['files_sha256'].items():
        if digest(controls/name) != expected:
            raise ValueError('Literal control artifact changed')
    sources = json.loads((controls/'sources.json').read_text())
    for name, content in sources.items():
        # The control writer changed only its JSON sequence serialization after
        # this archive. Its saved source is retained; runtime source must match.
        if name != 'Experiment/Synister/validate_isolated_ablations.py' and Path(name).read_text() != content:
            raise ValueError('Experimental runtime differs from literal controls: '+name)
    variants = json.loads((controls/'variants.json').read_text())
    for variant in VARIANTS:
        if compile_variant(variant)[1] != variants[variant]:
            raise ValueError('Transformed experimental solver changed')
    return summary


def execute(output, task):
    runtime = {**task,'map_path':str((output/'maps'/(task['task_id']+'.json')).resolve())}
    result = isolated(output,'isolated_ablation_worker',runtime,task['seconds']+15)
    result['task'] = task
    (output/'cases'/(task['task_id']+'.json')).write_text(encode(result))
    print(json.dumps({'task':task['task_id'],'complete':result['complete'],'termination':result['termination']}),flush=True)
    return result


def run(output, controls, *, instrumented=False, ordinary=None):
    validate_controls(controls)
    if instrumented:
        if ordinary is None:
            raise ValueError('Instrumentation follows an audited ordinary study')
        audit = json.loads((ordinary/'audit.json').read_text())
        if not audit['all_output_comparisons_consistent'] or audit['summary_sha256'] != digest(ordinary/'summary.json'):
            raise ValueError('Ordinary study has not passed its output audit')
    source = Path('paper/synister/evidence/ablation_main_v1')
    rows = json.loads((source/'inputs.json').read_text())
    old_manifest = json.loads((source/'manifest.json').read_text())
    if len(rows) != 40 or digest(source/'inputs.json') != old_manifest['file_sha256']['inputs.json']:
        raise ValueError('Existing forty-input selection changed')
    seconds = 5 if instrumented else 60
    matched = Path('paper/synister/evidence/enumeration_main_v1')
    tasks, unavailable = derive_tasks(rows,matched,seconds=seconds,memory_gib=6,max_maps=100000,variants=tuple(VARIANTS))
    tasks = [{**task,'instrumented':instrumented} for task in tasks]
    output.mkdir(parents=True,exist_ok=False)
    for name in ('cases','maps','frozen_source'):
        (output/name).mkdir()
    paths = sorted(Path('synkit').rglob('*.py'))+sorted(Path('Experiment/Synister').glob('*.py'))
    if Path('Experiment/__init__.py').exists():
        paths.append(Path('Experiment/__init__.py'))
    sources = {str(p):p.read_text() for p in paths}
    for name, content in sources.items():
        path = output/'frozen_source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
    artifacts = {'inputs.json':rows,'tasks.json':tasks,'unavailable_queries.json':unavailable,
                 'sources.json':sources,'variants.json':{v:compile_variant(v)[1] for v in VARIANTS}}
    for name,value in artifacts.items():
        (output/name).write_text(encode(value))
    manifest = {'schema':'synister.isolated-ablations.v1','selected':40,'attempts':len(tasks),
        'seconds':seconds,'external_seconds':seconds+15,'memory_gib':6,'max_maps':100000,
        'workers':4,'threads_per_worker':1,'variants':list(VARIANTS),'repeats':1,
        'instrumented':instrumented,'ordinary':str(ordinary) if ordinary else None,
        'ordinary_audit_sha256':digest(ordinary/'audit.json') if ordinary else None,
        'selection_source':str(source),'selection_sha256':digest(source/'inputs.json'),
        'literal_controls':str(controls),'literal_controls_sha256':digest(controls/'summary.json'),
        'python':sys.version,'platform':platform.platform(),
        'dependencies':{name:version(name) for name in ('numpy','scipy','networkx','rdkit')},
        'seed':'Existing deterministic feasible incident-histogram LAP and two improving swap sweeps, recomputed and charged per attempt.',
        'output_unit':'all_indexed_atom_maps','scope':VARIANTS,
        'timing_scope':'Ordinary parent/search times are headline observations. Separate five-second instrumented runs describe visited work only; their overlapping timers include trace overhead and are never speed comparisons.',
        'file_sha256':{name:digest(output/name) for name in artifacts}}
    (output/'manifest.json').write_text(encode(manifest))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda task:execute(output,task),tasks))
    comparisons = compare_records(records,output)
    summary = {'selected':40,'attempts':len(records),'instrumented':instrumented,
        'by_variant':{variant:dict(Counter('complete' if r['complete'] else r['termination']
            for r in records if r['task']['variant']==variant)) for variant in VARIANTS},
        'all_output_comparisons_consistent':all(c['consistent'] for c in comparisons),
        'comparisons':comparisons,'all_record_hashes':{p.name:digest(p) for p in sorted((output/'cases').glob('*.json'))}}
    (output/'summary.json').write_text(encode(summary))
    print(encode({k:v for k,v in summary.items() if k not in ('comparisons','all_record_hashes')}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controls',type=Path,default=Path('paper/synister/evidence/isolated_ablation_controls_v1'))
    parser.add_argument('--instrumented',action='store_true')
    parser.add_argument('--ordinary',type=Path)
    args = parser.parse_args()
    run(args.output,args.controls,instrumented=args.instrumented,ordinary=args.ordinary)
