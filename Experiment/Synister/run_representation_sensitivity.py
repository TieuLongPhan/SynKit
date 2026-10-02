"""Run the fixed 30-case exploratory R2 protocol without replacing failures."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from Experiment.Synister.audit_development import sha
from Experiment.Synister.confirmation_contract import source_contents
from Experiment.Synister.development import digest, encoded, execute, save
from Experiment.Synister.freeze_environment import closure, ROOTS
from Experiment.Synister.representation_sensitivity import union_labels
from Experiment.Synister.run_binary_sensitivity import select


def run(parent, selection, protocol, output):
    read = lambda p: json.loads(p.read_text())
    assert sha(protocol) == '441c869a5ab2b5efa7f39dbefa59439cca0e7547bcdba93ed2dcbdd9be35b16d'
    assert sha(selection) == 'd01c56e6eaab47232cdb2feefe45fac20c894683b739a66b2d8467e1423bf363'
    audit, manifest = read(parent/'audit.json'), read(parent/'manifest.json')
    assert sha(parent/'manifest.json') == audit['manifest_sha256']
    assert sha(parent/'summary.json') == audit['summary_sha256']
    assert sha(parent/'inputs.json') == manifest['inputs_sha256']
    inputs = {r['reaction_id']:r for r in read(parent/'inputs.json')}
    selected = select(read(selection))[:30]
    assert len(selected) == 30
    freeze = read(parent/'prediction_freeze.json')
    output.mkdir(parents=True,exist_ok=False)
    records = output/'cases'
    records.mkdir()
    tasks, bindings, predictions, originals, cases = [], {}, {}, {}, {}
    for row in selected:
        case = inputs[row['original_id']]
        assert row['reaction'] == case['reaction']
        key = case['case_id']
        cases[key] = case
        preds = {}
        for stage in ('slap','rxnmapper','exact','score'):
            name = f'{key}.{stage}.json'
            path = parent/'cases'/name
            bindings[name] = sha(path)
            shutil.copyfile(path,records/name)
            if stage in ('slap','rxnmapper'):
                assert sha(path) == freeze[f'{key}.{stage}']
                preds[stage] = read(path)
        original = read(records/f'{key}.exact.json')
        assert original['status'] == 'complete' and original['labels_complete']
        originals[key], predictions[key] = original, preds
        tasks.append(dict(case,stage='r2_kekule',search_seconds=60.0,
                          prediction_mappings=[x['prediction']['mapping'] for x in preds.values()
                                               if x['status']=='valid']))
        for offset in (1,2):
            tasks.append(dict(case,stage=f'r2_shell{offset}',search_seconds=60.0,
                              target=original['minimum']+offset))
    for name,value in (('selection',selected),('search_tasks',tasks),('parent_records',bindings),
                       ('all_sources',source_contents())):
        save(output/f'{name}.json',value)
    source_hash = sha(output/'all_sources.json')
    save(output/'manifest.json',{
        'scope':'post-C1/R1 exploratory R2 representation and near-minimum study',
        'protocol_sha256':sha(protocol),'parent_manifest_sha256':sha(parent/'manifest.json'),
        'parent_audit_sha256':sha(parent/'audit.json'),'parent_selection_sha256':sha(selection),
        **{name+'_sha256':sha(output/f'{name}.json') for name in
           ('selection','search_tasks','parent_records','all_sources')},
        'packages':closure(ROOTS),'settings':{'workers':4,'search_internal':60,'search_external':65,
        'score_internal':30,'score_external':35,'memory_gib':6,'numerical_threads':1,
        'emitted_map_cap':100000,'product_compression':False,'offsets':[1,2]},
        'policy':'original predictions/labels; retain all attempts; no adaptive retries'})
    def check_sources():
        assert digest(encoded(source_contents())) == source_hash
    with ThreadPoolExecutor(max_workers=4) as pool:
        exact = list(pool.map(lambda t:execute(t,65,records),tasks))
        check_sources()
        indexed = {(t['case_id'],t['stage']):r for t,r in zip(tasks,exact)}
        unions = {key:union_labels(originals[key],
                  [indexed[key,f'r2_shell{i}'] for i in (1,2)]) for key in cases}
        save(output/'unions.json',unions)
        score_status = {}
        for scenario in ('kekule','union'):
            score_dir = output/f'{scenario}_scores'
            score_dir.mkdir()
            jobs = []
            for key,case in cases.items():
                result = indexed[key,'r2_kekule'] if scenario=='kekule' else unions[key]
                preds = predictions[key]
                if result['status']=='complete' and all(x['status']=='valid' for x in preds.values()):
                    jobs.append(dict(case,stage='score',score_seconds=30.0,score_backend='support-stabilizer',
                                     labels=result['labels'],prediction_a=preds['slap']['prediction']['mapping'],
                                     prediction_b=preds['rxnmapper']['prediction']['mapping']))
            save(output/f'{scenario}_score_tasks.json',jobs)
            scores = list(pool.map(lambda t:execute(t,35,score_dir),jobs))
            score_status[scenario] = dict(Counter(x['status'] for x in scores))
            check_sources()
    assert all(sha(records/name)==value for name,value in bindings.items())
    summary = {'selected':30,'search_status':{stage:dict(Counter(r['status'] for t,r in zip(tasks,exact)
                if t['stage']==stage)) for stage in ('r2_kekule','r2_shell1','r2_shell2')},
               'union_status':dict(Counter(x['status'] for x in unions.values())),
               'score_status':score_status,'scope':'execution accounting; scientific audit pending'}
    save(output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ('parent','selection','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(args.parent,args.selection,args.protocol,args.output)
