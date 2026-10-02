"""R2 archived witness audit; orbit-engine replay, not independent search proof."""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from Experiment.Synister.audit_development import bonds, f1, is_automorphism, label, sha, transformed
from Experiment.Synister.development import digest, encoded, save
from Experiment.Synister.representation_sensitivity import kekule_endpoint, ordering_control
from Experiment.Synister.run_binary_sensitivity import select
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator
from synkit.Chem.Mapper.cohort_evaluation import paired_cohort_bounds


def score_replay(r,p,records,task,result):
    assert result['task_sha256'] == digest(encoded(task))
    if result['status'] != 'complete':
        return None
    candidates = {bonds(x['label']) for x in records}
    predictions = [bonds(label(r,p,task[f'prediction_{name}'])) for name in ('a','b')]
    engine = SupportOrbitEvaluator(r,time_limit_seconds=120)
    differences, keys = [],set()
    for y in candidates:
        orbit = engine.orbit(y)
        for image,g in orbit.items():
            assert is_automorphism(r,g) and transformed(y,g)==image
        keys.add(min(tuple(sorted(z)) for z in orbit))
        differences.append(max(f1(predictions[0],z) for z in orbit)-max(f1(predictions[1],z) for z in orbit))
    lo,hi = min(differences),max(differences)
    assert Fraction(result['width'])==hi-lo
    assert result['fixed_bond_labels']==len(candidates)
    assert result['bond_label_orbits']==len(keys)
    for end,value in (('lower',lo),('upper',hi)):
        witness = result[end]
        assert Fraction(witness['difference'])==value
        y = frozenset(tuple(x) for x in witness['label'])
        assert y in candidates
        for name,pred in zip(('a','b'),predictions):
            g = witness[f'{name}_transporter']
            assert is_automorphism(r,g)
            assert f1(pred,transformed(y,g))==Fraction(witness[f'{name}_score'])
            assert Fraction(witness[f'{name}_score'])==max(f1(pred,z) for z in engine.orbit(y))
    return lo,hi


def audit(directory,parent,selection,protocol):
    read = lambda path:json.loads(path.read_text())
    manifest = read(directory/'manifest.json')
    assert manifest['settings']=={'workers':4,'search_internal':60,'search_external':65,
        'score_internal':30,'score_external':35,'memory_gib':6,'numerical_threads':1,
        'emitted_map_cap':100000,'product_compression':False,'offsets':[1,2]}
    for path,field in ((parent/'manifest.json','parent_manifest_sha256'),
                       (parent/'audit.json','parent_audit_sha256'),
                       (selection,'parent_selection_sha256'),(protocol,'protocol_sha256')):
        assert sha(path)==manifest[field]
    for name in ('selection','search_tasks','parent_records','all_sources'):
        assert sha(directory/f'{name}.json')==manifest[name+'_sha256']
    selected = read(directory/'selection.json')
    assert selected==select(read(selection))[:30] and len(selected)==30
    for name,value in read(directory/'parent_records.json').items():
        assert sha(directory/'cases'/name)==value==sha(parent/'cases'/name)
    parent_inputs = {x['reaction_id']:x for x in read(parent/'inputs.json')}
    tasks = read(directory/'search_tasks.json')
    assert len(tasks)==90
    indexed = {(t['case_id'],t['stage']):t for t in tasks}
    assert len(indexed)==90
    scores = {s:{t['case_id']:t for t in read(directory/f'{s}_score_tasks.json')}
              for s in ('kekule','union')}
    unions = read(directory/'unions.json')
    expected_ids = {parent_inputs[x['original_id']]['case_id'] for x in selected}
    assert set(unions)==expected_ids
    expected_files = {f'{key}.{stage}.json' for key in expected_ids for stage in
                      ('slap','rxnmapper','exact','score','r2_kekule','r2_shell1','r2_shell2')}
    assert {p.name for p in (directory/'cases').iterdir()}==expected_files
    counts,rows = Counter(),[]
    intervals = {'kekule':[],'union':[]}
    expected_score_ids = {'kekule':set(),'union':set()}
    for row in selected:
        case = parent_inputs[row['original_id']]
        assert row['reaction']==case['reaction']
        key = case['case_id']
        r,p = parse_reaction(case['reaction'])
        original = read(directory/'cases'/f'{key}.exact.json')
        preds = {m:read(directory/'cases'/f'{key}.{m}.json') for m in ('slap','rxnmapper')}
        common = all(x['status']=='valid' for x in preds.values())
        counts['common_valid'] += common
        results = {}
        for stage in ('r2_kekule','r2_shell1','r2_shell2'):
            task = indexed[key,stage]
            assert all(task[k]==v for k,v in case.items() if k!='stage')
            assert task['search_seconds']==60
            result = read(directory/'cases'/f'{key}.{stage}.json')
            assert result['task_sha256']==digest(encoded(task))
            results[stage]=result
            counts[f'{stage}:{result["status"]}'] += 1
            if stage=='r2_kekule':
                assert task['prediction_mappings']==[x['prediction']['mapping'] for x in preds.values() if x['status']=='valid']
                left,right = case['reaction'].split('>>')
                _,a = kekule_endpoint(left)
                _,b = kekule_endpoint(right)
                if 'objective_endpoints' in result:
                    assert encoded(result['objective_endpoints'])==encoded([asdict(a),asdict(b)])
                    assert result['endpoint_changed']==[r!=a,p!=b]
                counts['aromatic_encoding_changed'] += (a!=r or b!=p)
                if 'ordering_controls' in result:
                    expected = [ordering_control(left),ordering_control(right)]
                    assert encoded(result['ordering_controls'])==encoded(expected)
                    counts['reverse_order_encoding_changed'] += any(x['encoding_changed'] for x in expected)
                target = result.get('minimum')
            else:
                a,b = r,p
                target = original['minimum']+int(stage[-1])
                assert task['target']==target
            if result['status']!='complete':
                continue
            assert result['enumeration_complete'] and result['labels_complete'] and result['joint_labels_complete']
            assert not result['symmetry_pruning']
            for witness in result['labels']+result['joint_labels']:
                assert label(r,p,witness['mapping'])==witness['label']
                edits = label(a,b,witness['mapping'])['typed_bond_edits']
                cost = Fraction(sum(abs(x[3]-x[2]) for x in edits),2)
                assert cost==target
            assert result['empty']==(not result['joint_labels'])
            counts[f'{stage}:empty'] += result['empty']
            old = {bonds(x['label']) for x in original['labels']}
            new = {bonds(x['label']) for x in result['labels']}
            counts[f'{stage}:same_bond_labels'] += old==new
        shell_complete = all(results[f'r2_shell{i}']['status']=='complete' for i in (1,2))
        union = unions[key]
        assert (union['status']=='complete')==shell_complete
        if shell_complete:
            expected = {bonds(x['label']) for source in (original,results['r2_shell1'],results['r2_shell2']) for x in source['labels']}
            assert {bonds(x['label']) for x in union['labels']}==expected
            for witness in union['labels']:
                assert label(r,p,witness['mapping'])==witness['label']
        else:
            assert not union['labels_complete'] and union['labels']==[]
        record = {'case_id':key,'reaction_id':case['reaction_id']}
        for scenario,result in (('kekule',results['r2_kekule']),('union',union)):
            value = None
            if common and result['status']=='complete':
                expected_score_ids[scenario].add(key)
                task = scores[scenario][key]
                assert task['labels']==result['labels'] and task['reaction']==case['reaction']
                assert task['score_seconds']==30 and task['score_backend']=='support-stabilizer'
                for method,name in (('slap','a'),('rxnmapper','b')):
                    assert task[f'prediction_{name}']==preds[method]['prediction']['mapping']
                score = read(directory/f'{scenario}_scores'/f'{key}.score.json')
                value = score_replay(r,p,result['labels'],task,score)
                counts[f'{scenario}_score:{score["status"]}'] += 1
            if common:
                intervals[scenario].append(value)
            record[scenario] = None if value is None else [str(x) for x in value]
        rows.append(record)
    for scenario in scores:
        assert set(scores[scenario])==expected_score_ids[scenario]
        assert {p.name for p in (directory/f'{scenario}_scores').iterdir()}=={f'{key}.score.json' for key in scores[scenario]}
    summary = read(directory/'summary.json')
    assert summary['selected']==30
    for stage in ('r2_kekule','r2_shell1','r2_shell2'):
        observed = Counter(read(directory/'cases'/f'{key}.{stage}.json')['status'] for key in expected_ids)
        assert summary['search_status'][stage]==dict(observed)
    assert summary['union_status']==dict(Counter(x['status'] for x in unions.values()))
    for scenario in scores:
        observed = Counter(read(directory/f'{scenario}_scores'/f'{key}.score.json')['status'] for key in scores[scenario])
        assert summary['score_status'][scenario]==dict(observed)
    bounds = {s:{k:str(v) if isinstance(v,Fraction) else v for k,v in asdict(paired_cohort_bounds(values)).items()}
              for s,values in intervals.items()}
    return {'scope':__doc__,'manifest_sha256':sha(directory/'manifest.json'),
            'summary_sha256':sha(directory/'summary.json'),'auditor_sha256':sha(Path(__file__)),
            'counts':dict(counts),'bounds':bounds,'rows':rows}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ('directory','parent','selection','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.directory,args.parent,args.selection,args.protocol)
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
