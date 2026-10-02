"""Independent recursive heavy-map enumeration and archived R1-H audit."""
import argparse
from collections import Counter
from math import factorial,prod
import json
from pathlib import Path

from Experiment.Synister.audit_development import label,sha,bonds
from Experiment.Synister.development import digest,encoded,save
from synkit.Chem.Mapper.identifiability import parse_reaction


def enumerate_maps(r,p):
    def visit(prefix,used):
        if len(prefix)==len(r.atomic_numbers):
            yield tuple(prefix)
            return
        z=r.atomic_numbers[len(prefix)]
        for j,v in enumerate(p.atomic_numbers):
            if v==z and j not in used:
                yield from visit(prefix+[j],used|{j})
    yield from visit([],set())


def audit(directory,selection,protocol,primary):
    read=lambda p:json.loads(p.read_text())
    manifest=read(directory/'manifest.json')
    assert sha(selection)==manifest['selection_sha256']
    assert sha(protocol)==manifest['protocol_sha256']
    for name in ('accounting','selected','all_sources'):
        assert sha(directory/f'{name}.json')==manifest[name+'_sha256']
    eligible=[];accounting=[]
    for row in read(selection):
        r,p=parse_reaction(row['reaction'])
        count=prod(factorial(n) for n in Counter(r.atomic_numbers).values())
        balanced=sum(r.hcounts)==sum(p.hcounts)
        ok=balanced and count<=100000
        accounting.append({'original_id':row['original_id'],'compatible_maps':count,
                           'balanced_h':balanced,'eligible':ok})
        if ok:eligible.append(row)
    assert accounting==read(directory/'accounting.json')
    expected=sorted(eligible,key=lambda x:(digest(('synister-r1-hydrogen-v1\0'+x['original_id']).encode()),x['original_id']))[:30]
    assert expected==read(directory/'selected.json')
    assert len(eligible)==manifest['eligible'] and len(expected)==manifest['selected']
    inputs={x['reaction_id']:x for x in read(primary/'inputs.json')}
    results=read(directory/'results.json')
    assert len(results)==len(expected)
    tested_total=0;different=Counter()
    for row,result in zip(expected,results):
        assert result['original_id']==row['original_id'] and result['reaction']==row['reaction']
        r,p=parse_reaction(row['reaction'])
        best={'heavy':None,'combined':None};maps={k:set() for k in best};tested=0
        for m in enumerate_maps(r,p):
            edits=label(r,p,m)['typed_bond_edits']
            heavy=sum(abs(x[3]-x[2]) for x in edits)
            # Balanced pendant H: each unmatched parent attachment must move.
            retained=sum(min(r.hcounts[i],p.hcounts[m[i]]) for i in range(len(m)))
            combined=heavy+4*(sum(r.hcounts)-retained)
            for name,cost in (('heavy',heavy),('combined',combined)):
                if best[name] is None or cost<best[name]:best[name]=cost;maps[name]=set()
                if cost==best[name]:maps[name].add(m)
            tested+=1
        assert tested==result['compatible_maps_tested']
        assert best==result['minimum_doubled_cost'] and result['status']=='complete'
        labels={k:{encoded(label(r,p,m)) for m in values} for k,values in maps.items()}
        supports={k:{bonds(label(r,p,m)) for m in values} for k,values in maps.items()}
        for name in maps:
            assert maps[name]=={tuple(m) for m in result['optimizers'][name]}
            assert labels[name]=={encoded(x) for x in result['joint_labels'][name]}
        for field,sets in (('optimizer',maps),('joint_label',labels),('bond_label',supports)):
            equal=sets['heavy']==sets['combined']
            assert result[field+'_sets_equal']==equal
            different[field]+=not equal
        source=inputs[row['original_id']]
        primary_result=read(primary/'cases'/f'{source["case_id"]}.exact.json')
        assert source['reaction']==row['reaction'] and primary_result['status']=='complete'
        assert 2*primary_result['minimum']==best['heavy']
        assert {encoded(x['label']) for x in primary_result['joint_labels']}==labels['heavy']
        tested_total+=tested
    expected_summary={'attempts':len(results),'complete':len(results),'total_heavy_maps':tested_total,
                      **{'different_'+k+'_sets':v for k,v in different.items()}}
    assert read(directory/'summary.json')==expected_summary
    return {'status':'verified','scope':__doc__,'manifest_sha256':sha(directory/'manifest.json'),
            'results_sha256':sha(directory/'results.json'),'auditor_sha256':sha(Path(__file__)),
            'summary':expected_summary,'note':'Independent recursive enumeration and retained-H cost identity; strict endpoint parser and label extractor reused.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('directory','selection','protocol','primary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.directory,args.selection,args.protocol,args.primary)
    save(args.output,result)
    print(json.dumps(result,indent=2))
