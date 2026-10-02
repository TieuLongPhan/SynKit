"""Post-primary fixed-coordinate diagnostic; never substitutes for aligned scores."""
import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from Experiment.Synister.audit_development import bonds,f1,label,sha
from Experiment.Synister.development import save
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.cohort_evaluation import paired_cohort_bounds


def report(directory):
    read=lambda p:json.loads(p.read_text())
    audit=read(directory/'audit.json')
    assert sha(directory/'manifest.json')==audit['manifest_sha256']
    assert sha(directory/'summary.json')==audit['summary_sha256']
    intervals=[];rows=[];bindings={}
    for case in read(directory/'inputs.json'):
        key=case['case_id']
        def record(stage):
            path=directory/'cases'/f'{key}.{stage}.json'
            bindings[path.name]=sha(path)
            return read(path)
        predictions=[record(m) for m in ('slap','rxnmapper')]
        if not all(x['status']=='valid' for x in predictions):
            rows.append({'case_id':key,'status':'invalid_prediction'})
            continue
        exact=record('exact')
        if exact['status']!='complete' or not exact['labels_complete']:
            intervals.append(None);rows.append({'case_id':key,'status':'unresolved_search'})
            continue
        r,p=parse_reaction(case['reaction'])
        supports=[bonds(label(r,p,x['prediction']['mapping'])) for x in predictions]
        scored=[]
        for candidate in exact['labels']:
            assert label(r,p,candidate['mapping'])==candidate['label']
            target=bonds(candidate['label'])
            values=[f1(pred,target) for pred in supports]
            scored.append({'difference':str(values[0]-values[1]),'scores':list(map(str,values)),
                           'mapping':candidate['mapping'],'label':candidate['label']})
        lo=min(scored,key=lambda x:Fraction(x['difference']))
        hi=max(scored,key=lambda x:Fraction(x['difference']))
        intervals.append((Fraction(lo['difference']),Fraction(hi['difference'])))
        aligned=record('score')
        rows.append({'case_id':key,'status':'complete','lower':lo,'upper':hi,
                     'aligned_status':aligned['status'],
                     'aligned_interval':None if aligned['status']!='complete' else
                     [aligned[e]['difference'] for e in ('lower','upper')]})
    bounds={k:str(v) if isinstance(v,Fraction) else v for k,v in asdict(paired_cohort_bounds(intervals)).items()}
    return {'scope':'Post-primary fixed-coordinate diagnostic; not a replacement primary endpoint or confirmation test',
            'primary_audit_sha256':sha(directory/'audit.json'),'reporter_sha256':sha(Path(__file__)),
            'common_valid':len(intervals),'resolved':sum(x is not None for x in intervals),
            'changed_interval_on_jointly_resolved':sum(row['status']=='complete' and row['aligned_interval'] is not None and
                 row['aligned_interval']!=[row[e]['difference'] for e in ('lower','upper')] for row in rows),
            'bounds':bounds,'record_hashes':bindings,'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=report(args.directory);save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','record_hashes')},indent=2))
