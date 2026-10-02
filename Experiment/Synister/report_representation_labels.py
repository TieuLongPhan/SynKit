"""Descriptive R2 label overlap and audited bond-orbit transitions."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from Experiment.Synister.audit_development import bonds,sha
from Experiment.Synister.development import encoded,save


def overlap(a,b):
    return {'original':len(a),'alternative':len(b),'intersection':len(a&b),
            'jaccard':str(Fraction(len(a&b),len(a|b))) if a|b else '1',
            'equal':a==b}


def report(directory):
    read=lambda p:json.loads(p.read_text())
    audit=read(directory/'audit_v2.json')
    assert audit['manifest_sha256']==sha(directory/'manifest.json')
    assert audit['summary_sha256']==sha(directory/'summary.json')
    tasks=read(directory/'search_tasks.json')
    rows=[];counts=Counter();bindings={}
    def record(path):
        bindings[str(path.relative_to(directory))]=sha(path)
        return read(path)
    for task in tasks:
        if task['stage']!='r2_kekule':continue
        key=task['case_id']
        old=record(directory/'cases'/f'{key}.exact.json')
        oldscore=record(directory/'cases'/f'{key}.score.json')
        row={'case_id':key,'reaction_id':task['reaction_id'],'weighted_minimum':old['minimum']}
        for stage in ('r2_kekule','r2_shell1','r2_shell2'):
            result=record(directory/'cases'/f'{key}.{stage}.json')
            entry={'status':result['status']}
            counts[stage+':'+result['status']]+=1
            if result['status']=='complete':
                entry.update(minimum=result['minimum'],empty=result['empty'])
                for kind,transform in (('bond',lambda xs:{bonds(x['label']) for x in xs}),
                                       ('joint',lambda xs:{encoded(x['label']) for x in xs})):
                    entry[kind]=overlap(transform(old['joint_labels']),transform(result['joint_labels']))
                    counts[stage+':'+kind+'_equal']+=entry[kind]['equal']
            row[stage]=entry
        for scenario in ('kekule','union'):
            path=directory/f'{scenario}_scores'/f'{key}.score.json'
            if not path.exists():
                row[scenario+'_bond_orbits']={'status':'unresolved'}
                continue
            score=record(path)
            entry={'status':score['status']}
            if score['status']=='complete' and oldscore['status']=='complete':
                before,after=oldscore['bond_label_orbits'],score['bond_label_orbits']
                entry.update(original=before,alternative=after,
                             original_width=oldscore['width'],alternative_width=score['width'])
                transition=f'{scenario}:'+('unique' if before==1 else 'multiple')+'_to_'+('unique' if after==1 else 'multiple')
                counts[transition]+=1
            row[scenario+'_bond_orbits']=entry
        rows.append(row)
    assert len(rows)==30
    return {'scope':'Fixed-coordinate bond/joint label overlap and audited bond-label orbit transitions; no new searches or joint-orbit computation',
            'audit_sha256':sha(directory/'audit_v2.json'),'reporter_sha256':sha(Path(__file__)),
            'record_hashes':bindings,'counts':dict(counts),'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=report(args.directory)
    save(args.output,result)
    print(json.dumps(result['counts'],indent=2))
