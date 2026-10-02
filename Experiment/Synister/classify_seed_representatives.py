"""Separate ITS classification of complete E2 subgroup-representative outputs.

The classifier receives the identity group on the supplied representatives.
Coverage of the full indexed set follows from the audited source enumeration
under its separately verified product subgroup, not from treating Q as L.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import sys
import time

from Experiment.Synister.classify_enumeration import digest, encode, isolated


def perform(task):
    from Experiment.Synister.classification_worker import classify, verify_cyclic_group
    from Experiment.Synister.audit_classification import audit_details
    from synkit.Chem.Mapper.identifiability import parse_reaction
    r,p=parse_reaction(task['reaction'])
    path=Path(task['map_path'])
    if digest(path)!=task['mapping_sha256']:
        raise ValueError('Representative input changed')
    maps=json.loads(path.read_text())
    group=verify_cyclic_group(p,task['source_group'])
    if len(maps)!=task['mapping_count'] or not task['source_complete']:
        raise ValueError('Representative set lacks complete audited input')
    before=time.perf_counter()
    raw=classify(r,p,maps,task['target_doubled_cd'],seconds=30,
                 group=[tuple(range(len(r.atomic_numbers)))])
    classification_seconds=time.perf_counter()-before
    details={key:raw.pop(key) for key in ('class_codes','representatives','failures','bond_pattern_frequencies','joint_pattern_frequencies')}
    before=time.perf_counter()
    checked=audit_details(r,p,maps,raw,details,task['target_doubled_cd'],seconds=30)
    audit_seconds=time.perf_counter()-before
    encoded=encode(details).encode()
    with Path(task['detail_path']).open('xb') as stream:
        stream.write(encoded)
    return {'complete':raw['complete'], 'termination':raw['termination'],
        'input_unit':'complete_verified_subgroup_representatives',
        'input_count':len(maps), 'implied_indexed_count':len(maps)*len(group), 'source_group_order':len(group),
        'classification_seconds':classification_seconds,'audit_seconds':audit_seconds,
        'its_classes':raw['its_classes'] if checked.get('verified') else None,
        'raw_classifier':raw,'structural_audit':checked,'detail_sha256':digest(Path(task['detail_path']))}


def run(study,output):
    from Experiment.Synister.audit_seed_output import audit
    checked=audit(study,Path('paper/synister/evidence/seed_output_controls_v1.json'))
    output.mkdir(parents=True,exist_ok=False)
    (output/'details').mkdir()
    sources={str(p):p.read_text() for p in sorted(Path('synkit/Chem/Mapper').rglob('*.py'))}
    sources.update({str(p):p.read_text() for p in [Path(__file__),Path('Experiment/Synister/classification_worker.py'),
                    Path('Experiment/Synister/audit_classification.py'),Path('Experiment/Synister/structural_oracle.py')]})
    (output/'sources.json').write_text(encode(sources))
    tasks=[]
    for path in sorted((study/'cases').glob('*.json')):
        record=json.loads(path.read_text())
        if not record['complete'] or record.get('output_unit')!='verified_cyclic_subgroup_representatives':
            continue
        key=record['task']['task_id']
        tasks.append({'task_id':key,'reaction':record['task']['reaction'],
            'source_complete':True,'source_group':record['group'], 'source_record_sha256':digest(path),
            'map_path':str((study/'maps'/path.name).resolve()), 'mapping_sha256':record['mapping_sha256'],
            'mapping_count':record['mapping_count'],'target_doubled_cd':record['minimum_doubled_cd'],
            'memory_gib':6,'detail_path':str((output/'details'/path.name).resolve())})
    (output/'tasks.json').write_text(encode(tasks))
    def execute(task):
        record=isolated('Experiment.Synister.classify_seed_representatives',task,65)
        record['task']=task
        (output/(task['task_id']+'.json')).write_text(encode(record))
        print(encode({'task':task['task_id'],'termination':record['termination']}),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(execute,tasks))
    summary={'schema':'synister.seed-representative-classification.v1','study_summary_sha256':checked['summary_sha256'],
        'attempts':len(records),'complete':sum(r['complete'] for r in records),
        'verified':sum(r.get('structural_audit',{}).get('verified',False) for r in records),
        'contradictions':sum(r.get('structural_audit',{}).get('consistent') is False for r in records),
        'files_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*.json'))}}
    (output/'summary.json').write_text(encode(summary))
    print(encode({k:v for k,v in summary.items() if k!='files_sha256'}))
    return summary


if __name__=='__main__':
    if len(sys.argv)==1:
        task=json.load(sys.stdin)
        limit=task['memory_gib']*1024**3
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        started=time.perf_counter()
        try:
            result=perform(task)
        except Exception as exc:
            result={'complete':False,'termination':'worker_error','error_type':type(exc).__name__,'error':str(exc)}
        result.update(worker_seconds=time.perf_counter()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(encode(result))
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--study',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        args=parser.parse_args()
        run(args.study,args.output)
