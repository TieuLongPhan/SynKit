"""Independent rank/accounting audit plus all selected Rhea input checks.

Does not independently reparse every unselected/unsupported source record.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

from Experiment.Synister.audit_development import sha
from Experiment.Synister.development import save
from Experiment.Synister.select_development import endpoint_key
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.numerical_scope import validate_study_domain
from synkit.Chem.Mapper.prediction_adapter import unmapped_input


def audit(directory, source, protocol):
    read=lambda name:json.loads((directory/name).read_text())
    m=read('manifest.json')
    for name,value in m['source_sha256'].items(): assert sha(source/name)==value
    assert sha(protocol)==m['protocol_sha256']
    for name in ('selection','frame','accounting','flower_accounting','source_snapshot'):
        assert sha(directory/f'{name}.json')==m[name+'_sha256']
    assert sha(directory/'inputs.csv.gz')==m['dataset_sha256']
    selected,frame,records,flower=(read(n+'.json') for n in ('selection','frame','accounting','flower_accounting'))
    with (source/'rhea-directions.tsv').open() as f:
        directions=list(csv.DictReader(f,delimiter='\t'))
    with (source/'rhea-reaction-smiles.tsv').open() as f:
        smiles=dict(csv.reader(f,delimiter='\t'))
    assert len(records)==len(directions)==len({r['master_id'] for r in records})
    assert len(flower)==38745==len({r['r_id'] for r in flower})
    assert Counter(r['status'] for r in records)==m['row_accounting']
    assert Counter(r['status'] for r in flower)==m['flower_accounting']
    overlap={r['endpoint_sha256'] for r in flower if r['status']=='supported'}
    survivors=[]
    for record,direction in zip(records,directions):
        assert record['master_id']==direction['RHEA_ID_MASTER'] and record['lr_id']==direction['RHEA_ID_LR']
        status=record['status']
        if status=='missing_lr_smiles': assert record['lr_id'] not in smiles
        else: assert record['original_reaction']==smiles[record['lr_id']]
        if status in ('selected','frame_not_selected','duplicate_endpoint'):
            assert record['endpoint_sha256'] not in overlap
            survivors.append(record)
        elif status=='flower_endpoint_overlap': assert record['endpoint_sha256'] in overlap
        else: assert status in ('missing_lr_smiles','unsupported_input')
    representatives={}
    for r in sorted(survivors,key=lambda r:int(r['master_id'])):
        representatives.setdefault(r['endpoint_sha256'],r)
    def rank(r):
        identifier=str(int(r['master_id']))
        return hashlib.sha256(b'synister-c2-rhea-v1\0'+identifier.encode()).digest(),int(identifier)
    expected=sorted(representatives.values(),key=rank)
    assert [r['master_id'] for r in frame]==[r['master_id'] for r in expected]
    assert selected==frame[:500] and len(selected)==500
    assert len(frame)==m['frame']
    for row in selected:
        reaction=unmapped_input(smiles[row['lr_id']])
        assert reaction==row['reaction']
        r,p=parse_reaction(reaction)
        validate_study_domain(r,p)
        assert len(r.atomic_numbers)==row['heavy_atoms']
        assert endpoint_key(reaction)==row['endpoint_sha256']
    with gzip.open(directory/'inputs.csv.gz','rt') as f:
        inputs=list(csv.DictReader(f))
    assert inputs==[{'source_line':r['lr_id'],'reaction_id':'RHEA:'+r['master_id'],
                    'mapped_reaction':r['reaction']} for r in selected]
    return {'status':'verified','scope':__doc__,'selected':500,'frame':len(frame),
            'master_records':len(records),'flower_records':len(flower),
            'manifest_sha256':sha(directory/'manifest.json'),'auditor_sha256':sha(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('directory','source','protocol','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.directory,args.source,args.protocol)
    save(args.output,result)
    print(json.dumps(result,indent=2))
