"""C2 input-only Rhea frame with full FlowER composite overlap exclusion."""
import argparse
from collections import Counter
import csv
import gzip
import io
from pathlib import Path

from rdkit import rdBase
from Experiment.Synister.audit_development import sha
from Experiment.Synister.development import digest, save, snapshot
from Experiment.Synister.select_confirmation import normalized
from Experiment.Synister.select_development import endpoint_key, EXPECTED, stratum
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.numerical_scope import validate_study_domain
from synkit.Chem.Mapper.prediction_adapter import unmapped_input


def choose(candidates, count=500):
    unique = {}
    for row in sorted(candidates, key=lambda x:int(x["master_id"])):
        unique.setdefault(row["endpoint_sha256"], row)
    frame = sorted(unique.values(), key=lambda x:(digest(("synister-c2-rhea-v1\0"+str(int(x["master_id"]))).encode()),int(x["master_id"])))
    if len(frame)<count:
        raise ValueError(f"Only {len(frame)} eligible masters; prospective amendment required")
    return frame[:count], frame


def supported(reaction):
    value = unmapped_input(reaction)
    r,p = parse_reaction(value)
    validate_study_domain(r,p)
    return value, endpoint_key(value), len(r.atomic_numbers)


def run(source, flower, protocol, output):
    expected={"rhea-reaction-smiles.tsv":"3e55e8f7abce42951f22a759e05457f028afd5d4c8cc7d200e59d31f750fb5e5",
              "rhea-directions.tsv":"deae1911da372c70ee8fc97992809c7923e63b288a8702fc4396edcb2633fb04"}
    assert sha(flower)==EXPECTED["composite"]
    for name,value in expected.items(): assert sha(source/name)==value
    with (source/"rhea-directions.tsv").open() as stream:
        directions=list(csv.DictReader(stream,delimiter="\t"))
    assert len({r['RHEA_ID_MASTER'] for r in directions})==len(directions)
    ids=[r[k] for r in directions for k in ('RHEA_ID_MASTER','RHEA_ID_LR','RHEA_ID_RL','RHEA_ID_BI')]
    assert len(ids)==len(set(ids))
    with (source/"rhea-reaction-smiles.tsv").open() as stream:
        smiles_rows=list(csv.reader(stream,delimiter="\t"))
    assert all(len(row)==2 for row in smiles_rows)
    smiles=dict(smiles_rows)
    assert len(smiles)==len(smiles_rows) and set(smiles)<=set(ids)
    overlap, flower_accounting = set(), []
    accounting,candidates=[],[]
    with rdBase.BlockLogs(), gzip.open(flower,"rt") as stream:
        for row in csv.DictReader(stream):
            record={"r_id":row['r_id'],"original_id":row['original_id']}
            try:
                _,key,_=supported(normalized(row['original_id'],row['ground_truth']))
                overlap.add(key)
                record.update(status="supported",endpoint_sha256=key)
            except ValueError as exc:
                record.update(status="unsupported",reason=str(exc))
            flower_accounting.append(record)
    assert len(flower_accounting)==38745
    with rdBase.BlockLogs():
        for row in directions:
            master,lr=row['RHEA_ID_MASTER'],row['RHEA_ID_LR']
            record={"master_id":master,"lr_id":lr}
            if lr not in smiles:
                record['status']='missing_lr_smiles'
            else:
                record['original_reaction']=smiles[lr]
                try:
                    reaction,key,n=supported(smiles[lr])
                    record.update(reaction=reaction,endpoint_sha256=key,heavy_atoms=n,stratum=stratum(n))
                    record['status']='flower_endpoint_overlap' if key in overlap else 'eligible'
                    if record['status']=='eligible': candidates.append(dict(record))
                except ValueError as exc:
                    record.update(status="unsupported_input",reason=str(exc))
            accounting.append(record)
    selected,frame=choose(candidates)
    selected_ids={r['master_id'] for r in selected}; frame_ids={r['master_id'] for r in frame}
    for row in accounting:
        if row['status']=='eligible':
            row['status']=('selected' if row['master_id'] in selected_ids else
                           'frame_not_selected' if row['master_id'] in frame_ids else 'duplicate_endpoint')
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (("selection",selected),("frame",frame),("accounting",accounting),("flower_accounting",flower_accounting)):
        save(output/f'{name}.json',value)
    with (output/"inputs.csv.gz").open("xb") as stream:
        with gzip.GzipFile(filename="",fileobj=stream,mode="wb",mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding="utf-8",newline="") as text:
                writer=csv.DictWriter(text,fieldnames=("source_line","reaction_id","mapped_reaction"))
                writer.writeheader()
                for r in selected:
                    writer.writerow(dict(source_line=r['lr_id'],reaction_id='RHEA:'+r['master_id'],mapped_reaction=r['reaction']))
    manifest={"scope":"C2_outcome_blind_replication_selection","selected":len(selected),"frame":len(frame),
              "source_sha256":expected,"flower_sha256":sha(flower),"protocol_sha256":sha(protocol),
              "row_accounting":dict(Counter(r['status'] for r in accounting)),
              "flower_accounting":dict(Counter(r['status'] for r in flower_accounting)),
              "source_snapshot_sha256":snapshot(output),"dataset_sha256":sha(output/'inputs.csv.gz')}
    for name in ('selection','frame','accounting','flower_accounting'):
        manifest[name+'_sha256']=sha(output/f'{name}.json')
    save(output/'manifest.json',manifest)
    print(manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','flower','protocol','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.flower,args.protocol,args.output)
