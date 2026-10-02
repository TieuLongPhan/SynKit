"""Create the C2 replication lock only after audited input-only selection."""
import argparse
import json
from pathlib import Path

from Experiment.Synister.audit_development import sha
from Experiment.Synister.confirmation_contract import SETTINGS, source_contents
from Experiment.Synister.development import save, environment
from Experiment.Synister.freeze_environment import closure, ROOTS


def make_lock(selection, primary, protocol, output):
    read=lambda p:json.loads(p.read_text())
    selected=read(selection/'manifest.json')
    audit=read(selection/'selection_audit.json')
    assert selected['scope']=='C2_outcome_blind_replication_selection' and selected['selected']==500
    assert audit['status']=='verified' and audit['selected']==500
    assert audit['manifest_sha256']==sha(selection/'manifest.json')
    assert selected['protocol_sha256']==sha(protocol)
    for name in ('selection','frame','accounting','flower_accounting','source_snapshot'):
        assert sha(selection/f'{name}.json')==selected[name+'_sha256']
    assert sha(selection/'inputs.csv.gz')==selected['dataset_sha256']
    original=read(primary/'execution_lock.json')
    packages=closure(ROOTS)
    assert packages==original['packages']
    env=environment()
    assert env['rxnmapper_resource_sha256']==original['rxnmapper_resource_sha256']
    output.mkdir(parents=True,exist_ok=False)
    save(output/'all_sources.json',source_contents())
    save(output/'environment.json',env)
    lock={'protocol':'identifiability-c2-rhea-v1','scope':'pre-outcome_C2_execution_lock',
          'settings':dict(SETTINGS,limit=500),'margin':'1/50',
          'selection_manifest_sha256':sha(selection/'manifest.json'),
          'selection_audit_sha256':sha(selection/'selection_audit.json'),
          'dataset_sha256':selected['dataset_sha256'],'protocol_sha256':sha(protocol),
          'primary_lock_sha256':sha(primary/'execution_lock.json'),
          'all_sources_sha256':sha(output/'all_sources.json'),'packages':packages,
          'rxnmapper_resource_sha256':env['rxnmapper_resource_sha256'],
          'environment_sha256':sha(output/'environment.json'),
          'policy':'all predictions frozen before searches; C1 resource/method/metric contract; no reference seeds or retry substitution'}
    save(output/'lock.json',lock)
    print(json.dumps({'lock_sha256':sha(output/'lock.json')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('selection','primary','protocol','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    make_lock(args.selection,args.primary,args.protocol,args.output)
