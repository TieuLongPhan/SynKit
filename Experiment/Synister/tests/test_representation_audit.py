"""Negative controls for archived R2 score/task/witness verification."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from Experiment.Synister.audit_representation_sensitivity import score_replay
from synkit.Chem.Mapper.identifiability import parse_reaction


def example():
    root=Path(__file__).resolve().parents[3]/'paper/synister/evidence/identifiability_r2_representation_v1'
    read=lambda p:json.loads(p.read_text())
    task=next(t for t in read(root/'kekule_score_tasks.json') if t['case_id']=='case_0007')
    result=read(root/'kekule_scores/case_0007.score.json')
    return (*parse_reaction(task['reaction']),task,result)


def test_original_score_replay_and_unresolved_preservation():
    r,p,task,result=example()
    lo,hi=score_replay(r,p,task['labels'],task,result)
    assert lo==Fraction(result['lower']['difference'])
    assert hi==Fraction(result['upper']['difference'])
    incomplete=deepcopy(result);incomplete['status']='hard_timeout'
    assert score_replay(r,p,task['labels'],task,incomplete) is None


@pytest.mark.parametrize('field',['task','width','orbit_count','score','transporter'])
def test_score_auditor_rejects_corrupted_evidence(field):
    r,p,task,result=example()
    result=deepcopy(result)
    if field=='task':result['task_sha256']='0'*64
    elif field=='width':result['width']=str(Fraction(result['width'])+1)
    elif field=='orbit_count':result['bond_label_orbits']+=1
    elif field=='score':result['lower']['a_score']=str(Fraction(result['lower']['a_score'])+1)
    else:result['lower']['a_transporter']=[0]*len(r.atomic_numbers)
    with pytest.raises(AssertionError):
        score_replay(r,p,task['labels'],task,result)
