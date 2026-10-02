"""Independent direct-set arithmetic checks for the archived diagnostics."""
from fractions import Fraction
import json
from pathlib import Path

import pytest

from synkit.Chem.Mapper.identifiability import extract_label,parse_reaction


@pytest.mark.parametrize('cohort,expected_resolved,expected_changed',[
    ('c1',1000,88),('c2',491,39)])
def test_archived_fixed_coordinate_scores_against_direct_sets(cohort,expected_resolved,expected_changed):
    root=Path(__file__).resolve().parents[3]/'paper/synister/evidence'
    read=lambda path:json.loads(path.read_text())
    report=read(root/f'{cohort}_unaligned_diagnostic_v1.json')
    parent=root/f'identifiability_{cohort}_primary_v1'
    inputs={x['case_id']:x for x in read(parent/'inputs.json')}
    def f1(a,b):
        return Fraction(2*len(a&b),len(a)+len(b)) if a or b else Fraction(1)
    intervals=[];changed=0;invalid=0
    for row in report['rows']:
        key=row['case_id']
        preds=[read(parent/'cases'/f'{key}.{m}.json') for m in ('slap','rxnmapper')]
        if row['status']=='invalid_prediction':
            assert any(x['status']!='valid' for x in preds)
            invalid+=1;continue
        exact=read(parent/'cases'/f'{key}.exact.json')
        if row['status']=='unresolved_search':
            assert exact['status']!='complete' or not exact['labels_complete']
            intervals.append(None);continue
        r,p=parse_reaction(inputs[key]['reaction'])
        supports=[extract_label(r,p,x['prediction']['mapping']).changed_bonds for x in preds]
        candidates=[extract_label(r,p,x['mapping']).changed_bonds for x in exact['labels']]
        values=[f1(supports[0],y)-f1(supports[1],y) for y in candidates]
        lo,hi=min(values),max(values)
        for end,expected in (('lower',lo),('upper',hi)):
            witness=row[end]
            y=extract_label(r,p,witness['mapping']).changed_bonds
            assert y in candidates
            actual=[f1(x,y) for x in supports]
            assert list(map(Fraction,witness['scores']))==actual
            assert Fraction(witness['difference'])==actual[0]-actual[1]==expected
        intervals.append((lo,hi))
        if row['aligned_interval'] is not None:
            changed+=list(map(Fraction,row['aligned_interval']))!=[lo,hi]
    resolved=[x for x in intervals if x is not None]
    assert len(resolved)==report['resolved']==expected_resolved
    assert changed==report['changed_interval_on_jointly_resolved']==expected_changed
    assert len(inputs)==len(intervals)+invalid
    q=Fraction(len(intervals)-len(resolved),len(intervals))
    lo=sum(x[0] for x in resolved)/len(resolved)
    hi=sum(x[1] for x in resolved)/len(resolved)
    assert Fraction(report['bounds']['outer_lower'])==(1-q)*lo-q
    assert Fraction(report['bounds']['outer_upper'])==(1-q)*hi+q
