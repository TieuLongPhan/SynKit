from itertools import permutations
import random
import json
from pathlib import Path

import pytest

from Experiment.Synister.representation_sensitivity import (
    kekule_endpoint, ordering_control, search, union_labels, weighted_cost)
from synkit.Chem.Mapper.identifiability import Endpoint, extract_label


def test_overlap_report_keeps_empty_and_disjoint_sets_distinct():
    from Experiment.Synister.report_representation_labels import overlap
    assert overlap({1,2},{2,3})=={'original':2,'alternative':2,'intersection':1,'jaccard':'1/3','equal':False}
    assert overlap({1},set())['jaccard']=='0'
    assert overlap(set(),set())['jaccard']=='1'
    assert overlap({1},{2})['equal'] is False


def test_numeric_shells_against_literal_oracle():
    rng = random.Random(20260920)
    for _ in range(12):
        n = rng.randrange(2,6)
        def endpoint():
            return Endpoint((6,)*n,(0,)*n,(1,)*n,
                            tuple((i,j,rng.choice((2,3,4))) for i in range(n)
                                  for j in range(i+1,n) if rng.random()<0.5))
        r,p = endpoint(),endpoint()
        a = {(i,j):w for i,j,w in r.bonds}
        b = {(i,j):w for i,j,w in p.bonds}
        # Deliberately compute the literal oracle independently of weighted_cost.
        costs = {m:sum(abs(a.get((i,j),0)-b.get(tuple(sorted((m[i],m[j]))),0))
                       for i in range(n) for j in range(i+1,n))/2
                 for m in permutations(range(n))}
        best = min(costs.values())
        minimum = search(r,p)
        assert minimum["minimum"] == best
        for target in (best+1,best+2,max(costs.values())+1):
            result = search(r,p,target=target)
            maps = {m for m,c in costs.items() if c==target}
            assert result["status"] == "complete"
            assert result["empty"] == (not maps)
            assert result["emitted_representatives"] == len(maps)
            assert {extract_label(r,p,x["mapping"]) for x in result["joint_labels"]} == {
                extract_label(r,p,m) for m in maps}


def test_kekule_preserves_attributes_and_original_label_extraction():
    r,a = kekule_endpoint("c1cc[nH]c1.C[O-]")
    assert r != a
    assert (r.atomic_numbers,r.charges,r.hcounts) == (a.atomic_numbers,a.charges,a.hcounts)
    assert {(i,j) for i,j,_ in r.bonds} == {(i,j) for i,j,_ in a.bonds}
    plain,unchanged = kekule_endpoint("CCO")
    assert plain == unchanged
    r,a = kekule_endpoint("c1ccccc1")
    result = search(r,r,objective_endpoints=(a,a))
    assert result["status"] == "complete" and result["minimum"] == 0
    for record in result["joint_labels"]:
        assert weighted_cost(a,a,record["mapping"]) == 0
        assert not extract_label(r,r,record["mapping"]).typed_bond_edits


def test_cap_and_union_unknown_rule():
    r = Endpoint((6,6),(0,0),(1,1),((0,1,2),))
    minimum = search(r,r)
    empty = search(r,r,target=1)
    capped = search(r,r,max_mappings=1)
    assert capped["status"] == "unresolved" and capped["labels"] == []
    assert union_labels(minimum,[empty,empty])["status"] == "complete"
    assert union_labels(minimum,[empty,capped]) == {
        "status":"unresolved","labels_complete":False,"labels":[]}
    with pytest.raises(ValueError):
        union_labels(minimum,[empty])


def test_r2_runner_keeps_all_failed_attempts(monkeypatch,tmp_path):
    from Experiment.Synister import run_representation_sensitivity as runner
    evidence = Path(__file__).resolve().parents[3]/'paper/synister/evidence'
    seen = []
    def failed(task,seconds,records):
        assert seconds == 65 and (records.parent/'manifest.json').exists()
        seen.append(task)
        return {'case_id':task['case_id'],'status':'hard_timeout'}
    monkeypatch.setattr(runner,'execute',failed)
    monkeypatch.setattr(runner,'source_contents',lambda:{'test':'fixed'})
    runner.run(evidence/'identifiability_c1_primary_v1',
               evidence/'identifiability_c1_selection_v1/selection.json',
               evidence.parent/'protocols/R2_REPRESENTATION_LANDSCAPE_V1.md',tmp_path/'run')
    assert len(seen) == 90 and len({x['case_id'] for x in seen}) == 30
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['union_status'] == {'unresolved':30}
    assert summary['score_status'] == {'kekule':{},'union':{}}
    for key in {x['case_id'] for x in seen}:
        shells = sorted(x['target'] for x in seen if x['case_id']==key and 'target' in x)
        assert shells[1]-shells[0] == 1


def test_ordering_control_transports_attributes_and_bond_presence_back():
    for smiles in ('CCO.[Na+]','c1ccccc1','c1cc[nH]c1.C[O-]','Oc1ccccc1C'):
        control = ordering_control(smiles)
        a,b = control['forward'],control['transported_reverse']
        for attr in ('atomic_numbers','charges','hcounts'):
            assert a[attr] == b[attr]
        assert {(i,j) for i,j,_ in a['bonds']} == {(i,j) for i,j,_ in b['bonds']}
        assert control['encoding_changed'] == (a != b)
    assert not ordering_control('CCO.[Na+]')['encoding_changed']
    with pytest.raises(ValueError,match='permutation'):
        kekule_endpoint('CCO',[0,0,1])
