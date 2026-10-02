from itertools import permutations
import random
import json
from pathlib import Path

from Experiment.Synister.binary_sensitivity import binary_cost, search, seed
from synkit.Chem.Mapper.identifiability import Endpoint, extract_label


def test_binary_minima_against_literal_oracle_with_weighted_labels():
    rng = random.Random(20260919)
    for _ in range(20):
        n = rng.randrange(2, 6)
        def endpoint():
            return Endpoint((6,)*n, (0,)*n, tuple(rng.randrange(3) for _ in range(n)),
                            tuple((i, j, rng.choice((2, 3, 4))) for i in range(n)
                                  for j in range(i+1, n) if rng.random() < 0.5))
        r, p = endpoint(), endpoint()
        a = {(i,j) for i,j,_ in r.bonds}
        b = {(i,j) for i,j,_ in p.bonds}
        costs = {m: len(a ^ {tuple(sorted((i,j))) for i in range(n) for j in range(i+1,n)
                            if tuple(sorted((m[i],m[j]))) in b}) for m in permutations(range(n))}
        best = min(costs.values())
        maps = {m for m,c in costs.items() if c==best}
        result = search(r,p,max(costs,key=costs.get))
        assert result["status"] == "complete" and result["minimum"] == best
        assert result["emitted_representatives"] == len(maps)
        observed = {extract_label(r,p,x["mapping"]) for x in result["joint_labels"]}
        assert observed == {extract_label(r,p,m) for m in maps}
        assert not result["symmetry_pruning"]


def test_binary_optimum_keeps_weight_only_changes_and_cap_is_incomplete():
    r = Endpoint((6,6), (0,0), (1,1), ((0,1,2),))
    p = Endpoint((6,6), (0,0), (1,1), ((0,1,4),))
    assert binary_cost(r,p,(0,1)) == 0
    result = search(r,p)
    assert result["minimum"] == 0 and result["labels"][0]["label"]["typed_bond_edits"]
    capped = search(r,p,max_mappings=1)
    assert capped["status"] == "unresolved" and capped["labels"] == []
    mapping, info = seed(r,p,{"z":{"status":"valid","prediction":{"mapping":[1,0]}},
                              "a":{"status":"valid","prediction":{"mapping":[0,1]}}})
    assert mapping == [0,1] and info["binary_cost"] == 0


def test_r1_selection_and_all_timeout_accounting(monkeypatch, tmp_path):
    from Experiment.Synister import run_binary_sensitivity as runner
    evidence = Path(__file__).resolve().parents[3] / "paper/synister/evidence"
    selection = evidence / "identifiability_c1_selection_v1/selection.json"
    rows = json.loads(selection.read_text())
    assert runner.select(rows) == runner.select(list(reversed(rows)))
    assert len(runner.select(rows)) == 150
    seen = []
    def fake_execute(task, seconds, records):
        assert task["stage"] == "binary_exact" and seconds == 65
        assert (records.parent / "manifest.json").exists()
        seen.append(task["case_id"])
        return {"case_id":task["case_id"], "status":"hard_timeout"}
    monkeypatch.setattr(runner, "execute", fake_execute)
    monkeypatch.setattr(runner, "source_contents", lambda: {"test":"fixed"})
    runner.run(evidence / "identifiability_c1_primary_v1", selection,
               evidence.parent / "protocols/R1_OBJECTIVE_SENSITIVITY_V1.md", tmp_path / "r1")
    summary = json.loads((tmp_path / "r1/summary.json").read_text())
    assert len(seen) == len(set(seen)) == 150
    assert summary["binary_search_status"] == {"hard_timeout":150}
    assert summary["binary_score_attempts"] == 0
