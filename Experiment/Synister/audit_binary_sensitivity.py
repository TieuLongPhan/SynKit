"""R1 witness/orbit replay and objective comparison with full missingness bounds."""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import json

from Experiment.Synister.audit_development import label, bonds, f1, is_automorphism, transformed, sha
from Experiment.Synister.development import digest, encoded, save
from Experiment.Synister.run_binary_sensitivity import select
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator
from synkit.Chem.Mapper.cohort_evaluation import paired_cohort_bounds
from synkit.Chem.Mapper.prediction_adapter import align_mapped_prediction


def audit(directory, parent, selection, references):
    read = lambda p: json.loads(p.read_text())
    manifest = read(directory / "manifest.json")
    assert sha(parent / "manifest.json") == manifest["parent_manifest_sha256"]
    assert sha(parent / "audit.json") == manifest["parent_audit_sha256"]
    assert sha(selection) == manifest["parent_selection_sha256"]
    for name, field in (("selection.json","selection_sha256"), ("search_tasks.json","search_tasks_sha256"),
                        ("parent_records.json","parent_records_sha256"),("all_sources.json","all_sources_sha256")):
        assert sha(directory/name) == manifest[field]
    assert read(directory/"selection.json") == select(read(selection))
    selection_manifest = read(selection.parent/"manifest.json")
    assert sha(references) == selection_manifest["references_sha256"]
    assert sha(selection.parent/"manifest.json") == read(parent/"manifest.json")["selection_manifest_sha256"]
    refs = {r["reaction_id"]:r["mapped_reaction"] for r in read(references)}
    for name, value in read(directory/"parent_records.json").items():
        assert sha(directory/"cases"/name) == value == sha(parent/"cases"/name)
    score_tasks = {t["case_id"]:t for t in read(directory/"score_tasks.json")}
    counts, rows = Counter(), []
    intervals, weighted = [], []
    for task in read(directory/"search_tasks.json"):
        key = task["case_id"]
        r,p = parse_reaction(task["reaction"])
        result = read(directory/"cases"/f"{key}.binary_exact.json")
        assert result["task_sha256"] == digest(encoded(task))
        counts[f"search:{result['status']}"] += 1
        old = read(directory/"cases"/f"{key}.exact.json")
        old_score = read(directory/"cases"/f"{key}.score.json")
        weighted.append([old_score[e]["difference"] for e in ("lower","upper")]
                        if old_score["status"] == "complete" else None)
        if result["status"] != "complete":
            intervals.append(None)
            rows.append({"case_id":key,"status":result["status"]})
            continue
        assert result["minimum_proved"] and result["enumeration_complete"] and not result["symmetry_pruning"]
        def cost(mapping):
            value = label(r,p,mapping)
            return sum((e[2]==0) != (e[3]==0) for e in value["typed_bond_edits"])
        for candidate in result["joint_labels"] + result["labels"]:
            assert label(r,p,candidate["mapping"]) == candidate["label"]
            assert cost(candidate["mapping"]) == result["minimum"]
        canonical = lambda xs: {encoded(x["label"]) for x in xs}
        bj,wj = canonical(result["joint_labels"]), canonical(old["joint_labels"])
        bb,wb = ({bonds(x["label"]) for x in result["labels"]}, {bonds(x["label"]) for x in old["labels"]})
        same_joint, same_bond = bj==wj, bb==wb
        counts["same_joint_labels"] += same_joint
        counts["same_bond_labels"] += same_bond
        reference = align_mapped_prediction(task["reaction"],refs[task["reaction_id"]]).mapping
        admitted = cost(reference) == result["minimum"]
        ref_label = label(r,p,reference)
        weighted_admitted = Fraction(sum(abs(e[3]-e[2]) for e in ref_label["typed_bond_edits"]),2) == old["minimum"]
        counts["binary_reference_in_minimum"] += admitted
        counts["weighted_reference_in_minimum_on_binary_closed"] += weighted_admitted
        record = read(directory/"binary_scores"/f"{key}.score.json")
        st = score_tasks[key]
        assert record["task_sha256"] == digest(encoded(st))
        assert st["labels"] == result["labels"]
        predictions = []
        for method, name in (("slap","a"),("rxnmapper","b")):
            saved = read(directory/"cases"/f"{key}.{method}.json")
            assert saved["status"] == "valid" and st[f"prediction_{name}"] == saved["prediction"]["mapping"]
            predictions.append(bonds(label(r,p,saved["prediction"]["mapping"])))
        counts[f"score:{record['status']}"] += 1
        if record["status"] != "complete":
            intervals.append(None)
            continue
        engine = SupportOrbitEvaluator(r,time_limit_seconds=120)
        candidates = {bonds(x["label"]) for x in result["labels"]}
        differences, orbit_keys = [], set()
        for y in candidates:
            orbit = engine.orbit(y)
            for image,g in orbit.items():
                assert is_automorphism(r,g) and transformed(y,g) == image
            orbit_keys.add(min(tuple(sorted(z)) for z in orbit))
            differences.append(max(f1(predictions[0],z) for z in orbit)-max(f1(predictions[1],z) for z in orbit))
        lo,hi = min(differences),max(differences)
        assert Fraction(record["width"]) == hi-lo
        assert record["fixed_bond_labels"] == len(candidates)
        assert record["bond_label_orbits"] == len(orbit_keys)
        for end,value in (("lower",lo),("upper",hi)):
            witness = record[end]
            assert Fraction(witness["difference"]) == value
            y = frozenset(tuple(x) for x in witness["label"])
            assert y in candidates
            for name,pred in zip(("a","b"),predictions):
                g = witness[f"{name}_transporter"]
                assert is_automorphism(r,g)
                assert f1(pred,transformed(y,g)) == Fraction(witness[f"{name}_score"])
        intervals.append((lo,hi))
        counts["positive_width"] += hi>lo
        counts["local_reversal"] += lo<0<hi
        transition = f"weighted_{'multiple' if old_score['bond_label_orbits']>1 else 'unique'}_to_binary_{'multiple' if len(orbit_keys)>1 else 'unique'}"
        counts[transition] += 1
        rows.append({"case_id":key,"reaction_id":task["reaction_id"],"status":"verified",
                     "same_joint_labels":same_joint,"same_bond_labels":same_bond,
                     "joint_jaccard":str(Fraction(len(bj & wj),len(bj | wj))),
                     "bond_jaccard":str(Fraction(len(bb & wb),len(bb | wb))),
                     "weighted_bond_orbits":old_score["bond_label_orbits"],"binary_bond_orbits":len(orbit_keys),
                     "weighted_reference_in_minimum":weighted_admitted,
                     "binary_reference_in_minimum":admitted,"lower":str(lo),"upper":str(hi)})
    assert len(intervals)==len(weighted)==150
    summary=read(directory/"summary.json")
    assert summary["selected"]==150
    assert summary["binary_search_status"]=={k.split(':',1)[1]:v for k,v in counts.items() if k.startswith('search:')}
    assert summary["binary_score_status"]=={k.split(':',1)[1]:v for k,v in counts.items() if k.startswith('score:')}
    def bounds(values):
        return {k:str(v) if isinstance(v,Fraction) else v for k,v in asdict(paired_cohort_bounds(values)).items()}
    return {"scope":"R1 artifact and score replay; same orbit engine, independent cost/transporter checks; no independent search-closure proof",
            "manifest_sha256":sha(directory/"manifest.json"),"auditor_sha256":sha(Path(__file__)),
            "counts":dict(counts),"binary_bounds":bounds(intervals),"weighted_bounds":bounds(weighted),"rows":rows}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("directory","parent","selection","references","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.directory,args.parent,args.selection,args.references)
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
