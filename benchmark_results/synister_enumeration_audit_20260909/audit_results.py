"""Audit completed mathematical observables, keeping protocol and partials explicit."""
import json,hashlib
from fractions import Fraction
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent
B=R.parent
selection=json.loads((R/"selection_1200.json").read_text())
tasks={(t["source_line"],t["mode"]) for t in selection["tasks"]}
assert len(tasks)==1200
def records(directory):
    result={}
    if not directory.exists(): return result
    for p in directory.glob("*.json"):
        if p.stem in ("manifest","summary","case_timings","exact_audit","regression_audit"): continue
        d=json.loads(p.read_text())
        if "source_line" in d and "mode" in d:
            result[(d["source_line"],d["mode"])]=(p,d)
    return result
def normalized(result,field):
    rc=result["reaction_center"]
    return {tuple(row[:-1]):Fraction(row[-1],rc["frequency_denominator"]) for row in rc[field] if row[-1]}
def differences(old,new):
    errors=[]
    for field in ("target","minimum_cost","reference_cd","reference_class_observed","reference_is_global_minimum_proven"):
        if old[field]!=new[field]: errors.append(field)
    if old["labeled_solution_count"] is not None and new["labeled_solution_count"] is not None:
        if int(old["labeled_solution_count"])!=int(new["labeled_solution_count"]):errors.append("labeled_solution_count")
    if old["symmetry_quotient_complete"] and new["symmetry_quotient_complete"]:
        for field in ("atom_change_counts","bond_change_counts"):
            if normalized(old,field)!=normalized(new,field):errors.append(field)
        if old["structure"]["complete"] and new["structure"]["complete"]:
            for field in ("its_class_counts","template_class_counts"):
                a=sorted((name,int(count)*int(old["symmetry_group_order"])) for name,count in old["structure"][field])
                b=sorted((name,int(count)*int(new["symmetry_group_order"])) for name,count in new["structure"][field])
                if a!=b:errors.append(field)
    return errors
original=records(B/"synister_efficiency_1200_timeouts_20260908/run")
previous=dict(original)
for key,value in records(B/"synister_efficiency_timeouts_v6_20260908/final_verification").items():
    if key in tasks and value[1].get("result",{}).get("complete"):previous[key]=value
for name in ("completion_run","completion_retry"):
    for key,value in records(B/"synister_efficiency_timeouts_v9_20260908"/name).items():
        if key in tasks and value[1].get("result",{}).get("complete"):previous[key]=value
report={"cohort_tasks":1200,"modes":dict(Counter(mode for _,mode in tasks)),"runs":{}}
for name in ("python_1200","native_449_cache_on","native_449_cache_off","native_hard_cache_off_diagnostic","guarded_60s_30474","guarded_60s_22361"):
    items=records(R/name)
    if not items:continue
    expected=({(int(name.rsplit("_",1)[1]),"reference_cd")} if name.startswith("guarded_60s_") else (tasks if name=="python_1200" else ({(30474,"reference_cd"),(22361,"reference_cd")} if name=="native_hard_cache_off_diagnostic" else {key for key in tasks if key[1]=="reference_cd"})))
    failures=[];classification_regressions=[];checked=0;fields_checked=Counter()
    entries=[]
    for key,(path,doc) in items.items():
        new=doc.get("result",{})
        if new.get("complete"):
            for field in ("its_class_counts","template_class_counts"):
                if new["structure"]["complete"] and sum(c for _,c in new["structure"][field])!=new["representative_solution_count"]:
                    failures.append([list(key),field+".sum"])
        reference=previous.get(key)
        if reference and new.get("complete") and reference[1].get("result",{}).get("complete"):
            old=reference[1]["result"]
            errors=differences(old,new)
            checked+=1
            if errors:failures.append([list(key),errors])
            if old["structure"]["complete"] and not new["structure"]["complete"]:
                classification_regressions.append([list(key),new["structure"]["incomplete_reason"]])
            entries.append({"source_line":key[0],"mode":key[1],"reference":str(reference[0]),"reference_sha256":hashlib.sha256(reference[0].read_bytes()).hexdigest(),"differences":errors,"both_structure_complete":old["structure"]["complete"] and new["structure"]["complete"]})
    values=[doc for _,doc in items.values()]
    stats={"processed":len(items),"expected":len(expected),"complete":sum(d.get("result",{}).get("complete",False) for d in values),
      "search_and_structure_complete":sum(d.get("result",{}).get("complete",False) and d["result"]["structure"]["complete"] for d in values),
      "errors":[{"key":list(k),"error":d["error"]} for k,(_,d) in items.items() if "error" in d],
      "incomplete":[{"key":list(k),"reason":d.get("result",{}).get("truncation_reason"),"wall_seconds":d["wall_seconds"]} for k,(_,d) in items.items() if not d.get("result",{}).get("complete")],
      "completed_exact_comparisons":checked,"observable_failures":failures,"classification_regressions":classification_regressions,
      "unexpected_keys":[list(k) for k in items.keys()-expected],"missing_keys":[list(k) for k in expected-items.keys()],
      "max_complete_analysis_wall_seconds":max((d["wall_seconds"] for d in values if d.get("result",{}).get("complete")),default=None)}
    timing=R/name/"case_timings.json"
    if timing.exists():
        ts=json.loads(timing.read_text())
        manifest=json.loads((R/name/"manifest.json").read_text())
        stats["configured_seconds"]=manifest["seconds"]
        stats["strict_60s_acceptance_eligible"]=manifest["seconds"]==60 and manifest["absolute_case_deadline"]
        stats["observed_complete_below_60"]=sum(t["complete_below_60_seconds"] for t in ts)
        stats["strict_complete_below_60"]=(stats["observed_complete_below_60"] if stats["strict_60s_acceptance_eligible"] else None)
    stats["native_pattern_cache_hits"]=sum(d.get("result",{}).get("backend_statistics",{}).get("native_orbits",{}).get("pattern_cache_hits",0) for d in values)
    report["runs"][name]=stats
    (R/(name+"_comparison.json")).write_text(json.dumps(entries,indent=2)+"\n")
on,off=records(R/"native_449_cache_on"),records(R/"native_449_cache_off")
cache_pairs=[]
for key in on.keys()&off.keys():
    a,b=on[key][1].get("result",{}),off[key][1].get("result",{})
    if not(a.get("complete") and b.get("complete")):continue
    issues=differences(a,b)
    for field in ("representative_solution_count","symmetry_group_order","mapping_hartley_entropy_nats"):
        if a[field]!=b[field]:issues.append(field)
    # Entire structure object must agree; exact frequencies exclude order digest.
    if a["structure"]!=b["structure"]:issues.append("structure")
    for field in a["reaction_center"]:
        if field!="representative_stream_sha256" and a["reaction_center"][field]!=b["reaction_center"][field]:
            issues.append("reaction_center."+field)
    cache_pairs.append({"key":list(key),"differences":issues})
report["cache_differential"]={"completed_pairs":len(cache_pairs),"failures":[p for p in cache_pairs if p["differences"]]}
python_items=records(R/"python_1200")
routed={key:(python_items if key[1]=="minimal" else off).get(key) for key in tasks}
present={key:value for key,value in routed.items() if value is not None}
report["combined_mode_partition_coverage"]={
    "description":"Combine all 751 minimum tasks from Python and all 449 reference-CD tasks from native cache-off; separate campaigns and different recorded resource protocols.",
    "processed":len(present),
    "search_complete":sum(value[1].get("result",{}).get("complete",False) for value in present.values()),
    "search_and_structure_complete":sum(value[1].get("result",{}).get("complete",False) and value[1]["result"]["structure"]["complete"] for value in present.values()),
    "remaining_incomplete_structure":[{"key":list(key),"reason":value[1].get("result",{}).get("structure",{}).get("incomplete_reason")} for key,value in present.items() if not(value[1].get("result",{}).get("complete") and value[1]["result"]["structure"]["complete"])]
}
(R/"cache_differential.json").write_text(json.dumps(cache_pairs,indent=2)+"\n")
(R/"audit_summary.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:{f:v for f,v in s.items() if f not in ("missing_keys","classification_regressions")} for k,s in report["runs"].items()},indent=2))
