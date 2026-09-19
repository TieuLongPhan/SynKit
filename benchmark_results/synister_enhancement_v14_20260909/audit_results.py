import json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
R=Path(__file__).resolve().parent
B=R.parent
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

selection=json.loads((R/"selection_1200.json").read_text())
expected={(t["source_line"],t["mode"]) for t in selection["tasks"]}
previous={}
for directory in [
    B/"synister_efficiency_1200_timeouts_20260908/run",
    B/"synister_efficiency_timeouts_v6_20260908/final_verification",
    B/"synister_efficiency_timeouts_v9_20260908/completion_run",
    B/"synister_efficiency_timeouts_v9_20260908/completion_retry",
    B/"synister_enumeration_audit_20260909/python_1200",
    B/"synister_enumeration_audit_20260909/native_449_cache_off",
    B/"synister_enhancement_v11_20260909/cohort_1200",
    B/"synister_enhancement_v12_20260909/cohort_1200",
]:
    for key,value in records(directory).items():
        if value[1].get("result",{}).get("complete"):
            previous[key]=value
summary={"expected_tasks":len(expected),"runs":{}}
for directory in [R/"cohort_1200",R/"17156_single_worker_diagnostic",*sorted(R.glob("hard_repeat_*"))]:
    items=records(directory)
    if not items: continue
    failures=[];checked=0;classification_regressions=[];incomplete=[];classes=0;complete=0;coverage=Counter()
    for key,(path,doc) in items.items():
        new=doc.get("result",{})
        if not new.get("complete"):
            incomplete.append({"key":key,"error":doc.get("error"),"reason":new.get("truncation_reason")})
            continue
        complete+=1
        if new["structure"]["complete"]:
            classes+=1
            for field in ("its_class_counts","template_class_counts"):
                if sum(c for _,c in new["structure"][field])!=new["representative_solution_count"]:
                    failures.append([key,field+".sum"])
        if key in previous:
            old=previous[key][1]["result"]; checked+=1
            coverage["target_minimum_reference_fields"]+=1
            if old["labeled_solution_count"] is not None and new["labeled_solution_count"] is not None:
                coverage["labeled_solution_count"]+=1
            if old["symmetry_quotient_complete"] and new["symmetry_quotient_complete"]:
                coverage["exact_rational_coordinate_frequencies"]+=1
                if old["structure"]["complete"] and new["structure"]["complete"]:
                    coverage["full_class_maps_in_labeled_units"]+=1
            changes=differences(old,new)
            if changes: failures.append([key,changes])
            if old["structure"]["complete"] and not new["structure"]["complete"]:
                classification_regressions.append(key)
        if key==(13067,"minimal"):
            proof=json.loads((R/"13067_independent_classification.json").read_text())["python"]["result"]
            coverage["independent_13067_classification_check"]+=1
            changes=differences(proof,new)
            if changes: failures.append([key,changes])
    timing=json.loads((directory/"case_timings.json").read_text()) if (directory/"case_timings.json").exists() else []
    summary["runs"][directory.name]={
        "processed":len(items),"search_complete":complete,"structure_complete":classes,
        "strict_search_complete_below_60":sum(t["complete_below_60_seconds"] for t in timing),
        "strict_structure_complete_below_60":sum(
            t["complete_below_60_seconds"]
            and items[(t["source_line"],t["mode"])][1].get("result",{}).get("structure",{}).get("complete",False)
            for t in timing if (t["source_line"],t["mode"]) in items),
        "comparisons":checked,"comparison_coverage":dict(coverage),"failures":failures,"classification_regressions":classification_regressions,
        "incomplete":incomplete,"missing_count":len(expected-items.keys()) if directory.name=="cohort_1200" else None,
        "max_full_output_seconds":max((t["end_to_end_wall_seconds"] for t in timing),default=None),
        "total_full_output_seconds":sum(t["end_to_end_wall_seconds"] for t in timing),
        "completed_over_budget":[{"source_line":t["source_line"],"mode":t["mode"],
          "end_to_end_wall_seconds":t["end_to_end_wall_seconds"]}
          for t in timing if t["end_to_end_wall_seconds"]>=60
          and items.get((t["source_line"],t["mode"]),(None,{}))[1].get("result",{}).get("complete")],
      "pattern_cache_hits":sum(d.get("result",{}).get("backend_statistics",{}).get("native_orbits",{}).get("pattern_cache_hits",0) for _,d in items.values()),
    }
(R/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
