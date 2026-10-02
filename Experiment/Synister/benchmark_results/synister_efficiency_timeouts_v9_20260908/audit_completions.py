"""Audit final results against completed independent schedule runs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
checks=[]
for source_line in (28163,30474,22361):
    path=ROOT/"completion_run"/f"{source_line}_reference_cd.json"
    if not path.exists():
        checks.append({"source_line":source_line,"status":"pending"})
        continue
    retry=ROOT/"completion_retry"/f"{source_line}_reference_cd.json"
    if retry.exists():
        path=retry
    record=json.loads(path.read_text())
    result=record.get("result",{})
    if not result.get("complete"):
        checks.append({"source_line":source_line,"status":"incomplete","error":record.get("error")})
        continue
    comparison=ROOT/(f"direct_{source_line}.json" if source_line==28163 else f"long_{source_line}.json")
    previous=json.loads(comparison.read_text())
    assert previous["complete"]
    expected={
        "representative_solution_count":previous["weighted_product_representatives"],
        "labeled_solution_count":str(previous["labeled_mappings"]),
        "reference_class_observed":True,
    }
    mismatches=[name for name,value in expected.items() if result[name]!=value]
    for name,value in (("observed_its_class_count",previous["its_classes"]),
                       ("observed_template_class_count",previous["template_classes"]),
                       ("complete",True),("reference_its_class_observed",True),
                       ("reference_template_class_observed",True)):
        if result["structure"][name]!=value:
            mismatches.append("structure."+name)
    if result["backend_statistics"]["native_orbits"]["candidate_count"]!=previous["candidate_count"]:
        mismatches.append("candidate_count")
    if source_line==28163:
        old=json.loads((ROOT.parent/"synister_efficiency_timeouts_v8_20260908/verified_run/28163_reference_cd.json").read_text())["result"]
        if old["structure"]!=result["structure"]:
            mismatches.append("v8_structure")
        for name,value in old["reaction_center"].items():
            if name!="representative_stream_sha256" and result["reaction_center"][name]!=value:
                mismatches.append("v8_reaction_center."+name)
    checks.append({"source_line":source_line,"status":"mismatch" if mismatches else "exact_match",
                   "mismatches":mismatches,"comparison":str(comparison)})
audit={"comparisons":checks,
       "exact_matches":sum(c["status"]=="exact_match" for c in checks),
       "pending":sum(c["status"]=="pending" for c in checks)}
(ROOT/"completion_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit))
