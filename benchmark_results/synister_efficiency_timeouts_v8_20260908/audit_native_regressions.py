"""Compare exact observables, allowing the documented compressed stream to differ."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
baseline=ROOT.parent/"synister_efficiency_timeouts_v7_20260908/verified_run"
selection=json.loads((ROOT/"native_regression_selection.json").read_text())
results=[]
for task in selection["tasks"]:
    filename=f'{task["source_line"]}_reference_cd.json'
    path=ROOT/"native_regression"/filename
    if not path.exists():
        results.append({"source_line":task["source_line"],"status":"pending"})
        continue
    record=json.loads(path.read_text())
    actual=record.get("result",{})
    expected=json.loads((baseline/filename).read_text())["result"]
    if not actual.get("complete"):
        results.append({"source_line":task["source_line"],"status":"incomplete",
                        "error":record.get("error"),"reason":actual.get("truncation_reason")})
        continue
    mismatches=[]
    fields=("target","reference_cd","representative_solution_count",
            "labeled_solution_count","symmetry_group_order","symmetry_quotient_complete",
            "mapping_hartley_entropy_nats","reference_class_observed",
            "shell_complete_and_reference_class_observed")
    for field in fields:
        if actual[field]!=expected[field]:
            mismatches.append(field)
    for field in expected["reaction_center"]:
        if field == "representative_stream_sha256":
            continue
        if actual["reaction_center"][field]!=expected["reaction_center"][field]:
            mismatches.append("reaction_center."+field)
    for field in expected["structure"]:
        if actual["structure"][field]!=expected["structure"][field]:
            mismatches.append("structure."+field)
    results.append({"source_line":task["source_line"],
                    "status":"mismatch" if mismatches else "exact_match",
                    "mismatches":mismatches})
summary={status:sum(item["status"]==status for item in results)
         for status in ("exact_match","mismatch","incomplete","pending")}
summary["total"]=len(results)
summary["comparisons"]=results
(ROOT/"native_regression_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({key:value for key,value in summary.items() if key!="comparisons"}))
