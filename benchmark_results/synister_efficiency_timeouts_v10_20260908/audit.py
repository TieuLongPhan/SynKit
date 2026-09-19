# Compare all exact observables against frozen V9, including entire class maps.
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
V9=ROOT.parent/"synister_efficiency_timeouts_v9_20260908"
fields=("target","reference_cd","representative_solution_count","labeled_solution_count",
        "symmetry_group_order","symmetry_quotient_complete","mapping_hartley_entropy_nats",
        "reference_class_observed","shell_complete_and_reference_class_observed")
results=[]
for name in sys.argv[1:]:
    directory=ROOT/name
    for path in sorted(directory.glob("*_reference_cd.json")):
        case=int(path.name.split("_")[0])
        baseline=V9/("completion_retry" if case==30474 else "completion_run")/path.name
        if not baseline.exists():
            baseline=V9/"native_regression"/path.name
        old=json.loads(baseline.read_text())["result"]
        doc=json.loads(path.read_text())
        new=doc.get("result",{})
        errors=[]
        if not new.get("complete"):
            errors.append("incomplete")
        else:
            errors.extend(k for k in fields if old[k]!=new[k])
            errors.extend("reaction_center."+k for k in old["reaction_center"] if k!="representative_stream_sha256" and old["reaction_center"][k]!=new["reaction_center"][k])
            errors.extend("structure."+k for k in old["structure"] if old["structure"][k]!=new["structure"][k])
            for field in ("its_class_counts","template_class_counts"):
                if sum(x[1] for x in new["structure"][field])!=new["representative_solution_count"]:
                    errors.append(field+".weight_sum")
        results.append({"run":name,"source_line":case,"exact_match":not errors,"mismatches":errors})
        print(json.dumps(results[-1]),flush=True)
    (directory/"exact_audit.json").write_text(json.dumps([x for x in results if x["run"]==name],indent=2)+"\n")
assert results and all(x["exact_match"] for x in results)
