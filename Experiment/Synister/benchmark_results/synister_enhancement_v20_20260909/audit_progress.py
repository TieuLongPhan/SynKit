"""Compare each newly closed result once; final source/coverage audit is separate."""
import json
from pathlib import Path
import runpy
import time
R=Path(__file__).resolve().parent
module=runpy.run_path(str(R/"audit_results.py"))
expected={(t["source_line"],t["mode"]):t for t in module["selection"]["tasks"]}
seen=set()
checked=0
incomplete=[]
failures=[]
while True:
    finished=[]
    for part in range(2):
        directory=R/f"cohort_part_{part}"
        try:
            timing=json.loads((directory/"case_timings.json").read_text())
            manifest=json.loads((directory/"manifest.json").read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            continue
        finished.append(bool(manifest.get("source_unchanged")))
        for item in timing:
            key=item["source_line"],item["mode"]
            if key in seen:continue
            name="{}_{}.json".format(*key)
            doc=json.loads((directory/name).read_text())
            assert key in expected
            assert doc["reaction_id"]==expected[key]["reaction_id"]
            assert doc["source_sha256_python_and_cpp"]==manifest["source_sha256_python_and_cpp"]
            new=doc.get("result",{})
            if new.get("complete") and new.get("structure",{}).get("complete"):
                old=json.loads((module["BASE"]/name).read_text())["result"]
                changes=module["differences"](old,new)
                if changes:failures.append(dict(key=key,fields=changes))
                checked+=1
            else:incomplete.append(key)
            seen.add(key)
    payload=dict(processed=len(seen),exact_comparisons=checked,incomplete=incomplete,failures=failures)
    temporary=R/"partial_audit.tmp"
    temporary.write_text(json.dumps(payload,indent=2)+"\n")
    temporary.replace(R/"partial_audit.json")
    assert not failures,failures
    if len(finished)==2 and all(finished):break
    time.sleep(20)
print(json.dumps(payload),flush=True)
