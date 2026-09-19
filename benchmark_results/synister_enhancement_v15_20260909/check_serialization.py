"""Verify JSON identity on every measured V14 result; never a search input."""
import json,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
prior=R.parent/"synister_enhancement_v14_20260909"
while not (prior/"cohort_1200/manifest.json").exists():time.sleep(5)
sys.path.insert(0,str(R/"source"))
from synkit.Chem.Mapper.spectrum import ExactStructureSpectrum
selection=json.loads((R/"selection_1200.json").read_text())
entries=[]
for task in selection["tasks"]:
    key=f"{task['source_line']}_{task['mode']}"
    record=json.loads((prior/"cohort_1200"/(key+".json")).read_text())
    payload=record["result"]
    args=dict(payload["structure"])
    for field in ("its_class_counts","template_class_counts"):
        args[field]=tuple(tuple(row) for row in args[field])
    spectrum=ExactStructureSpectrum(**args)
    default=dict(payload,structure=spectrum.as_dict())
    immutable=dict(payload,structure=spectrum.as_dict(copy_sequences=False))
    a=json.dumps(payload,separators=(",",":"))
    b=json.dumps(default,separators=(",",":"))
    c=json.dumps(immutable,separators=(",",":"))
    assert a==b==c,key
    entries.append({"key":key,"json_sha256":hashlib.sha256(a.encode()).hexdigest()})
report={"records_checked":len(entries),"identical_json":True,
        "scope":"public result JSON; saved outputs used only for formatter equivalence, not enumeration",
        "records":entries}
(R/"serialization_equivalence.json").write_text(json.dumps(report,indent=2)+"\n")
print("All",len(entries),"public result payloads are byte-identical under both serializers",flush=True)
