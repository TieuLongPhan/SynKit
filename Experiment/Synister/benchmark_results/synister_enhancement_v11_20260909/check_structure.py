import csv,gzip,json,time
from pathlib import Path
from synkit.Chem.Mapper import GlobalShellConfig,blinded_mapped_reaction_problem
from synkit.Chem.Mapper.analysis import analyze_reference_blinded_global_shell
R=Path(__file__).resolve().parent
selection=json.loads((R/"selection_1200.json").read_text())
with gzip.open(selection["dataset"],"rt") as f:
    row=next(row for row in csv.DictReader(f) if int(row["source_line"])==13067)
reaction=row["mapped_reaction"].rsplit("|",1)[0]
problem=blinded_mapped_reaction_problem(reaction,heavy_only=True,blind_seed="synister-global-v1")
records={}
for backend in ("python","native_certificate"):
    started=time.perf_counter()
    result=analyze_reference_blinded_global_shell(
        problem.lgp,problem.reference_mapping,target_mode="minimal",
        config=GlobalShellConfig(time_limit_seconds=60,structure_timeout_seconds=5 if backend=="python" else .25,
             structure_native_library_path=None if backend=="python" else (R/"library.txt").read_text().strip()))
    records[backend]={"result":result.as_dict(),"wall_seconds":time.perf_counter()-started}
(R/"13067_independent_classification.json").write_text(json.dumps(records,indent=2)+"\n")
a,b=(records[k]["result"] for k in ("python","native_certificate"))
assert a["complete"] and b["complete"] and a["structure"]["complete"] and b["structure"]["complete"]
assert a["structure"]==b["structure"]
print({k:{"complete":v["result"]["structure"]["complete"],"wall_seconds":v["wall_seconds"]} for k,v in records.items()},flush=True)
