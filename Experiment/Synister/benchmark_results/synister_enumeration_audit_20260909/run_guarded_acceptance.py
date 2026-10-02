"""Strict fresh-process checks of both hard cases with the post-audit input guard."""
import os,time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
while True:
    try:done=json.loads((R/"native_hard_cache_off_diagnostic/manifest.json").read_text()).get("source_unchanged")
    except (FileNotFoundError,json.JSONDecodeError):done=False
    if done:break
    time.sleep(5)
base=json.loads((R/"selection_hard_2.json").read_text())
launches=[]
for case in (30474,22361):
    selection=dict(base,tasks=[t for t in base["tasks"] if t["source_line"]==case])
    path=R/f"selection_guarded_{case}.json"
    path.write_text(json.dumps(selection,indent=2)+"\n")
    name=f"guarded_60s_{case}"
    command=["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python",str(R/"benchmark_synister_native.py"),"--selection",str(path),"--source-root",str(R/"validated_source"),"--library",(R/"portable_library.txt").read_text().strip(),"--output",str(R/name),"--seconds","60","--workers","16","--mapping-cap","1000000","--wall-budget","--compact-json"]
    env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    launches.append({"case":case,"command":command,"pattern_cache":0})
    (R/"guarded_acceptance_launches.json").write_text(json.dumps(launches,indent=2)+"\n")
    with (R/(name+".log")).open("w") as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0
