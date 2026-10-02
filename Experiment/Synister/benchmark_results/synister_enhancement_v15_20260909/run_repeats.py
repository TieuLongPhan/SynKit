import os,json,subprocess,time,sys
from pathlib import Path
R=Path(__file__).resolve().parent
cpus=None
while cpus is None:
    for part,affinity in enumerate(("0-15","16-31")):
        try:done=json.loads((R.parent/f"synister_enhancement_v14_20260909/cohort_part_{part}/manifest.json").read_text()).get("source_unchanged")
        except (FileNotFoundError,json.JSONDecodeError):done=False
        if done:
            cpus=affinity
            break
    if cpus is None:time.sleep(5)
base=json.loads((R/"selection_1200.json").read_text())
launches=[]
for repeat in range(1,4):
    for case,mode in ((17156,"minimal"),(30474,"reference_cd"),(22361,"reference_cd")):
        selection=dict(base,tasks=[t for t in base["tasks"] if (t["source_line"],t["mode"])==(case,mode)])
        path=R/f"selection_repeat_{case}.json"
        path.write_text(json.dumps(selection,indent=2)+"\n")
        name=f"hard_repeat_{repeat}_{case}"
        command=["taskset","-c",cpus,sys.executable,str(R/"benchmark_synister_native.py"),
             "--selection",str(path),"--source-root",str(R/"source"),
             "--library",(R/"library.txt").read_text().strip(),"--output",str(R/name),
             "--seconds","60","--workers","16","--mapping-cap","1000000","--wall-budget","--compact-json","--immutable-json"]
        env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        launches.append({"name":name,"command":command})
        (R/"hard_launches.json").write_text(json.dumps(launches,indent=2)+"\n")
        with (R/(name+".log")).open("w") as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(name,result.returncode,flush=True)
        if result.returncode: raise RuntimeError(name)
