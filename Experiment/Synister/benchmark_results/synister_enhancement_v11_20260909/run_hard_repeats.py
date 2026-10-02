import os,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
base=json.loads((R/"selection_hard_2.json").read_text())
launches=[]
for repeat in range(1,4):
    for case in (30474,22361):
        selection=dict(base,tasks=[t for t in base["tasks"] if t["source_line"]==case])
        path=R/f"selection_repeat_{case}.json"
        path.write_text(json.dumps(selection,indent=2)+"\n")
        name=f"hard_repeat_{repeat}_{case}"
        command=["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python",str(R/"benchmark_synister_native.py"),
             "--selection",str(path),"--source-root",str(R/"final_source"),
             "--library",(R/"library.txt").read_text().strip(),"--output",str(R/name),
             "--seconds","60","--workers","16","--mapping-cap","1000000","--wall-budget","--compact-json"]
        env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        launches.append({"name":name,"command":command,"concurrent_full_cohort_cpus":"0-15"})
        (R/"hard_launches.json").write_text(json.dumps(launches,indent=2)+"\n")
        with (R/(name+".log")).open("w") as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(name,result.returncode,flush=True)
        if result.returncode: raise RuntimeError(name)
