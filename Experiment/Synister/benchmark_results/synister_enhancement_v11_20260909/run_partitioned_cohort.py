"""Continue one frozen cohort in disjoint per-case CPU partitions."""
import json,os,subprocess,time,hashlib,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parent
out=R/"cohort_1200"
out.mkdir()
base=json.loads((R/"selection_1200.json").read_text())
expected={(t["source_line"],t["mode"]) for t in base["tasks"]}
env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
processes=[];logs=[];launches=[]
started=time.perf_counter()
for part,cpus in enumerate(("0-15","16-31")):
    command=["taskset","-c",cpus,sys.executable,str(R/"benchmark_synister_native.py"),
        "--selection",str(R/f"selection_part_{part}.json"),
        "--source-root",str(R/"final_source"),"--library",(R/"library.txt").read_text().strip(),
        "--output",str(R/f"cohort_part_{part}"),"--seconds","60","--workers","16",
        "--mapping-cap","1000000","--wall-budget","--compact-json"]
    log=(R/f"cohort_part_{part}.log").open("w")
    logs.append(log)
    processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
    launches.append({"part":part,"pid":processes[-1].pid,"command":command})
(R/"cohort_launches.json").write_text(json.dumps(launches,indent=2)+"\n")
origins={};timings={};last=-40
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
while True:
    for directory in (R/"cohort_prefix",R/"cohort_part_0",R/"cohort_part_1"):
        for timing in read(directory/"case_timings.json") or []:
            key=(timing["source_line"],timing["mode"])
            name=f"{key[0]}_{key[1]}.json"
            if key in origins:
                assert origins[key]==str(directory/name),("overlap",key)
                continue
            doc=read(directory/name)
            if doc is None:continue
            assert key in expected and (doc["source_line"],doc["mode"])==key
            shutil.copy(directory/name,out/name)
            origins[key]=str(directory/name);timings[key]=timing
    ordered=[timings[(t["source_line"],t["mode"])] for t in base["tasks"]
             if (t["source_line"],t["mode"]) in timings]
    (out/"case_timings.json").write_text(json.dumps(ordered,indent=2)+"\n")
    (out/"origins.json").write_text(json.dumps(
        [{"source_line":k[0],"mode":k[1],"record":v} for k,v in origins.items()],indent=2)+"\n")
    count=len(origins)
    if count>=last+40 or (count==1200 and last!=1200):
        completed=0;failed=[]
        for key in origins:
            d=read(out/f"{key[0]}_{key[1]}.json")
            if d.get("result",{}).get("complete"):completed+=1
            else:failed.append({"key":key,"error":d.get("error"),"reason":d.get("result",{}).get("truncation_reason")})
        print(json.dumps({"processed":count,"complete":completed,"incomplete":failed,
             "parallel_phase_seconds":time.perf_counter()-started}),flush=True)
        subprocess.run([sys.executable,str(R/"audit_results.py")],stdout=subprocess.DEVNULL,check=True)
        last=count
    if all(p.poll() is not None for p in processes):
        assert all(p.returncode==0 for p in processes),[p.returncode for p in processes]
        if count<1200:
            # Files can have closed immediately after this iteration's snapshot.
            time.sleep(1)
            continue
        break
    time.sleep(5)
assert set(origins)==expected
manifests=[read(R/name/"manifest.json") for name in ("cohort_prefix","cohort_part_0","cohort_part_1")]
assert all(m["source_unchanged"] for m in manifests)
assert len({m["source_sha256_python_and_cpp"] for m in manifests})==1
manifest=dict(manifests[0])
manifest.update(execution_status="complete",tasks_run_sequentially=False,retained_tasks=1200,
    original_output=None,partition_directories=["cohort_prefix","cohort_part_0","cohort_part_1"],
    concurrent_case_limit=2,total_physical_cpu_limit=32,
    per_case_physical_cpu_limit=16,parallel_phase_seconds=time.perf_counter()-started,
    selection_sha256=hashlib.sha256((R/"selection_1200.json").read_bytes()).hexdigest(),
    coordinator_cpu_affinity=None,partition_cpu_affinities=[[0,15],[16,31]])
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
subprocess.run([sys.executable,str(R/"audit_results.py")],stdout=subprocess.DEVNULL,check=True)
print("ALL 1200 TASKS FINISHED; exact audit written",flush=True)
