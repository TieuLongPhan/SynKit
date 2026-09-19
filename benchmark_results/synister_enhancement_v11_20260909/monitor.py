import json,time,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
last=-1
while True:
    path=R/"full_1200/case_timings.json"
    try: timings=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): timings=[]
    count=len(timings)
    if count!=last:
        lines=[]
        for line in (R/"full_1200.log").read_text().splitlines():
            try: lines.append(json.loads(line))
            except json.JSONDecodeError: pass
        if count>=last+40 or count==1200 or last<0:
            elapsed=sum(t["end_to_end_wall_seconds"] for t in timings)
            print(json.dumps({"processed":count,"complete":sum(x.get("complete",False) for x in lines),
                 "elapsed_case_seconds":elapsed,
                 "estimated_remaining_minutes":round(elapsed/max(1,count)*(1200-count)/60,1),
                 "incomplete":[x for x in lines if not x.get("complete")]}),flush=True)
            result=subprocess.run([sys.executable,str(R/"audit_results.py")],stdout=subprocess.DEVNULL)
            if result.returncode: print("audit failed",flush=True)
            last=count
    manifests=[R/"full_1200/manifest.json",*[R/f"hard_repeat_{i}_{case}/manifest.json" for i in range(1,4) for case in (30474,22361)]]
    try: done=all(json.loads(p.read_text()).get("source_unchanged") is True for p in manifests)
    except (FileNotFoundError,json.JSONDecodeError): done=False
    if done:
        subprocess.run([sys.executable,str(R/"audit_results.py")],check=True,stdout=subprocess.DEVNULL)
        print("ALL RUNS FINISHED; final audit written",flush=True)
        break
    time.sleep(5)
