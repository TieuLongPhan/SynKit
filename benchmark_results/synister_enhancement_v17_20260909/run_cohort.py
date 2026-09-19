"""Fresh original 1,200 tasks, two disjoint 16-CPU partitions, no reused results."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
selection = json.loads((R / "selection_1200.json").read_text())
expected = {(t["source_line"], t["mode"]) for t in selection["tasks"]}
assert len(expected) == len(selection["tasks"]) == 1200
out = R / "cohort_1200"
out.mkdir()
env = dict(os.environ, SYNKIT_NATIVE_PATTERN_CACHE="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
env.pop("SYNKIT_NATIVE_PROFILE", None)
started = time.perf_counter()
processes, logs, launches = [], [], []
for part, cpus in enumerate(("0-15", "16-31")):
    command = ["taskset", "-c", cpus, sys.executable, str(R / "benchmark_synister_native.py"),
               "--selection", str(R / f"selection_part_{part}.json"), "--source-root", str(R / "source"),
               "--library", (R / "library.txt").read_text().strip(), "--output", str(R / f"cohort_part_{part}"),
               "--seconds", "60", "--workers", "16", "--mapping-cap", "1000000", "--wall-budget", "--compact-json", "--immutable-json"]
    log = (R / f"cohort_part_{part}.log").open("w")
    logs.append(log)
    processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
    launches.append(dict(part=part, pid=processes[-1].pid, command=command))
(R / "cohort_launches.json").write_text(json.dumps(launches, indent=2) + "\n")
last = -1
while True:
    timings = []
    for part in range(2):
        try:
            timings += json.loads((R / f"cohort_part_{part}/case_timings.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    if len(timings) // 20 != last:
        last = len(timings) // 20
        print(json.dumps(dict(processed=len(timings), strict_complete=sum(t["complete_below_60_seconds"] for t in timings),
                              over_budget=[(t["source_line"], t["mode"], t["end_to_end_wall_seconds"]) for t in timings if t["end_to_end_wall_seconds"] >= 60],
                              wall_seconds=time.perf_counter() - started)), flush=True)
    if all(p.poll() is not None for p in processes):
        break
    time.sleep(5)
assert all(p.returncode == 0 for p in processes), [p.returncode for p in processes]
seen, timings, origins, manifests = set(), [], [], []
for part in range(2):
    directory = R / f"cohort_part_{part}"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_unchanged"]
    manifests.append(manifest)
    for timing in json.loads((directory / "case_timings.json").read_text()):
        key = timing["source_line"], timing["mode"]
        assert key in expected and key not in seen
        seen.add(key)
        name = f"{key[0]}_{key[1]}.json"
        # Separate immutable evidence files; never substitute an earlier run.
        shutil.copyfile(directory / name, out / name)
        timings.append(timing)
        origins.append(dict(source_line=key[0], mode=key[1], record=str(directory / name)))
assert seen == expected
assert len({m["source_sha256_python_and_cpp"] for m in manifests}) == 1
manifest = dict(manifests[0], execution_status="complete", tasks_run_sequentially=False, retained_tasks=1200,
                concurrent_case_limit=2, total_physical_cpu_limit=32, per_case_physical_cpu_limit=16,
                parallel_phase_seconds=time.perf_counter() - started,
                selection_sha256=hashlib.sha256((R / "selection_1200.json").read_bytes()).hexdigest(),
                partition_cpu_affinities=[[0,15], [16,31]])
(out / "case_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
(out / "origins.json").write_text(json.dumps(origins, indent=2) + "\n")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
subprocess.run([sys.executable, str(R / "audit_results.py")], check=True)
print("ALL 1200 FRESH TASKS FINISHED AND AUDITED", flush=True)
