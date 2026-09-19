"""Three fresh runs per hard sentinel after a cohort CPU partition is free."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
cpus = None
while cpus is None:
    for part, affinity in enumerate(("0-15", "16-31")):
        try:
            done = json.loads((R / f"cohort_part_{part}/manifest.json").read_text()).get("source_unchanged")
        except (FileNotFoundError, json.JSONDecodeError):
            done = False
        if done:
            cpus = affinity
            break
    if cpus is None:
        time.sleep(5)
env = dict(os.environ, SYNKIT_NATIVE_PATTERN_CACHE="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
env.pop("SYNKIT_NATIVE_PROFILE", None)
for repeat in range(1, 4):
    command = ["taskset", "-c", cpus, sys.executable, str(R / "benchmark_synister_native.py"),
               "--selection", str(R / "selection_hard.json"), "--source-root", str(R / "source"),
               "--library", (R / "library.txt").read_text().strip(), "--output", str(R / f"hard_repeat_{repeat}"),
               "--seconds", "60", "--workers", "16", "--mapping-cap", "1000000", "--wall-budget", "--compact-json", "--immutable-json"]
    with (R / f"hard_repeat_{repeat}.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Fresh hard repeat {repeat}/3 finished", flush=True)
