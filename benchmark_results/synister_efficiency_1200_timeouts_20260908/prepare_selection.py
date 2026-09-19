import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

root = Path("/home/labhhc4/Documents/Workspace/Long/SynKit")
base = root / "benchmark_results/synister_global_shells_v4_120s_timeouts_w16"
prior = root / "benchmark_results/synister_efficiency_timeouts_v4_20260907"
manifest = json.loads((base / "manifest.json").read_text())
summary = json.loads((base / "summary.json").read_text())
assert summary["remaining_cases"] == summary["error_records"] == 0
assert summary["campaign_manifest_sha256"] == manifest["manifest_sha256"]

def digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")).hexdigest()

excluded = set()
for name in ("development", "heldout"):
    selection = json.loads((prior / (name + "_selection.json")).read_text())
    excluded.update(t["source_line"] for t in selection["tasks"])
dataset = base / "timeout_cases.csv.gz"
assert hashlib.sha256(dataset.read_bytes()).hexdigest() == selection["dataset_sha256"]
with gzip.open(dataset, "rt") as stream:
    rows = {int(row["source_line"]): row for row in csv.DictReader(stream)}
tasks = []
records = sorted((base / "cases").glob("line_*.json.gz"))
assert len(records) == manifest["rows"]
for path in records:
    with gzip.open(path, "rt") as stream:
        record = json.load(stream)
    claimed = record.pop("record_sha256")
    assert digest(record) == claimed, path
    assert record["campaign_manifest_sha256"] == manifest["manifest_sha256"]
    line = record["source_line"]
    assert rows[line]["reaction_id"] == record["reaction_id"]
    if line in excluded:
        continue
    for mode, shell in record["shells"].items():
        if shell["status"] != "timeout":
            continue
        tasks.append(dict(source_line=line, reaction_id=record["reaction_id"], mode=mode,
                          atom_count=record["atom_count"], historical_status="timeout",
                          historical_reason=shell["truncation_reason"],
                          historical_minimum_cost=shell["minimum_cost"],
                          historical_nodes=shell["visited_nodes"], historical_record_sha256=claimed))
seed = "synister-1200-expansion-20260908-v1"
tasks.sort(key=lambda t: hashlib.sha256((seed + ":" + str(t["source_line"]) + ":" + t["mode"]).encode()).hexdigest())
selected = tasks[:1200]
assert len(selected) == 1200
spec = importlib.util.spec_from_file_location("bench", root / "scripts/benchmark_synister_timeouts.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
expected = "687605c9da025b8eb7c7e4efbb4156c5e4a39bc68bfc608cb7e04b4fab5caf3c"
assert bench.implementation_hash(root) == bench.implementation_hash(prior / "dynamic_source") == expected
run = root / "benchmark_results/synister_efficiency_1200_timeouts_20260908"
run.mkdir(exist_ok=False)
shutil.copytree(prior / "dynamic_source", run / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy2(root / "scripts/benchmark_synister_timeouts.py", run / "benchmark_synister_timeouts.py")
shutil.copy2(__file__, run / "prepare_selection.py")
shutil.copy2(dataset, run / "timeout_cases.csv.gz")
assert bench.implementation_hash(run / "source") == expected
payload = dict(kind="synister_timeout_efficiency_cohort", source_campaign=str(base),
               source_manifest_sha256=manifest["manifest_sha256"], dataset=str(run / "timeout_cases.csv.gz"),
               dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
               selection="First 1200 shell tasks in deterministic SHA-256 order from historical 120-second timeout tasks, excluding all reaction source lines in the prior 64-task development and reused validation cohorts. Both modes eligible; mapping-limit timeouts retained and identified separately.",
               seed=seed, eligible_tasks=len(tasks), excluded_source_lines=sorted(excluded),
               selected_tasks=len(selected), selected_reactions=len({t["source_line"] for t in selected}),
               mode_counts=dict(collections.Counter(t["mode"] for t in selected)),
               historical_reason_counts=dict(collections.Counter(t["historical_reason"] for t in selected)),
               tasks=selected)
(run / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
unit = "synister-efficiency-1200-60s-20260908"
command = ["/home/labhhc4/anaconda3/envs/synfrag/bin/python", str(run / "benchmark_synister_timeouts.py"),
           "--selection", str(run / "selection.json"), "--source-root", str(run / "source"),
           "--output", str(run / "run"), "--seconds", "60", "--workers", "8"]
launch = dict(prepared_at=datetime.now().astimezone().isoformat(), service=unit + ".service", command=command,
              implementation_sha256=expected, runner_sha256=hashlib.sha256((run / "benchmark_synister_timeouts.py").read_bytes()).hexdigest(),
              validation="176 passed, 1 skipped, 1 known provenance test deselected; git diff --check passed")
(run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
print(json.dumps({k:v for k,v in payload.items() if k not in ("tasks", "excluded_source_lines")}, indent=2))
print(run)
