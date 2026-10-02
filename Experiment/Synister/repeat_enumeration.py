"""Three serial repeats of matched minimum enumeration on an input-selected subset."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

from Experiment.Synister.ablation_benchmark import compare_records, select_inputs
from Experiment.Synister.audit_enumeration_benchmark import audit
from Experiment.Synister.enumeration_benchmark import encode, execute


def make_tasks(rows, *, repeats, seconds, memory_gib, max_maps):
    result = []
    for repeat in range(repeats):
        for index, row in enumerate(rows):
            methods = ("synister", "milp") if (index+repeat) % 2 == 0 else ("milp", "synister")
            for method in methods:
                result.append({"task_id": f'{row["benchmark_id"]}.minimum.r{repeat}.{method}',
                               "benchmark_id": row["benchmark_id"], "reaction": row["reaction"],
                               "query": "minimum", "repeat": repeat, "method": method,
                               "target_doubled_cd": "minimal", "seconds": seconds,
                               "memory_gib": memory_gib, "max_maps": max_maps})
    return result


def run(args):
    prior = json.loads((args.matched/"audit.json").read_text())
    if not prior["all_output_comparisons_consistent"]:
        raise ValueError("The matched study must pass its saved-output audit first")
    if args.after is not None:
        preceding = json.loads((args.after/"audit.json").read_text())
        if not preceding["all_verified"]:
            raise ValueError("Preceding timing-control study has not passed its audit")
    rows = select_inputs(args.matched, args.per_bin)
    tasks = make_tasks(rows, repeats=args.repeats, seconds=args.seconds,
                       memory_gib=args.memory_gib, max_maps=args.max_maps)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"cases").mkdir()
    (args.output/"maps").mkdir()
    paths = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    paths += [Path(__file__), Path("Experiment/Synister/enumeration_benchmark.py"),
              Path("Experiment/Synister/enumeration_worker.py"), Path("Experiment/Synister/global_milp.py"),
              Path("Experiment/Synister/ablation_benchmark.py"), Path("Experiment/Synister/ablation_worker.py"),
              Path("Experiment/Synister/audit_enumeration_benchmark.py")]
    for name, value in {"inputs.json": rows, "minimum_tasks.json": tasks, "numeric_tasks.json": [],
                        "sources.json": {str(p): p.read_text() for p in paths}}.items():
        (args.output/name).write_text(encode(value))
    manifest = {"schema": "synister.repeated-matched-enumeration.v1", "python": sys.version,
                "platform": platform.platform(), "dependencies": {name: version(name)
                    for name in ("numpy", "scipy", "networkx", "rdkit")},
                "seconds": args.seconds, "external_seconds": args.seconds+15,
                "memory_gib": args.memory_gib, "max_maps": args.max_maps, "workers": 1,
                "threads_per_worker": 1, "methods": ["synister", "milp"], "repeats": args.repeats,
                "sources_sha256": sha256((args.output/"sources.json").read_bytes()).hexdigest(),
                "inputs_sha256": sha256((args.output/"inputs.json").read_bytes()).hexdigest(),
                "parent_inputs_sha256": sha256((args.matched/"inputs.json").read_bytes()).hexdigest(),
                "parent_audit_sha256": sha256((args.matched/"audit.json").read_bytes()).hexdigest(),
                "selection": "First input hashes per source and atom-count bin; no completion filtering",
                "per_bin": args.per_bin, "parent_study": str(args.matched),
                "atom_order": "Same saved endpoint order in every repeat",
                "method_order": "Alternates by case and repeat; fresh worker for every attempt",
                "seed": "No external seed for either method", "output_unit": "all_indexed_atom_maps",
                "query_scope": "Minimum proof followed by complete indexed-map enumeration",
                "after": str(args.after) if args.after else None}
    (args.output/"manifest.json").write_text(encode(manifest))
    records = [execute(task, args.output) for task in tasks]
    comparisons = compare_records(records, args.output)
    summary = {"selected": len(rows), "attempts": len(records), "repeats": args.repeats,
               "by_method": {method: dict(Counter(r["termination"] for r in records if r["task"]["method"] == method))
                             for method in ("synister", "milp")},
               "all_repeat_comparisons_consistent": all(c["consistent"] for c in comparisons),
               "repeat_comparisons": comparisons,
               "all_record_hashes": {p.name: sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((args.output/"cases").glob("*.json"))}}
    (args.output/"summary.json").write_text(encode(summary))
    checked = audit(args.output)
    checked.update(all_repeat_comparisons_consistent=summary["all_repeat_comparisons_consistent"],
                   repeats=args.repeats, repeat_comparison_count=len(comparisons))
    (args.output/"audit.json").write_text(encode(checked))
    print(encode({k: v for k, v in checked.items() if k != "comparisons"}))
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--per-bin", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--memory-gib", type=int, default=6)
    parser.add_argument("--max-maps", type=int, default=100000)
    args = parser.parse_args()
    if min(args.per_bin, args.repeats, args.seconds, args.memory_gib, args.max_maps) <= 0:
        parser.error("Counts and limits must be positive")
    result = run(args)
    raise SystemExit(0 if result["all_output_comparisons_consistent"] and result["all_repeat_comparisons_consistent"] else 1)
