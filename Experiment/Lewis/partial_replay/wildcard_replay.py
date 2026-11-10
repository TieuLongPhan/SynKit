"""Audit extraction with observed-root wildcards and replay in both directions.

This profile requires complete mapped carriers. Every selected source record,
including extraction and native refusals, remains in the report. Scores are
unranked self-replay, not held-out product prediction.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter

from rdkit import RDLogger

from .benchmark import ROOT, ReplayTimeout, canonical_fragments, deadline, digest, largest_fragment
from .fragment_metrics import METRICS, score_outputs
from synkit.Rule import WildcardRuleExtractionError, WildcardRuleExtractor
from synkit.Synthesis.Reactor import NativeReactorUnsupportedError, WildcardEngine, build_native_reactor

SETTINGS = None
MODES = ("full", "partial_all", "partial_largest")
DIRECTIONS = ("fw", "bw")


def worker_fingerprint(path):
    tree = ast.parse(Path(path).read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in {"worker_init", "execute", "process"}]
    return hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()


def restore_prefix(directory, source_hashes, dataset_hash):
    """Retain complete JSON records from an interrupted compressed stream."""
    manifest = json.loads((directory / "run-manifest.json").read_text())
    runner = str(Path(__file__).resolve().relative_to(ROOT))
    assert manifest["dataset_sha256"] == dataset_hash
    initial = directory / "initial-wildcard-replay.py"
    assert digest(initial) == manifest["source_sha256"][runner]
    assert worker_fingerprint(initial) == worker_fingerprint(__file__), "Worker semantics changed"
    for name, value in manifest["source_sha256"].items():
        if name != runner:
            assert source_hashes[name] == value, f"Source changed: {name}"
    old = directory / "records.jsonl.gz"
    stage = directory / "records.repaired.jsonl.gz"
    count = 0
    with gzip.open(old, "rt") as stream, gzip.open(stage, "wt") as output:
        while True:
            try:
                line = stream.readline()
            except (EOFError, gzip.BadGzipFile):
                break
            if not line:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            assert record["row"] == count, "Resume requires a contiguous complete-table prefix"
            output.write(line)
            count += 1
    number = len(list(directory.glob("records.interrupted-*.jsonl.gz"))) + 1
    old.rename(directory / f"records.interrupted-{number}.jsonl.gz")
    stage.rename(old)
    evidence = {"prefix_records": count, "initial_runner_sha256": digest(initial),
                "resumed_runner_sha256": digest(__file__),
                "worker_semantics_sha256": worker_fingerprint(__file__),
                "kernel_sources_unchanged": True,
                "elapsed_time_scope": "New segment only; no whole-table runtime claimed."}
    (directory / f"resume-{number}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return count, evidence


def worker_init(settings):
    global SETTINGS
    SETTINGS = settings
    logging.disable(logging.CRITICAL)
    RDLogger.DisableLog("rdApp.*")


def execute(rule, source, target, backend, library, direction, mode, args):
    selected = (largest_fragment(source),) if mode == "partial_largest" else source
    started = perf_counter()
    record = {"status": "error", "withheld_components": len(source) - len(selected),
              "input": ".".join(selected), **score_outputs((), target)}
    try:
        with deadline(args.seconds):
            def application(first=False):
                return WildcardEngine(
                    record["input"], rule, invert=direction == "bw", partial=mode != "full",
                    backend=backend, library_path=library, embed_threshold=args.candidate_limit,
                    electron_diagnostics=True, first_application=first,
                )
            engine = application(args.recovery_only)
            outputs = engine.smarts_list
            outcomes = {canonical_fragments(value.split(">>")[0 if direction == "bw" else 1])
                        for value in outputs}
            if args.recovery_only:
                record.update(score_outputs(outcomes, target))
                record["first_application_exact"] = record["exact_set"]
                record["full_search_retry"] = not record["exact_set"]
                if not record["exact_set"]:
                    engine = application()
                    outputs = engine.smarts_list
                    outcomes = {canonical_fragments(value.split(">>")[0 if direction == "bw" else 1])
                                for value in outputs}
            record.update(score_outputs(outcomes, target))
            record.update(status="complete", mapping_count=len(engine.mappings),
                          application_count=len(outputs), outcome_count=len(outcomes),
                          outcomes=[list(value) for value in sorted(outcomes)],
                          witness=next((value for value in outputs if
                              canonical_fragments(value.split(">>")[0 if direction == "bw" else 1]) == target), None),
                          electron_mismatches=sum(d["mismatch_count"] for d in engine.diagnostics),
                          rejected_completions=(engine.native_statistics.get("rejected_completions", 0)
                                                if backend == "cpp" else len(engine.partial_completion_failures)),
                          search_complete=engine.native_statistics.get("search_complete", True) if backend == "cpp" else True,
                          symbolic=any("*" in fragment for value in outcomes for fragment in value))
    except (ReplayTimeout, Exception) as error:
        record.update(status="timeout" if isinstance(error, ReplayTimeout) else
                      "native_refusal" if isinstance(error, NativeReactorUnsupportedError) else "error",
                      error_type=type(error).__name__, error=str(error))
    record["seconds"] = perf_counter() - started
    return record


def process(item):
    args, backends, library = SETTINGS
    i, row = item
    record = {"row": i, "r_id": row["r_id"], "reaction": row["aam"]}
    try:
        with deadline(args.seconds):
            extracted = WildcardRuleExtractor(endpoint_policy=args.endpoint_policy).extract(row["aam"])
            left, right = map(canonical_fragments, row["aam"].split(">>"))
        record.update(status="extracted", extraction=extracted.to_dict(),
                      rule_atoms=len(extracted.rule.rc.raw), runs={})
        for backend, mode, direction in itertools.product(backends, args.modes, DIRECTIONS):
            key = f"{backend}/{mode}/{direction}"
            record["runs"][key] = execute(
                extracted.rule, left if direction == "fw" else right,
                right if direction == "fw" else left, backend,
                library if backend == "cpp" else None, direction, mode, args,
            )
        comparison = Counter()
        if len(backends) == 2:
            for mode, direction in itertools.product(args.modes, DIRECTIONS):
                a, b = (record["runs"][f"{backend}/{mode}/{direction}"] for backend in backends)
                if a["status"] == b["status"] == "complete":
                    comparison["common_completed"] += 1
                    comparison["outcome_set_differences"] += a["outcomes"] != b["outcomes"]
                else:
                    comparison["status_differences"] += a["status"] != b["status"]
        record["backend_comparison"] = dict(comparison)
        # Compare complete sets before reducing the saved display sample.
        for value in record["runs"].values():
            if "outcomes" in value:
                encoded = json.dumps(value["outcomes"], separators=(",", ":")).encode()
                value["outcome_set_sha256"] = hashlib.sha256(encoded).hexdigest()
                if len(value["outcomes"]) > args.outcome_sample:
                    value["outcomes"] = value["outcomes"][:args.outcome_sample]
                    value["outcomes_truncated"] = True
    except (ReplayTimeout, Exception) as error:
        code = error.code if isinstance(error, WildcardRuleExtractionError) else type(error).__name__
        record.update(status="extraction_refused", code=code, error=str(error),
                      context=getattr(error, "context", {}))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "SynRXN/Data/tables/canonical/synthesis/uspto_50k.csv.gz")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--all", action="store_true", help="Process the complete source table.")
    parser.add_argument("--resume", action="store_true", help="Resume a source-checked complete-table prefix.")
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("both", "python", "cpp"), default="both")
    parser.add_argument("--endpoint-policy", choices=("closed", "observed"), default="closed")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--recovery-only", action="store_true",
                        help="Native first full-source witness, with full native retry only if it misses GT; no exhaustive-set claim.")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--candidate-limit", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--outcome-sample", type=int, default=3)
    args = parser.parse_args()
    if args.recovery_only and (args.backend != "cpp" or args.endpoint_policy != "observed" or args.modes != ["full"]):
        parser.error("--recovery-only requires --backend cpp --endpoint-policy observed --modes full")
    if min(args.limit, args.workers, args.progress_every, args.outcome_sample) < 1 or args.seconds <= 0 or args.candidate_limit < 1:
        parser.error("Limits must be positive.")
    if args.resume and (not args.all or args.id or (args.output / "summary.json").exists()):
        parser.error("Resume needs an incomplete --all run.")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new output directory to preserve earlier evidence.")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    RDLogger.DisableLog("rdApp.*")
    backends = ("python", "cpp") if args.backend == "both" else (args.backend,)
    library = str(build_native_reactor(ROOT / "build/partial/native")) if "cpp" in backends else None
    source_files = [ROOT / "synkit/Rule/wildcard.py", ROOT / "synkit/Rule/syn_rule.py",
                    ROOT / "synkit/Graph/Morphism/constraints.py", ROOT / "synkit/Graph/Morphism/llg.py",
                    ROOT / "synkit/Rule/__init__.py", Path(__file__).resolve(),
                    Path(__file__).with_name("fragment_metrics.py"), Path(__file__).with_name("benchmark.py")]
    source_files += list((ROOT / "synkit/Synthesis/Reactor").rglob("*.py"))
    source_files += list((ROOT / "synkit/Synthesis/Reactor/cpp").rglob("*.cpp"))
    source_files += list((ROOT / "synkit/Synthesis/Reactor/cpp").rglob("*.h"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    dataset_hash = digest(args.dataset)
    tallies, refusals, comparison = defaultdict(Counter), Counter(), Counter()
    counts = Counter()
    ids = set()
    started = perf_counter()
    resume_evidence = None
    prefix = 0
    if args.resume:
        prefix, resume_evidence = restore_prefix(args.output, hashes, dataset_hash)
        with gzip.open(args.output / "records.jsonl.gz", "rt") as saved:
            for line in saved:
                record = json.loads(line)
                counts["source_rows"] += 1
                ids.add(record["r_id"])
                if record["status"] == "extracted":
                    counts["extracted"] += 1
                    counts["with_ports"] += bool(record["extraction"]["ports"])
                    comparison.update(record["backend_comparison"])
                    for key, value in record["runs"].items():
                        tally = tallies[key]
                        tally["attempted"] += 1
                        tally[value["status"]] += 1
                        tally["withheld_cases"] += value["withheld_components"] > 0
                        tally["symbolic_cases"] += value.get("symbolic", False)
                        tally["electron_mismatch_cases"] += value.get("electron_mismatches", 0) > 0
                        for metric in METRICS:
                            tally[metric] += value[metric]
                else:
                    refusals[record["code"]] += 1
        print(json.dumps({"resumed_prefix": prefix}), file=sys.stderr, flush=True)
    else:
        (args.output / "run-manifest.json").write_text(json.dumps({
        "source_sha256": hashes, "dataset_sha256": dataset_hash,
        "dataset": str(args.dataset.resolve()), "workers": args.workers,
        "seconds_per_run": args.seconds, "candidate_limit": args.candidate_limit,
        "endpoint_policy": args.endpoint_policy, "modes": args.modes,
        "recovery_only": args.recovery_only,
        }, indent=2, sort_keys=True) + "\n")
    pool = mp.Pool(args.workers, initializer=worker_init, initargs=((args, backends, library),)) if args.workers > 1 else None
    worker_init((args, backends, library))
    with gzip.open(args.dataset, "rt") as stream:
        rows = enumerate(csv.DictReader(stream))
        selected = ((i, row) for i, row in rows if row["r_id"] in args.id) if args.id else itertools.islice(rows, None if args.all else args.limit)
        selected = itertools.islice(selected, prefix, None)
        processed = pool.imap(process, selected, chunksize=8) if pool else map(process, selected)
        with gzip.open(args.output / "records.jsonl.gz", "at" if args.resume else "wt") as output:
            for record in processed:
                counts["source_rows"] += 1
                ids.add(record["r_id"])
                output.write(json.dumps(record, sort_keys=True) + "\n")
                if record["status"] == "extracted":
                    counts["extracted"] += 1
                    counts["with_ports"] += bool(record["extraction"]["ports"])
                    comparison.update(record["backend_comparison"])
                    for key, value in record["runs"].items():
                        tally = tallies[key]
                        tally["attempted"] += 1
                        tally[value["status"]] += 1
                        tally["withheld_cases"] += value["withheld_components"] > 0
                        tally["symbolic_cases"] += value.get("symbolic", False)
                        tally["electron_mismatch_cases"] += value.get("electron_mismatches", 0) > 0
                        for metric in METRICS:
                            tally[metric] += value[metric]
                else:
                    refusals[record["code"]] += 1
                if counts["source_rows"] % args.progress_every == 0:
                    print(json.dumps({**dict(counts), "elapsed_seconds": round(perf_counter() - started, 1)}), file=sys.stderr, flush=True)
    if pool:
        pool.close()
        pool.join()
    if args.id and set(args.id) != ids:
        raise ValueError("One or more requested record IDs were not found.")
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in source_files}, "Sources changed during replay"
    assert digest(args.dataset) == dataset_hash
    summary = {"schema": "synkit.wildcard-extraction-replay/1", "source_rows": counts["source_rows"],
               "selection": {"first_rows": None if args.id or args.all else args.limit, "ids": args.id},
               "extracted": counts["extracted"], "with_ports": counts["with_ports"],
               "extraction_refusals": dict(refusals),
               "metrics": {key: {"source_rows": counts["source_rows"], **dict(value)} for key, value in tallies.items()},
               "backend_comparison": dict(comparison), "elapsed_seconds": perf_counter() - started,
               "resume": resume_evidence,
               "source_sha256": hashes, "dataset_sha256": dataset_hash,
               "settings": {"seconds": args.seconds, "candidate_limit": args.candidate_limit,
                            "endpoint_policy": args.endpoint_policy, "modes": args.modes,
                            "recovery_only": args.recovery_only,
                            "hydrogens": "absolute", "domains": "observed_root", "workers": args.workers,
                            "saved_outcome_sample": args.outcome_sample,
                            "comparison": ("Native witness recovery; no exhaustive outcome-set comparison. GT is used only by the scorer to decide whether a full native retry is needed."
                                           if args.recovery_only else "Complete outcome sets compared in each worker before saving samples.")},
               "interpretation": "Unranked self-replay with maps removed from hosts and scores; all refusals retained; no wildcard expansion counts as exact chemical identity. Search completeness is reported separately from recovery."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"source_sha256", "dataset_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
