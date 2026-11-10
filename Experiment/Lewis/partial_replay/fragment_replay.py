"""Replay observed USPTO endpoints and score exact molecular-fragment recovery.

The Python SynReactor observation profile admits one-sided endpoint records.
Full and component-subset matching are measured separately without a fallback.
This is not an extension of the closed typed Python/C++ implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
from time import perf_counter

from rdkit import rdBase

from .benchmark import (ROOT, ReplayTimeout, canonical_fragments, deadline, digest,
                        initialize, prepare)
from .fragment_metrics import METRICS, score_outputs
from synkit.Synthesis.Reactor import SynReactor, WildcardCompletionError

SETTINGS = {}
MODES = ("full", "partial")
DIRECTIONS = ("fw", "bw")


class EnumerationLimit(RuntimeError):
    pass


def worker_init(settings):
    global SETTINGS
    SETTINGS = settings
    initialize({})


def execute(rule, fragments, target, mode, direction):
    result = {name: False for name in METRICS}
    result["gt_molecule_recall"] = 0.0
    started = perf_counter()
    try:
        with deadline(SETTINGS["seconds"]):
            engine = SynReactor(
                ".".join(fragments), rule, partial=mode == "partial", invert=direction == "bw",
                explicit_h=False, automorphism=False, dedup_its=False, stereo_mode="ignore",
                # Legacy thresholds truncate search and partial result counts.
                # Enumerate without that heuristic, then enforce a visible cap.
                embed_threshold=None,
            )
            mappings = engine.mappings
            if len(mappings) > SETTINGS["candidate_limit"]:
                raise EnumerationLimit("Complete mapping population exceeds the declared cap")
            reactions = engine.smarts_list
            outcomes = {canonical_fragments(s.split(">>")[0 if direction == "bw" else 1])
                        for s in reactions}
            scores = score_outputs(outcomes, target)
            result.update(scores)
            result.update(status="complete", mappings=len(mappings), applications=len(reactions),
                          unique_outcomes=len(outcomes),
                          rejected_partial_completions=len(engine.partial_completion_failures),
                          outcome_sample=[".".join(value) for value in sorted(outcomes)[:3]],
                          output_molecules_with_wildcards=any("*" in f for o in outcomes for f in o))
    except (ReplayTimeout, Exception) as error:
        result.update(status=("timeout" if isinstance(error, ReplayTimeout)
                              else "search_limit" if isinstance(error, EnumerationLimit)
                              else "completion_error" if isinstance(error, WildcardCompletionError)
                              else "error"), error_type=type(error).__name__, error=str(error))
    result["seconds"] = perf_counter() - started
    return result


def process(item):
    position, row = item
    result = {"row": position, "r_id": row["r_id"], "split": row.get("split", "")}
    try:
        with deadline(SETTINGS["seconds"]):
            rule, inputs, targets, _, metadata = prepare(row["aam"], "absolute")
        result.update(metadata, status="attempted", targets={k: ".".join(v) for k, v in targets.items()})
        result["runs"] = {mode: {d: execute(rule, inputs[d], targets[d], mode, d) for d in DIRECTIONS}
                          for mode in MODES}
    except (ReplayTimeout, Exception) as error:
        result.update(status="preparation_timeout" if isinstance(error, ReplayTimeout) else "preparation_error",
                      error_type=type(error).__name__, error=str(error))
    return result


def summarize(records):
    metrics = defaultdict(Counter)
    errors = Counter()
    for record in records:
        for mode in MODES:
            for direction in DIRECTIONS:
                tally = metrics[mode + "/" + direction]
                tally["source_rows"] += 1
                if record["status"] != "attempted":
                    tally[record["status"]] += 1
                    continue
                run = record["runs"][mode][direction]
                tally["attempted"] += 1
                tally[run["status"]] += 1
                for name in METRICS:
                    tally[name] += run[name]
                tally["gt_molecule_recall_sum"] += run["gt_molecule_recall"]
                tally["rejected_partial_completions"] += run.get("rejected_partial_completions", 0)
                if run["status"] != "complete":
                    errors[(mode, direction, run["status"], run.get("error_type"), run.get("error"))] += 1
    return {"rows": len(records), "metrics": {k: dict(v) for k, v in sorted(metrics.items())},
            "errors": [{"mode": k[0], "direction": k[1], "status": k[2], "type": k[3],
                        "error": k[4], "count": v} for k, v in errors.most_common()]}


def write_report(output, report):
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = ("mode", "direction", "source_rows", "complete", *METRICS,
              "completion_error", "error", "timeout", "search_limit")
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, values in report["metrics"].items():
            mode, direction = key.split("/")
            writer.writerow({"mode": mode, "direction": direction,
                             **{k: values.get(k, 0) for k in fields[2:]}})
    lines = ["USPTO molecular-fragment replay", "==============================", "",
             f"Source records: {report['rows']:,}. Every source row remains in the denominator.",
             "The Python SynReactor observation profile admits atoms recorded on only one side.",
             "Full matching and partial subset matching are separate runs; neither falls back to the other.",
             "The closed typed Python/C++ profile is unchanged and is not the engine measured here.", "",
             "Each reaction supplies its own extracted rule. Hosts have no reference atom maps.",
             "Scoring ignores stereo and map labels, preserving charges, isotopes and molecule counts.",
             "An exact molecule identity is required; neither a substructure nor an unresolved wildcard",
             "counts as a match to a concrete GT molecule. Metrics use a single output application.", "",
             "exact_set: Complete predicted and GT component multisets are equal.",
             "all_gt_molecules: Every GT component occurs, allowing additional output molecules.",
             "any_gt_molecule: At least one exact GT molecule occurs (may be a spectator).",
             "largest_gt_recovered: A largest GT molecule occurs anywhere in the output.",
             "max_fragment_match: A largest output molecule equals a largest GT molecule.",
             "Largest means heavy-atom count; tied maxima are compared as identity sets.", "",
             ".. csv-table:: Recovery counts", "   :file: summary.csv", "   :header-rows: 1", "",
             "Errors, timeouts and search caps retain failed scores. Mapping enumeration uses no",
             "legacy truncation heuristic; the declared cap is checked before rewriting.",
             "In the observation profile, invalid proper-subset completions are rejected and",
             "counted while valid applications are retained. Failure of a full input binding,",
             "failure of every completion, search exhaustion and serialization errors still raise.", ""]
    (output / "report.rst").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT.parent / "SynRXN/Data/tables/canonical/synthesis/uspto_50k.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--candidate-limit", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds <= 0 or args.candidate_limit < 1:
        parser.error("workers, seconds and candidate-limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to retain prior evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.dataset, "rt") as stream:
        rows = list(enumerate(itertools.islice(csv.DictReader(stream), args.limit)))
    paths = list((ROOT / "synkit/Synthesis/Reactor").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("fragment_metrics.py"),
              Path(__file__).with_name("benchmark.py"), ROOT / "synkit/Rule/syn_rule.py",
              ROOT / "synkit/IO/chem_converter.py", ROOT / "synkit/Graph/ITS/rc_extractor.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    metadata = {"schema": "synkit.uspto-fragment-replay/1", "command": sys.argv,
                "engine": "Python SynReactor observation profile", "modes": list(MODES),
                "source": {"path": str(args.dataset.resolve()), "sha256": digest(args.dataset),
                           "selected_rows": len(rows), "selection": "prefix in stored order; full table if no limit"},
                "python": sys.version, "rdkit": rdBase.rdkitVersion, "platform": platform.platform(),
                "source_sha256": hashes, "settings": {"seconds": args.seconds,
                "candidate_limit": args.candidate_limit, "workers": args.workers, "hydrogen": "absolute"}}
    (args.output / "running.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records, started = [], perf_counter()
    with gzip.open(args.output / "records.jsonl.gz", "wt") as stream:
        with mp.get_context("spawn").Pool(args.workers, worker_init, (metadata["settings"],)) as pool:
            for record in pool.imap(process, rows, chunksize=8):
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                if len(records) % args.progress_every == 0 or len(records) == len(rows):
                    print(f"{len(records)}/{len(rows)} elapsed={perf_counter()-started:.1f}s", flush=True)
                    stream.flush()
    report = {**metadata, **summarize(records), "wall_seconds": perf_counter() - started,
              "source_unchanged": all(digest(ROOT / p) == h for p, h in hashes.items()),
              "records_sha256": digest(args.output / "records.jsonl.gz")}
    write_report(args.output, report)
    (args.output / "running.json").unlink()
    print(json.dumps({"metrics": report["metrics"], "errors": report["errors"]}, indent=2))


if __name__ == "__main__":
    main()
