"""Replay each USPTO record with its own extracted reaction-center rule.

This is a rule-execution audit, not held-out product prediction. The source
table is read without modification; every selected row remains in the report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import gzip
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
from pathlib import Path
import platform
import signal
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger, rdBase  # noqa: E402
from synkit.IO.chem_converter import rsmi_to_its  # noqa: E402
from synkit.Rule import SynRule  # noqa: E402
from synkit.Synthesis.Reactor import (  # noqa: E402
    NativeReactorUnsupportedError, WildcardCompletionError, WildcardEngine,
    WildcardSearchLimitError, build_native_reactor,
)

VARIANTS = ("full_control", "partial_all", "partial_largest")
DIRECTIONS = ("fw", "bw")
SETTINGS = {}


class ReplayTimeout(BaseException):
    """Interrupt Python preparation/serialization across chemistry handlers."""


def _alarm(_signum, _frame):
    raise ReplayTimeout("Replay exceeded the configured time budget")


@contextmanager
def deadline(seconds):
    """Interrupt on POSIX; otherwise reject work that finishes over budget."""
    if not all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")):
        started = perf_counter()
        yield
        if perf_counter() - started >= seconds:
            _alarm(None, None)
        return
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_fragments(side, *, stereo=False):
    """Canonical component multiset, preserving charges, isotopes and counts."""
    molecule = Chem.MolFromSmiles(side)
    if molecule is None:
        raise ValueError("An endpoint could not be parsed")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    if not stereo:
        Chem.RemoveStereochemistry(molecule)
    return tuple(sorted(Chem.MolToSmiles(fragment, isomericSmiles=True)
                        for fragment in Chem.GetMolFrags(molecule, asMols=True)))


def contains_fragments(actual, expected):
    return not (Counter(expected) - Counter(actual))


def largest_fragment(fragments):
    return min(fragments, key=lambda s: (-Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s))


def initialize(settings):
    global SETTINGS
    SETTINGS = settings
    logging.disable(logging.CRITICAL)
    RDLogger.DisableLog("rdApp.*")


def prepare(reaction, hydrogen_policy):
    sides = reaction.split(">>")
    if len(sides) != 2 or not all(sides):
        raise ValueError("Expected two nonempty reaction endpoints")
    mapped_sides, map_sets = [], []
    for side in sides:
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            raise ValueError("An endpoint could not be parsed")
        maps = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
        if any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
            raise ValueError("Reference atom maps must be positive and unique per endpoint")
        map_sets.append(set(maps))
        Chem.RemoveStereochemistry(molecule)
        mapped_sides.append(Chem.MolToSmiles(molecule))
    # The maps are used only to EXTRACT the rule, never to select a host binding.
    core = rsmi_to_its(
        ">>".join(mapped_sides), core=True, format="tuple",
        drop_non_aam=False, use_index_as_atom_map=True,
        include_context_edges=True,
    )
    rule = SynRule(core, canon=False, format="tuple",
                   implicit_h=hydrogen_policy == "relative")
    targets = {"fw": canonical_fragments(sides[1]), "bw": canonical_fragments(sides[0])}
    strict_targets = {"fw": canonical_fragments(sides[1], stereo=True),
                      "bw": canonical_fragments(sides[0], stereo=True)}
    inputs = {"fw": canonical_fragments(sides[0]), "bw": canonical_fragments(sides[1])}
    metadata = {
        "left_only_maps": sorted(map_sets[0] - map_sets[1]),
        "right_only_maps": sorted(map_sets[1] - map_sets[0]),
        "shared_maps": len(map_sets[0] & map_sets[1]),
        "core_atoms": len(core), "core_edges": core.number_of_edges(),
        "input_components": {key: len(value) for key, value in inputs.items()},
        "target_has_stereo": {key: targets[key] != strict_targets[key] for key in DIRECTIONS},
    }
    return rule, inputs, targets, strict_targets, metadata


def run_direction(rule, fragments, target, strict_target, direction, variant):
    started = perf_counter()
    selected = (largest_fragment(fragments),) if variant == "partial_largest" else fragments
    result = {"withheld_components": len(fragments) - len(selected),
              "input": ".".join(selected), "exact": False, "exact_stereo": False,
              "target_contained": False, "exact_with_introduced_atoms": False,
              "exact_with_ports": False}
    try:
        with deadline(SETTINGS["seconds"]):
            engine = WildcardEngine(
                result["input"], rule, invert=direction == "bw",
                partial=variant != "full_control", backend=SETTINGS["backend"],
                library_path=SETTINGS["library"], embed_threshold=SETTINGS["candidate_limit"],
                explicit_h=False, automorphism=False, dedup_its=False,
                stereo_mode="ignore",
            )
            reactions = engine.smarts_list
            outcomes, symbols, introduced = set(), False, False
            witness = None
            for reaction, graph in zip(reactions, engine.its_list):
                endpoint = reaction.split(">>")[0 if direction == "bw" else 1]
                actual = canonical_fragments(endpoint)
                outcomes.add(actual)
                completion = graph.graph.get("partial_completion", {})
                has_new = bool(completion.get("introduced_atoms"))
                has_ports = bool(completion.get("attachment_ports"))
                introduced |= has_new
                symbols |= any("*" in item for item in actual)
                result["target_contained"] |= contains_fragments(actual, target)
                if actual == target:
                    result["exact"] = True
                    result["exact_stereo"] |= canonical_fragments(endpoint, stereo=True) == strict_target
                    result["exact_with_introduced_atoms"] |= has_new
                    result["exact_with_ports"] |= has_ports
                    witness = witness or reaction
            result.update(status="complete", mappings=len(engine.mappings),
                          raw_applications=len(reactions), unique_outcomes=len(outcomes),
                          has_symbolic_outcome=symbols, has_introduced_atoms=introduced,
                          outcome_sha256=hashlib.sha256(json.dumps(sorted(outcomes)).encode()).hexdigest(),
                          outcome_sample=[".".join(value) for value in sorted(outcomes)[:3]])
            if witness:
                result["exact_witness"] = witness
            if SETTINGS["backend"] == "cpp":
                result["native"] = engine.native_statistics
    except (ReplayTimeout, Exception) as error:
        if isinstance(error, ReplayTimeout):
            status = "timeout"
        elif isinstance(error, WildcardSearchLimitError):
            status = "search_limit"
        elif isinstance(error, NativeReactorUnsupportedError):
            status = "unsupported_native"
        elif isinstance(error, WildcardCompletionError):
            status = "completion_error"
        else:
            status = "error"
        result.update(status=status, error_type=type(error).__name__, error=str(error))
    result["seconds"] = perf_counter() - started
    return result


def process(item):
    position, row = item
    started = perf_counter()
    record = {"row": position, "r_id": row.get("r_id", str(position)),
              "split": row.get("split", ""), "source": row.get("source", "")}
    try:
        with deadline(SETTINGS["seconds"]):
            rule, inputs, targets, strict_targets, metadata = prepare(row["aam"], SETTINGS["hydrogen_policy"])
        record.update(metadata)
        record["targets"] = {key: ".".join(value) for key, value in targets.items()}
        if metadata["left_only_maps"] or metadata["right_only_maps"]:
            record.update(status="unsupported_atom_carrier",
                          error="The raw extracted rule has atoms present at only one endpoint")
        elif not len(rule.rc.raw):
            record.update(status="empty_rule", error="No reaction-center edit was extracted")
        else:
            record["status"] = "attempted"
            record["runs"] = {
                variant: {direction: run_direction(rule, inputs[direction], targets[direction],
                                                   strict_targets[direction], direction, variant)
                          for direction in DIRECTIONS}
                for variant in VARIANTS
            }
    except (ReplayTimeout, Exception) as error:
        record.update(status="preparation_timeout" if isinstance(error, ReplayTimeout) else "preparation_error",
                      error_type=type(error).__name__, error=str(error))
    record["seconds"] = perf_counter() - started
    return record


def summarize(records):
    totals = Counter()
    splits = defaultdict(Counter)
    metrics = defaultdict(Counter)
    examples = defaultdict(list)
    for record in records:
        totals[record["status"]] += 1
        splits[record.get("split", "")][record["status"]] += 1
        key = record["status"]
        if len(examples[key]) < 5:
            examples[key].append(record["r_id"])
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                tally = metrics[variant + "/" + direction]
                tally["source_rows"] += 1
                if record["status"] != "attempted":
                    tally[record["status"]] += 1
                    continue
                result = record["runs"][variant][direction]
                tally["attempted"] += 1
                tally[result["status"]] += 1
                for flag in ("exact", "exact_stereo", "target_contained", "exact_with_introduced_atoms",
                             "exact_with_ports", "has_symbolic_outcome", "has_introduced_atoms"):
                    tally[flag] += bool(result.get(flag))
                if result["withheld_components"]:
                    tally["withheld_rows"] += 1
                    tally["withheld_exact"] += result["exact"]
                    tally["withheld_exact_with_introduced_atoms"] += result["exact_with_introduced_atoms"]
                label = variant + "/" + direction + "/" + result["status"] + ("/exact" if result["exact"] else "/miss")
                if len(examples[label]) < 5:
                    examples[label].append(record["r_id"])
    return {"rows": sum(totals.values()), "record_status": dict(totals),
            "split_status": {key: dict(value) for key, value in sorted(splits.items())},
            "metrics": {key: dict(value) for key, value in sorted(metrics.items())},
            "example_ids": dict(examples)}


def write_report(output, report):
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    columns = ["variant", "direction", "source_rows", "attempted", "complete", "exact", "exact_stereo",
               "target_contained", "withheld_rows", "withheld_exact", "withheld_exact_with_introduced_atoms",
               "completion_error", "unsupported_native", "search_limit", "timeout", "error"]
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, metrics in report["metrics"].items():
            variant, direction = key.split("/")
            writer.writerow({"variant": variant, "direction": direction,
                             **{column: metrics.get(column, 0) for column in columns[2:]}})
    lines = ["USPTO reaction-center replay", "===========================", "",
             "This is source-conditioned execution: each rule is extracted from the same",
             "mapped reaction being replayed. It is not a held-out prediction experiment.", "",
             "The primary score compares the complete multiset of unmapped, constitutional",
             "endpoint structures, preserving charges and isotopes. Stereo is removed from",
             "both engine inputs. Exact stereo equality is also counted against the original",
             "reference; it does not validate stereo processing in this native profile.", "",
             "Raw one-sided atom states are refused. No balancing, by-product invention,",
             "largest-product-only scoring, or favourable-row filtering is performed.", "",
             f"Source rows: {report['rows']}; statuses: {report['record_status']}", "",
             ".. csv-table:: Recovery counts (unranked output sets)", "   :file: summary.csv", "   :header-rows: 1", "",
             "full_control requires a full rule match on all recorded input fragments.",
             "partial_all enumerates component subsets on the same input.",
             "partial_largest supplies only the largest input fragment (heavy-atom count,",
             "canonical lexical tie-break). Withheld rows have at least one omitted fragment.",
             "Exact recovery with introduced atoms records an actual completion witness.", "",
             "Search-limit, timeout and serialization/completion failures retain their",
             "denominators. Candidate limits are not proofs of absent matches. The Python",
             "timer is delivered after a native call returns; it is not native preemption.", ""]
    (output / "report.rst").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT.parent / "SynRXN/Data/tables/canonical/synthesis/uspto_50k.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--hydrogen-policy", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds <= 0 or args.candidate_limit < 1:
        parser.error("workers, seconds and candidate-limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve prior evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    library = build_native_reactor(ROOT / "build/partial/native") if args.backend == "cpp" else None
    settings = {"library": str(library) if library else None, "backend": args.backend,
                "candidate_limit": args.candidate_limit, "seconds": args.seconds,
                "hydrogen_policy": args.hydrogen_policy}
    with gzip.open(args.dataset, "rt") as stream:
        reader = csv.DictReader(stream)
        rows = list(enumerate(itertools.islice(reader, args.limit)))
    sources = list((ROOT / "synkit/Synthesis/Reactor").rglob("*.py"))
    sources += list((ROOT / "synkit/Synthesis/Reactor/cpp").rglob("*.cpp"))
    sources += list((ROOT / "synkit/Synthesis/Reactor/cpp").rglob("*.h"))
    sources += [Path(__file__), ROOT / "synkit/Rule/syn_rule.py", ROOT / "synkit/Graph/ITS/rc_extractor.py",
                ROOT / "synkit/IO/chem_converter.py", ROOT / "synkit/IO/mol_to_graph.py",
                ROOT / "Test/Benchmark/Lewis/test_uspto_partial_replay.py"]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    metadata = {"schema": "synkit.uspto-partial-replay/1", "command": sys.argv,
                "source": {"path": str(args.dataset.resolve()), "sha256": digest(args.dataset),
                           "selected_rows": len(rows),
                           "selection": "prefix in stored row order; all if limit is absent"},
                "settings": settings, "workers": args.workers,
                "execution_contract": {
                    "rule": "own reaction-center, paired Lewis fields, unchanged edges between center atoms retained",
                    "input": "unmapped source endpoint; stereo removed; no mapping-anchored binding",
                    "score": "complete constitutional component multiset; no charge/isotope/fragment filtering",
                    "hydrogen": args.hydrogen_policy,
                    "reference_atom_creation": "refused; no invented opposite endpoint states",
                    "ranking": "none; recovery anywhere in the complete returned batch under declared limits",
                },
                "python": sys.version, "rdkit": rdBase.rdkitVersion, "platform": platform.platform(),
                "source_sha256": source_hashes,
                "native_manifest": json.loads(library.with_suffix(".json").read_text()) if library else None}
    (args.output / "running.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started, records = perf_counter(), []
    with gzip.open(args.output / "records.jsonl.gz", "wt") as stream:
        with mp.get_context("spawn").Pool(args.workers, initialize, (settings,)) as pool:
            for record in pool.imap(process, rows, chunksize=16):
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                if len(records) % args.progress_every == 0 or len(records) == len(rows):
                    print(f"{len(records)}/{len(rows)} {dict(Counter(r['status'] for r in records))} "
                          f"elapsed={perf_counter()-started:.1f}s", flush=True)
                    stream.flush()
    report = {**metadata, **summarize(records), "wall_seconds": perf_counter() - started,
              "source_unchanged": all(digest(ROOT / p) == value for p, value in source_hashes.items()),
              "records_sha256": digest(args.output / "records.jsonl.gz")}
    write_report(args.output, report)
    (args.output / "running.json").unlink()
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
