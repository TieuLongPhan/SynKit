#!/usr/bin/env python3
"""Run single-process hydrogen-extension and exact reference comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = ROOT / "Experiment" / "Lewis" / "Data" / "hydrogen.pkl.gz"
HEXTEND_METHODS = ("hextend_legacy", "hextend_new")
REFERENCE_BACKENDS = ("an_gm", "rb_gm", "rb_nx")


class CaseTimeout(TimeoutError):
    """Raised when one method call exceeds the configured wall-time limit."""


def raise_timeout(_signum, _frame) -> None:
    raise CaseTimeout("Hydrogen extension exceeded the per-case timeout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: Path) -> list[dict[str, Any]]:
    """Load the trusted corpus, whose historical .gz file is plain pickle."""
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list in {path}, found {type(payload).__name__}")
    return payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def timed_call(function, timeout: float) -> tuple[float, Any, Exception | None]:
    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    started = time.perf_counter()
    try:
        result = function()
        return time.perf_counter() - started, result, None
    except Exception as exc:
        return time.perf_counter() - started, None, exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_hextend(
    dataset: list[dict[str, Any]],
    repetitions: int,
    timeout: float,
) -> list[dict[str, Any]]:
    from synkit.Graph.Hyrogen.hextend import HExtend
    from synkit.Graph.Hyrogen.hextend_legacy import LegacyHExtend

    implementations = {
        "hextend_legacy": LegacyHExtend,
        "hextend_new": HExtend,
    }

    def extend_and_classify(implementation, its):
        _, completed_its, signatures = implementation.extend_its(its)
        clusters, _ = HExtend.cluster_full_its(completed_its, signatures)
        return completed_its, clusters

    rows = []
    for repetition in range(1, repetitions + 1):
        for method in HEXTEND_METHODS:
            implementation = implementations[method]
            for index, entry in enumerate(dataset):
                elapsed, result, error = timed_call(
                    lambda entry=entry, implementation=implementation: (
                        extend_and_classify(implementation, entry["ITS"])
                    ),
                    timeout,
                )
                row = {
                    "method": method,
                    "repetition": repetition,
                    "index": index,
                    "record_id": str(entry["R-id"]),
                    "seconds": elapsed,
                }
                if error is None:
                    completed_its, clusters = result
                    row.update(
                        status="OUTPUT",
                        unique_classes=len(clusters),
                        completed_its=len(completed_its),
                    )
                else:
                    row.update(
                        status="ERROR",
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                rows.append(row)
    return rows


def select_reference_cases(
    dataset: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the two filters used by the published hydrogen analysis."""
    from rdkit import Chem
    from synkit.Graph.Hyrogen._misc import check_hcount_change
    from synkit.Graph.Hyrogen.hcomplete import HComplete

    accepted, excluded = [], []
    for index, entry in enumerate(dataset):
        side_maps = []
        fully_mapped = True
        for side in str(entry["aam"]).split(">>"):
            maps = []
            atom_count = 0
            for fragment in side.split("."):
                molecule = Chem.MolFromSmiles(fragment)
                if molecule is None:
                    fully_mapped = False
                    break
                atom_count += molecule.GetNumAtoms()
                maps.extend(
                    atom.GetAtomMapNum()
                    for atom in molecule.GetAtoms()
                    if atom.GetAtomMapNum()
                )
            side_maps.append((atom_count, sorted(maps)))
        if (
            not fully_mapped
            or len(side_maps) != 2
            or side_maps[0][0] != len(side_maps[0][1])
            or side_maps[1][0] != len(side_maps[1][1])
            or side_maps[0][1] != side_maps[1][1]
        ):
            excluded.append(
                {
                    "index": index,
                    "record_id": str(entry["R-id"]),
                    "reason": "uneven_aam",
                }
            )
            continue

        resolved_format = HComplete._resolve_format(entry["ITS"], "auto")
        reactant, product = HComplete._decompose_its(entry["ITS"], resolved_format)
        if check_hcount_change(reactant, product) == 0:
            excluded.append(
                {
                    "index": index,
                    "record_id": str(entry["R-id"]),
                    "reason": "no_unmatched_hydrogens",
                }
            )
            continue
        accepted.append(entry)
    return accepted, excluded


def run_reference(
    dataset: list[dict[str, Any]],
    repetitions: int,
    timeout: float,
    backend: str = "rb_nx",
) -> list[dict[str, Any]]:
    """Run one published Method A/B backend."""
    from Experiment.Lewis.hydrogen_expand.reference_methods import (
        run_reference_methods,
    )

    rows = []
    for repetition in range(1, repetitions + 1):
        for index, entry in enumerate(dataset):
            elapsed, result, error = timed_call(
                lambda entry=entry: run_reference_methods(entry["ITS"], backend),
                timeout,
            )
            base = {
                "repetition": repetition,
                "index": index,
                "record_id": str(entry["R-id"]),
            }
            if error is not None:
                for method in ("a", "b"):
                    rows.append(
                        base
                        | {
                            "method": f"method_{method}_{backend}",
                            "backend": backend,
                            "status": "ERROR",
                            "seconds": elapsed,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                continue
            common = {
                "status": "OUTPUT",
                "automorphisms": result["automorphisms"],
                "automorphism_seconds": result["automorphism_seconds"],
                "permutations": result["permutations"],
                "unmatched_hydrogens": result["unmatched_hydrogens"],
                "backend": backend,
            }
            rows.append(
                base
                | common
                | {
                    "method": f"method_a_{backend}",
                    "seconds": result["method_a_seconds"],
                    "unique_classes": result["method_a_classes"],
                }
            )
            rows.append(
                base
                | common
                | {
                    "method": f"method_b_{backend}",
                    "seconds": result["method_b_seconds"],
                    "unique_classes": result["method_b_classes"],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries = []
    for method in sorted(grouped):
        method_rows = grouped[method]
        durations = [float(row["seconds"]) for row in method_rows]
        outputs = [row for row in method_rows if row["status"] == "OUTPUT"]
        summary = {
            "method": method,
            "attempts": len(method_rows),
            "outputs": len(outputs),
            "errors": len(method_rows) - len(outputs),
            "success_rate": len(outputs) / len(method_rows) if method_rows else 0.0,
            "mean_seconds_per_attempt": statistics.mean(durations),
            "median_seconds_per_attempt": statistics.median(durations),
            "maximum_seconds": max(durations),
        }
        error_types = Counter(
            str(row.get("error_type", "UnknownError"))
            for row in method_rows
            if row["status"] != "OUTPUT"
        )
        if error_types:
            summary["error_types"] = dict(sorted(error_types.items()))
        if outputs and "unique_classes" in outputs[0]:
            classes = [int(row["unique_classes"]) for row in outputs]
            summary.update(
                mean_unique_classes=statistics.mean(classes),
                maximum_unique_classes=max(classes),
            )
        summaries.append(summary)
    return summaries


def summarize_by_hcount(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Table-2-style mean and population std from reaction means."""
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["status"] != "OUTPUT" or "unmatched_hydrogens" not in row:
            continue
        key = (
            str(row["method"]),
            int(row["unmatched_hydrogens"]),
            str(row["record_id"]),
        )
        grouped[key].append(float(row["seconds"]) * 1000)

    reaction_means: dict[tuple[str, int], list[float]] = defaultdict(list)
    repetitions: dict[tuple[str, int], set[int]] = defaultdict(set)
    for (method, hcount, _), durations in grouped.items():
        reaction_means[(method, hcount)].append(statistics.mean(durations))
        repetitions[(method, hcount)].add(len(durations))

    table = []
    for (method, hcount), durations in sorted(reaction_means.items()):
        table.append(
            {
                "method": method,
                "unmatched_hydrogens": hcount,
                "reactions": len(durations),
                "repetitions_per_reaction": sorted(repetitions[(method, hcount)]),
                "mean_ms": statistics.mean(durations),
                "population_std_ms": statistics.pstdev(durations),
            }
        )
    return table


def reuse_reference_artifacts(
    source_dir: Path,
    dataset_path: Path,
    eligible_rows: int,
    repetitions: int,
    rb_target: Path,
    gmapache_target: Path,
    environment_target: Path,
) -> list[dict[str, Any]]:
    """Validate and copy reference artifacts from a completed run."""
    source_aggregate = json.loads((source_dir / "aggregate.json").read_text())
    if source_aggregate["dataset"]["sha256"] != sha256(dataset_path):
        raise ValueError("Reused references use a different dataset")
    if source_aggregate["dataset"]["eligible_rows"] != eligible_rows:
        raise ValueError("Reused references use a different eligible subset")
    if source_aggregate["execution"]["repetitions"] != repetitions:
        raise ValueError("Reused references use a different repetition count")

    rb_rows = read_jsonl(source_dir / "rb-nx-reference-runs.jsonl.gz")
    gmapache_rows = read_jsonl(source_dir / "gmapache-reference-runs.jsonl.gz")
    write_jsonl(rb_target, rb_rows)
    write_jsonl(gmapache_target, gmapache_rows)
    environment_target.write_text(
        (source_dir / "gmapache-environment.json").read_text()
    )
    return rb_rows + gmapache_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-timeout", type=float, default=60.0)
    parser.add_argument("--gmapache-env", default="aam")
    parser.add_argument(
        "--reuse-reference-dir",
        type=Path,
        help="Reuse completed RB/GM artifacts and rerun only SynKit HExtend",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    targets = (
        output_dir / "hextend-runs.jsonl.gz",
        output_dir / "rb-nx-reference-runs.jsonl.gz",
        output_dir / "gmapache-reference-runs.jsonl.gz",
        output_dir / "gmapache-environment.json",
        output_dir / "excluded-records.json",
        output_dir / "aggregate.json",
    )
    if not args.force and any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite output in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_pickle(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    dataset, excluded = select_reference_cases(dataset)
    targets[4].write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n")
    hextend_rows = run_hextend(dataset, args.repetitions, args.case_timeout)
    write_jsonl(targets[0], hextend_rows)
    reference_source = None
    if args.reuse_reference_dir is not None:
        reference_source = args.reuse_reference_dir.resolve()
        reference_rows = reuse_reference_artifacts(
            reference_source,
            dataset_path,
            len(dataset),
            args.repetitions,
            targets[1],
            targets[2],
            targets[3],
        )
    else:
        reference_rows = run_reference(
            dataset,
            args.repetitions,
            args.case_timeout,
            backend="rb_nx",
        )
        write_jsonl(targets[1], reference_rows)

        child_env = os.environ.copy()
        child_env.update(
            {
                "PYTHONPATH": str(ROOT),
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.gmapache_env,
            "python",
            str(HERE / "gmapache_worker.py"),
            "--dataset",
            str(dataset_path),
            "--output",
            str(targets[2]),
            "--environment-output",
            str(targets[3]),
            "--repetitions",
            str(args.repetitions),
            "--case-timeout",
            str(args.case_timeout),
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        subprocess.run(command, cwd=ROOT, env=child_env, check=True)
        reference_rows.extend(read_jsonl(targets[2]))

    hcounts = {
        row["record_id"]: row["unmatched_hydrogens"]
        for row in reference_rows
        if row["status"] == "OUTPUT"
    }
    for row in hextend_rows:
        if row["record_id"] in hcounts:
            row["unmatched_hydrogens"] = hcounts[row["record_id"]]
    write_jsonl(targets[0], hextend_rows)

    reference_classes = {
        (row["repetition"], row["record_id"]): row["unique_classes"]
        for row in reference_rows
        if row["method"] == "method_a_rb_nx" and row["status"] == "OUTPUT"
    }
    agreement = {}
    for method in HEXTEND_METHODS:
        comparable = [
            row
            for row in hextend_rows
            if row["method"] == method
            and row["status"] == "OUTPUT"
            and (row["repetition"], row["record_id"]) in reference_classes
        ]
        matches = [
            row
            for row in comparable
            if row["unique_classes"]
            == reference_classes[(row["repetition"], row["record_id"])]
        ]
        agreement[method] = {
            "matches": len(matches),
            "comparisons": len(comparable),
            "rate": len(matches) / len(comparable) if comparable else 0.0,
            "mismatched_record_ids": sorted(
                {row["record_id"] for row in comparable if row not in matches}
            ),
        }

    expected_reference_methods = {
        f"method_{method}_{backend}"
        for method in ("a", "b")
        for backend in REFERENCE_BACKENDS
    }
    reference_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        reference_groups[(row["repetition"], row["record_id"])].append(row)
    reference_matches = {}
    for key, rows in reference_groups.items():
        outputs = [row for row in rows if row["status"] == "OUTPUT"]
        reference_matches[key] = (
            {row["method"] for row in outputs} == expected_reference_methods
            and len({int(row["unique_classes"]) for row in outputs}) == 1
        )
    reference_agreement = {
        "comparisons": len(reference_matches),
        "matches": sum(reference_matches.values()),
        "mismatched_record_ids": sorted(
            {
                record_id
                for (_, record_id), matches in reference_matches.items()
                if not matches
            }
        ),
    }
    import synkit

    aggregate = {
        "schema": "synkit.hydrogen-expansion-comparison/6",
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
            "source_rows": len(dataset) + len(excluded),
            "eligible_rows": len(dataset),
            "excluded_rows": len(excluded),
        },
        "execution": {
            "processes_per_method": 1,
            "thread_limits": 1,
            "repetitions": args.repetitions,
            "gmapache_environment": args.gmapache_env,
            "reference_source": (
                str(reference_source) if reference_source is not None else None
            ),
        },
        "method_contracts": {
            "hextend_legacy": (
                "historical same-permutation candidate enumeration followed "
                "by full-ITS classification"
            ),
            "hextend_new": (
                "provenance-aware candidate enumeration followed by full-ITS "
                "classification"
            ),
            "method_a": (
                "classify every hydrogen permutation by full ITS isomorphism"
            ),
            "method_b": (
                "classify by base-ITS automorphisms plus anchored co-extension"
            ),
        },
        "reference_backends": {
            "an_gm": "native GranMapache stable extension",
            "rb_gm": "anchor-relabeling GranMapache",
            "rb_nx": "anchor-relabeling NetworkX",
        },
        "hextend_classification": (
            "RC-invariant and hydrogen-distance prefilters followed by "
            "exhaustive changed-core-anchored exact full-ITS stereo-aware "
            "isomorphism"
        ),
        "local_environment": {
            "python": platform.python_version(),
            "networkx": importlib.metadata.version("networkx"),
            "rdkit": importlib.metadata.version("rdkit"),
            "synkit": getattr(synkit, "__version__", "unknown"),
            "synkit_source": str(Path(synkit.__file__).resolve()),
        },
        "gmapache_environment": json.loads(targets[3].read_text()),
        "summaries": summarize(hextend_rows + reference_rows),
        "table_by_hcount": summarize_by_hcount(hextend_rows + reference_rows),
        "reference_class_count_agreement": reference_agreement,
        "class_count_agreement": agreement,
        "exclusions": excluded,
        "artifacts": [path.name for path in targets[:-1]],
    }
    targets[5].write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(f"Wrote hydrogen comparison: {targets[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
