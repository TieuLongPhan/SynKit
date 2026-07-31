#!/usr/bin/env python3
"""Run single-process hydrogen-extension and partial-AAM capability checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
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
PARTIALAAMS_COMMIT = "008173ed7a943ab03f1b8a33bfe5c7aea84dace9"
METHODS = ("gm", "rb1", "rb2")
HEXTEND_METHODS = ("hextend_legacy", "hextend_new")


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


def prepare_partial_cases(
    dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize transferred H atoms and remove only their atom maps.

    This creates a fair capability input for partial-AAM methods. It does not
    imply that those methods support reaction-centre hydrogen completion.
    """
    from synkit.Graph.Hyrogen._misc import check_hcount_change
    from synkit.Graph.Hyrogen.hcomplete import HComplete
    from synkit.IO.chem_converter import graph_to_rsmi

    prepared = []
    for index, entry in enumerate(dataset):
        its = entry["ITS"]
        resolved_format = HComplete._resolve_format(its, "auto")
        reactant, product = HComplete._decompose_its(its, resolved_format)
        hcount_change = check_hcount_change(reactant, product)
        try:
            partial_reactant, partial_product = next(
                HComplete._iter_hydrogen_side_graph_completions(
                    reactant,
                    product,
                    max_candidates=1,
                )
            )
        except StopIteration:
            partial_reactant, partial_product = reactant, product

        partial_reactant = deepcopy(partial_reactant)
        partial_product = deepcopy(partial_product)
        materialized_hydrogens = set()
        for graph in (partial_reactant, partial_product):
            for node, attrs in graph.nodes(data=True):
                if attrs.get("element") == "H":
                    materialized_hydrogens.add(node)
                    attrs["atom_map"] = 0

        partial = graph_to_rsmi(
            partial_reactant,
            partial_product,
            sanitize=True,
            explicit_hydrogen=True,
        )
        if not partial:
            raise ValueError(f"Could not serialize hydrogen input {entry['R-id']}")
        prepared.append(
            {
                "index": index,
                "record_id": str(entry["R-id"]),
                "hcount_change": int(hcount_change),
                "materialized_hydrogens": len(materialized_hydrogens),
                "partial": partial,
            }
        )
    return prepared


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
    rows = []
    for repetition in range(1, repetitions + 1):
        for method in HEXTEND_METHODS:
            implementation = implementations[method]
            for index, entry in enumerate(dataset):
                elapsed, result, error = timed_call(
                    lambda entry=entry, implementation=implementation: (
                        implementation._extend_unique(entry["ITS"])
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
                    rc_list, its_list, signatures = result
                    row.update(
                        status="OUTPUT",
                        unique_classes=len(rc_list),
                        completed_its=len(its_list),
                        signatures=list(signatures),
                    )
                else:
                    row.update(
                        status="ERROR",
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                rows.append(row)
    return rows


def isolate_historical_imports() -> None:
    repo_root = ROOT.resolve()
    retained = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except (OSError, RuntimeError):
            retained.append(entry)
            continue
        if resolved != repo_root:
            retained.append(entry)
    sys.path[:] = retained


def external_worker(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Historical hydrogen worker")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--case-timeout", type=float, required=True)
    args = parser.parse_args(arguments)

    isolate_historical_imports()
    from partialaams.aam_expand import partial_aam_extension_from_smiles
    import partialaams
    import synkit

    synkit_version = importlib.metadata.version("synkit")
    synkit_source = Path(synkit.__file__).resolve()
    environment_prefix = Path(sys.prefix).resolve()
    if synkit_version != "0.0.6" or not synkit_source.is_relative_to(
        environment_prefix
    ):
        raise RuntimeError(
            "Historical worker requires SynKit 0.0.6 from its conda environment; "
            f"found {synkit_version} at {synkit_source}"
        )

    prepared = read_jsonl(args.prepared)
    rows = []
    dispatch = {"gm": "gm", "rb1": "extend", "rb2": "extend_g"}
    for repetition in range(1, args.repetitions + 1):
        for method in METHODS:
            for case in prepared:
                elapsed, candidate, error = timed_call(
                    lambda case=case, method=method: (
                        partial_aam_extension_from_smiles(
                            case["partial"],
                            method=dispatch[method],
                        )
                    ),
                    args.case_timeout,
                )
                row = {
                    "method": method,
                    "repetition": repetition,
                    "index": case["index"],
                    "record_id": case["record_id"],
                    "seconds": elapsed,
                }
                if error is None:
                    row.update(status="OUTPUT", candidate=candidate)
                else:
                    row.update(
                        status="ERROR",
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                rows.append(row)
    write_jsonl(args.output, rows)
    summary = {
        "python": platform.python_version(),
        "python_prefix": str(environment_prefix),
        "synkit": synkit_version,
        "synkit_source": str(synkit_source),
        "partialaams_source": partialaams.__file__,
        "gmapache": importlib.metadata.version("gmapache"),
        "rdkit": importlib.metadata.version("rdkit"),
        "networkx": importlib.metadata.version("networkx"),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


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


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partialaams", type=Path, required=True)
    parser.add_argument("--external-env", default="aam")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    dataset_path = args.dataset.resolve()
    partialaams = args.partialaams.resolve()
    if git_commit(partialaams) != PARTIALAAMS_COMMIT:
        raise RuntimeError(f"PartialAAMs must be pinned at {PARTIALAAMS_COMMIT}")
    output_dir = args.output_dir.resolve()
    targets = (
        output_dir / "prepared-hydrogen-inputs.jsonl.gz",
        output_dir / "hextend-runs.jsonl.gz",
        output_dir / "partialaam-control-runs.jsonl.gz",
        output_dir / "external-environment.json",
        output_dir / "aggregate.json",
    )
    if not args.force and any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite output in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_pickle(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    prepared = prepare_partial_cases(dataset)
    write_jsonl(targets[0], prepared)

    hextend_rows = run_hextend(dataset, args.repetitions, args.case_timeout)
    write_jsonl(targets[1], hextend_rows)

    child_env = os.environ.copy()
    child_env.update(
        {
            "PYTHONPATH": str(partialaams),
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
        args.external_env,
        "python",
        str(Path(__file__).resolve()),
        "--external-worker",
        "--prepared",
        str(targets[0]),
        "--output",
        str(targets[2]),
        "--summary",
        str(targets[3]),
        "--repetitions",
        str(args.repetitions),
        "--case-timeout",
        str(args.case_timeout),
    ]
    subprocess.run(command, cwd=partialaams, env=child_env, check=True)
    external_rows = read_jsonl(targets[2])

    aggregate = {
        "schema": "synkit.hydrogen-expansion-comparison/1",
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
            "rows": len(dataset),
        },
        "execution": {
            "processes_per_method": 1,
            "thread_limits": 1,
            "repetitions": args.repetitions,
        },
        "method_contracts": {
            "hextend_legacy": "historical hydrogen-extension class enumerator",
            "hextend_new": "provenance-aware hydrogen-extension class enumerator",
            "gm_rb1_rb2": (
                "partial-AAM capability controls; reaction-centre hydrogen "
                "extension is outside their demonstrated contract"
            ),
            "analysis_method_a": (
                "classify every hydrogen permutation by full ITS isomorphism"
            ),
            "analysis_method_b": (
                "classify by base-ITS automorphisms plus anchored co-extension"
            ),
        },
        "summaries": summarize(hextend_rows + external_rows),
        "external_environment": json.loads(targets[3].read_text()),
        "partialaams": {
            "path": str(partialaams),
            "commit": git_commit(partialaams),
        },
        "artifacts": [path.name for path in targets[:-1]],
    }
    targets[4].write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(f"Wrote hydrogen comparison: {targets[4]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--external-worker":
        raise SystemExit(external_worker(sys.argv[2:]))
    raise SystemExit(main())
