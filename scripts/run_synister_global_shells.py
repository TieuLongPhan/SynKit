"""Run reference-blinded global Synister shell experiments on FlowER rows.

The dataset must be supplied explicitly.  For ``reference_cd`` mode, only the
held-out reference's scalar chemical distance defines the global shell; the
mapping is revealed after enumeration.  For ``minimal`` mode, both mapping and
distance remain hidden until the global optimizer shell has been explored.

The runner is resumable and defaults to one worker, 300 seconds per shell, and
a 6 GiB per-worker address-space limit.  Per-case gzip JSON records and the
summary distinguish complete, incomplete, and error outcomes.  In parallel
mode, workers compute only; the parent remains the sole filesystem writer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from synkit.Chem.Mapper import (  # noqa: E402
    GlobalShellConfig,
    analyze_reference_blinded_global_shell,
    blinded_mapped_reaction_problem,
)

SCHEMA_VERSION = 4
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "synister_global_shells_v4"
_WORKER_OPTIONS = None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "synkit/Chem/Mapper/analysis.py",
        ROOT / "synkit/Chem/Mapper/spectrum.py",
        ROOT / "synkit/Chem/Mapper/chem/blind.py",
        ROOT / "synkit/Chem/Mapper/exact/distance.py",
        ROOT / "synkit/Chem/Mapper/exact/distance_bounds.py",
        ROOT / "synkit/Chem/Mapper/exact/distance_records.py",
        ROOT / "synkit/Chem/Mapper/exact/distance_verify.py",
        ROOT / "synkit/Chem/Mapper/exact/symmetry.py",
        ROOT / "synkit/Chem/Mapper/exact/edit_support.py",
        ROOT / "synkit/Chem/Mapper/exact/hybrid.py",
        ROOT / "synkit/Chem/Mapper/graph/automorphism.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_gzip_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=6,
                mtime=0,
            ) as compressed:
                compressed.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="ascii") as stream:
        return json.load(stream)


def _record_payload_is_valid(record, manifest_sha256) -> bool:
    """Verify one immutable case payload and its campaign binding."""
    payload = dict(record)
    claimed = payload.pop("record_sha256", None)
    return bool(
        claimed == _payload_sha256(payload)
        and payload.get("campaign_manifest_sha256") == manifest_sha256
    )


def _load_rows(dataset: Path) -> list[dict[str, str]]:
    opener = gzip.open if dataset.suffix == ".gz" else open
    with opener(dataset, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_line", "reaction_id", "mapped_reaction"}
    if not rows or set(rows[0]) != required:
        raise ValueError("dataset must contain source_line,reaction_id,mapped_reaction")
    return rows


def _reaction_from_row(row) -> str:
    mapped = str(row["mapped_reaction"])
    reaction_id = str(row["reaction_id"])
    if "|" not in mapped:
        return mapped
    reaction, embedded_id = mapped.rsplit("|", 1)
    if embedded_id != reaction_id:
        raise ValueError("reaction ID does not match embedded provenance")
    return reaction


def _case_path(output: Path, source_line: int) -> Path:
    return output / "cases" / f"line_{source_line}.json.gz"


def _memory_limit_bytes(memory_limit_gib: float) -> int:
    limit = int(memory_limit_gib * 1024**3)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, int(hard))
    return limit


def _configure_memory_limit(memory_limit_gib: float) -> int:
    limit = _memory_limit_bytes(memory_limit_gib)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return limit


def _initialize_worker(options, memory_limit_gib: float) -> None:
    """Install immutable run options and the address-space cap in a worker."""
    global _WORKER_OPTIONS
    _configure_memory_limit(memory_limit_gib)
    _WORKER_OPTIONS = options


def _worker_run_case(row):
    """Compute one case without writing shared campaign state."""
    if _WORKER_OPTIONS is None:
        raise RuntimeError("parallel worker was not initialized")
    return _safe_run_case(row, _WORKER_OPTIONS)


def _run_case(row, options):
    reaction = _reaction_from_row(row)
    problem = blinded_mapped_reaction_problem(
        reaction,
        heavy_only=options["heavy_only"],
        blind_seed=options["blind_seed"],
    )
    config = GlobalShellConfig(
        binary=options["binary"],
        time_limit_seconds=options["time_limit_per_shell"],
        max_bijections=options["max_bijections"],
        max_mappings=options["max_mappings"],
        tolerance=options["tolerance"],
        symmetry_pruning=options["symmetry_pruning"],
        max_symmetry_automorphisms=options["max_symmetry_automorphisms"],
        symmetry_timeout_seconds=options["symmetry_timeout_seconds"],
        symmetry_max_search_nodes=options["symmetry_max_search_nodes"],
        backend=options["backend"],
        max_edit_support_pairs=options["max_edit_support_pairs"],
        use_slap_seed=options["use_slap_seed"],
        structure_analysis=options["structure_analysis"],
        template_radius=options["template_radius"],
        structure_timeout_seconds=options["structure_timeout_seconds"],
        structure_max_search_nodes=options["structure_max_search_nodes"],
    )
    modes = (
        ("reference_cd", "minimal") if options["mode"] == "both" else (options["mode"],)
    )
    shells = {
        mode: analyze_reference_blinded_global_shell(
            problem.lgp,
            problem.reference_mapping,
            target_mode=mode,
            config=config,
        ).as_dict()
        for mode in modes
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synister_reference_blinded_global_shell_case",
        "campaign_manifest_sha256": options["campaign_manifest_sha256"],
        "source_line": int(row["source_line"]),
        "reaction_id": str(row["reaction_id"]),
        "reaction_sha256": problem.reaction_sha256,
        "atom_count": len(problem.lgp[0].labels),
        "heavy_only": problem.heavy_only,
        "blind_seed": problem.blind_seed,
        "blind_reactant_order_sha256": _payload_sha256(problem.reactant_atom_maps),
        "blind_product_order_sha256": _payload_sha256(problem.product_atom_maps),
        "shells": shells,
    }
    payload["record_sha256"] = _payload_sha256(payload)
    return payload


def _safe_run_case(row, options):
    try:
        return _run_case(row, options)
    except Exception as error:
        reaction = str(row.get("mapped_reaction", "")).rsplit("|", 1)[0]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "synister_reference_blinded_global_shell_case",
            "campaign_manifest_sha256": options["campaign_manifest_sha256"],
            "source_line": int(row["source_line"]),
            "reaction_id": str(row["reaction_id"]),
            "reaction_sha256": hashlib.sha256(reaction.encode("utf-8")).hexdigest(),
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        payload["record_sha256"] = _payload_sha256(payload)
        return payload


def _case_is_current(path, row, manifest_sha256) -> bool:
    if not path.is_file():
        return False
    try:
        record = _read_gzip_json(path)
    except Exception:
        return False
    reaction = _reaction_from_row(row)
    return bool(
        _record_payload_is_valid(record, manifest_sha256)
        and record.get("schema_version") == SCHEMA_VERSION
        and record.get("reaction_id") == row["reaction_id"]
        and record.get("reaction_sha256")
        == hashlib.sha256(reaction.encode("utf-8")).hexdigest()
    )


def _parallel_case_records(
    rows,
    options,
    *,
    workers: int,
    memory_limit_gib: float,
):
    """Yield completed records with a bounded spawn-based worker queue."""
    row_iterator = iter(rows)
    in_flight = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(options, memory_limit_gib),
    ) as executor:
        for row in row_iterator:
            future = executor.submit(_worker_run_case, row)
            in_flight[future] = row
            if len(in_flight) == 2 * workers:
                break

        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                row = in_flight.pop(future)
                try:
                    yield future.result()
                except BrokenProcessPool as error:
                    for pending in in_flight:
                        pending.cancel()
                    raise RuntimeError(
                        "worker pool terminated while processing source line "
                        f"{row['source_line']}; preserve existing case records "
                        "and inspect the service memory/exit diagnostics"
                    ) from error

                try:
                    next_row = next(row_iterator)
                except StopIteration:
                    continue
                replacement = executor.submit(_worker_run_case, next_row)
                in_flight[replacement] = next_row


def _write_summary(output: Path, manifest) -> None:
    modes = (
        ("reference_cd", "minimal")
        if manifest["options"]["mode"] == "both"
        else (manifest["options"]["mode"],)
    )
    status_counts = {mode: Counter() for mode in modes}
    complete = Counter()
    reference_classes = Counter()
    reference_minima = Counter()
    ambiguous = Counter()
    unstable_centres = Counter()
    structure_complete = Counter()
    multiple_its_classes = Counter()
    multiple_template_classes = Counter()
    records = errors = 0
    incomplete_lines = set()
    for path in sorted((output / "cases").glob("line_*.json.gz")):
        try:
            record = _read_gzip_json(path)
        except Exception:
            errors += 1
            continue
        if not _record_payload_is_valid(record, manifest["manifest_sha256"]):
            errors += 1
            continue
        records += 1
        if record.get("status") == "error":
            errors += 1
            incomplete_lines.add(int(record["source_line"]))
            continue
        for mode, shell in record["shells"].items():
            status_counts[mode][str(shell["status"])] += 1
            if not shell["complete"]:
                incomplete_lines.add(int(record["source_line"]))
                continue
            complete[mode] += 1
            reference_classes[mode] += bool(shell["reference_class_observed"])
            reference_minima[mode] += bool(shell["reference_is_global_minimum_proven"])
            labeled_count = shell["labeled_solution_count"]
            ambiguous[mode] += labeled_count is not None and int(labeled_count) > 1
            centre = shell["reaction_center"]
            unstable_centres[mode] += (
                centre["bond_union"] != centre["bond_intersection"]
                or centre["atom_union"] != centre["atom_intersection"]
            )
            structure = shell["structure"]
            if structure["complete"]:
                structure_complete[mode] += 1
                multiple_its_classes[mode] += structure["observed_its_class_count"] > 1
                multiple_template_classes[mode] += (
                    structure["observed_template_class_count"] > 1
                )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synister_reference_blinded_global_shell_summary",
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "case_records": records,
        "remaining_cases": int(manifest["rows"]) - records,
        "error_records": errors,
        "status_counts": {mode: dict(counts) for mode, counts in status_counts.items()},
        "complete_cases": dict(complete),
        "reference_class_observed_complete_cases": dict(reference_classes),
        "reference_global_minimum_proven_cases": dict(reference_minima),
        "ambiguous_complete_cases": dict(ambiguous),
        "unstable_reaction_centre_complete_cases": dict(unstable_centres),
        "structure_complete_cases": dict(structure_complete),
        "multiple_its_classes_structure_complete_cases": dict(multiple_its_classes),
        "multiple_template_classes_structure_complete_cases": dict(
            multiple_template_classes
        ),
        "incomplete_source_lines": sorted(incomplete_lines),
    }
    summary["summary_sha256"] = _payload_sha256(summary)
    temporary = output / ".summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output / "summary.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("reference_cd", "minimal", "both"),
        default="both",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--time-limit-per-shell", type=float, default=300.0)
    parser.add_argument("--memory-limit-gib", type=float, default=6.0)
    parser.add_argument("--max-bijections", type=int, default=0)
    parser.add_argument("--max-mappings", type=int, default=100_000)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--include-explicit-hydrogens", action="store_true")
    parser.add_argument("--blind-seed", default="synister-global-v1")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("auto", "assignment", "edit_support"),
        default="auto",
    )
    parser.add_argument("--max-edit-support-pairs", type=int, default=50_000)
    parser.add_argument("--no-slap-seed", action="store_true")
    parser.add_argument("--max-symmetry-automorphisms", type=int, default=256)
    parser.add_argument("--symmetry-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--symmetry-max-search-nodes", type=int, default=10_000)
    parser.add_argument("--no-structure-analysis", action="store_true")
    parser.add_argument("--template-radius", type=int, default=1)
    parser.add_argument("--structure-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--structure-max-search-nodes", type=int, default=100_000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser


def main(argv=None) -> int:  # noqa: C901
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.time_limit_per_shell <= 0:
        parser.error("--time-limit-per-shell must be positive")
    if args.memory_limit_gib <= 0:
        parser.error("--memory-limit-gib must be positive")
    if args.max_bijections < 0 or args.max_mappings < 1:
        parser.error("bijection/mapping limits must be non-negative/positive")
    if args.template_radius < 0:
        parser.error("--template-radius must be non-negative")
    if args.structure_timeout_seconds < 0:
        parser.error("--structure-timeout-seconds must be non-negative")
    if args.structure_max_search_nodes < 1:
        parser.error("--structure-max-search-nodes must be positive")
    if args.max_edit_support_pairs < 1:
        parser.error("--max-edit-support-pairs must be positive")
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")

    memory_limit_bytes = _memory_limit_bytes(args.memory_limit_gib)
    if args.workers == 1:
        _configure_memory_limit(args.memory_limit_gib)
    rows = _load_rows(args.dataset)
    options = {
        "mode": args.mode,
        "time_limit_per_shell": args.time_limit_per_shell,
        "memory_limit_bytes": memory_limit_bytes,
        "max_bijections": args.max_bijections or None,
        "max_mappings": args.max_mappings,
        "binary": args.binary,
        "heavy_only": not args.include_explicit_hydrogens,
        "blind_seed": args.blind_seed,
        "symmetry_pruning": not args.no_symmetry,
        "backend": args.backend,
        "max_edit_support_pairs": args.max_edit_support_pairs,
        "use_slap_seed": not args.no_slap_seed,
        "max_symmetry_automorphisms": args.max_symmetry_automorphisms,
        "symmetry_timeout_seconds": args.symmetry_timeout_seconds,
        "symmetry_max_search_nodes": args.symmetry_max_search_nodes,
        "structure_analysis": not args.no_structure_analysis,
        "template_radius": args.template_radius,
        "structure_timeout_seconds": args.structure_timeout_seconds,
        "structure_max_search_nodes": args.structure_max_search_nodes,
        "tolerance": args.tolerance,
        "maximum_workers": args.workers,
        "thread_limit_per_worker": 1,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synister_reference_blinded_global_shell_campaign",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256_file(args.dataset),
        "rows": len(rows),
        "implementation_sha256": _implementation_sha256(),
        "options": options,
    }
    manifest["manifest_sha256"] = _payload_sha256(manifest)
    run_options = dict(options)
    run_options["campaign_manifest_sha256"] = manifest["manifest_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing campaign manifest does not match options")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    pending = [
        row
        for row in rows
        if not _case_is_current(
            _case_path(args.output, int(row["source_line"])),
            row,
            manifest["manifest_sha256"],
        )
    ]
    remaining_cases = len(pending)
    recorded_cases = len(rows) - remaining_cases
    if args.max_cases is not None:
        pending = pending[: args.max_cases]
    print(
        json.dumps(
            {
                "event": "campaign_start",
                "manifest_sha256": manifest["manifest_sha256"],
                "pending_cases": len(pending),
                "recorded_cases": recorded_cases,
                "remaining_cases": remaining_cases,
                "workers": args.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.workers == 1:
        records = (_safe_run_case(row, run_options) for row in pending)
    else:
        records = _parallel_case_records(
            pending,
            run_options,
            workers=args.workers,
            memory_limit_gib=args.memory_limit_gib,
        )
    completed_this_run = 0
    for completed_this_run, record in enumerate(records, 1):
        _atomic_gzip_json(_case_path(args.output, int(record["source_line"])), record)
        print(
            json.dumps(
                {
                    "completed_this_run": completed_this_run,
                    "remaining_this_run": len(pending) - completed_this_run,
                    "source_line": record["source_line"],
                    "reaction_id": record["reaction_id"],
                    "statuses": (
                        {
                            mode: shell["status"]
                            for mode, shell in record.get("shells", {}).items()
                        }
                        or {"case": record.get("status", "error")}
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _write_summary(args.output, manifest)
    print(
        json.dumps(
            {
                "event": "campaign_finish",
                "manifest_sha256": manifest["manifest_sha256"],
                "recorded_cases": recorded_cases + completed_this_run,
                "remaining_cases": remaining_cases - completed_this_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
