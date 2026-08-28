"""Export exact non-reference ITS representatives for mapped reactions.

The input dataset is explicit and is never copied into the output.  Each query
enumerates a global exact-CD or globally minimal shell.  A reference seed can
change the incumbent and traversal order, but never the candidate space.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import sys
import tempfile
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
    enumerate_mapped_reaction_its_alternatives,
)

SCHEMA_VERSION = 1


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
        ROOT / "synkit/Chem/Mapper/alternatives.py",
        ROOT / "synkit/Chem/Mapper/analysis.py",
        ROOT / "synkit/Chem/Mapper/spectrum.py",
        ROOT / "synkit/Chem/Mapper/chem/blind.py",
        ROOT / "synkit/Chem/Mapper/exact/distance.py",
        ROOT / "synkit/Chem/Mapper/exact/distance_bounds.py",
        ROOT / "synkit/Chem/Mapper/exact/distance_records.py",
        ROOT / "synkit/Chem/Mapper/exact/edit_support.py",
        ROOT / "synkit/Chem/Mapper/exact/hybrid.py",
        ROOT / "synkit/Chem/Mapper/exact/symmetry.py",
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


def _parse_target(value):
    normalized = value.lower()
    if normalized in {"reference", "minimal"}:
        return normalized
    try:
        target = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target must be 'reference', 'minimal', or a non-negative number"
        ) from error
    if target < 0:
        raise argparse.ArgumentTypeError("numeric target must be non-negative")
    return target


def _load_row(dataset, source_line):
    opener = gzip.open if dataset.suffix == ".gz" else open
    with opener(dataset, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["source_line"]) == source_line:
                return row
    raise ValueError(f"source line {source_line} is absent from the dataset")


def _reaction_from_row(row):
    reaction = str(row["mapped_reaction"])
    if "|" not in reaction:
        return reaction
    reaction, embedded_id = reaction.rsplit("|", 1)
    if embedded_id != row["reaction_id"]:
        raise ValueError("reaction ID does not match embedded provenance")
    return reaction


def _configure_memory_limit(memory_limit_gib):
    limit = int(memory_limit_gib * 1024**3)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, int(hard))
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return limit


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-line", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        help="repeat for reference, minimal, or numeric CD; default: reference",
    )
    parser.add_argument(
        "--seed-mode",
        action="append",
        choices=("reference", "slap", "none"),
        help="repeat to form a target-by-seed comparison; default: reference",
    )
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--include-explicit-hydrogens", action="store_true")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--memory-limit-gib", type=float, default=6.0)
    parser.add_argument("--max-mappings", type=int, default=100_000)
    parser.add_argument("--max-bijections", type=int)
    parser.add_argument("--blind-seed", default="synister-alternatives-v1")
    parser.add_argument("--no-symmetry-pruning", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.source_line < 1:
        raise ValueError("source-line must be positive")
    if args.time_limit < 0:
        raise ValueError("time-limit must be non-negative")
    if args.memory_limit_gib <= 0:
        raise ValueError("memory-limit-gib must be positive")
    memory_limit = _configure_memory_limit(args.memory_limit_gib)
    row = _load_row(args.dataset, args.source_line)
    reaction = _reaction_from_row(row)
    targets = args.target or ["reference"]
    seed_modes = args.seed_mode or ["reference"]
    config = GlobalShellConfig(
        binary=args.binary,
        time_limit_seconds=args.time_limit,
        max_bijections=args.max_bijections,
        max_mappings=args.max_mappings,
        symmetry_pruning=not args.no_symmetry_pruning,
        backend="auto",
        structure_analysis=True,
    )
    queries = []
    for target in targets:
        for seed_mode in seed_modes:
            result = enumerate_mapped_reaction_its_alternatives(
                reaction,
                CD=target,
                seed_mode=seed_mode,
                config=config,
                heavy_only=not args.include_explicit_hydrogens,
                blind_seed=args.blind_seed,
            )
            queries.append(
                {
                    "requested_target": target,
                    "seed_mode": seed_mode,
                    "result": result.as_dict(),
                }
            )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synister_exact_alternative_its_application",
        "dataset_name": args.dataset.name,
        "dataset_sha256": _sha256_file(args.dataset),
        "implementation_sha256": _implementation_sha256(),
        "source_line": args.source_line,
        "reaction_id": row["reaction_id"],
        "reaction_sha256": hashlib.sha256(reaction.encode("utf-8")).hexdigest(),
        "options": {
            "binary": args.binary,
            "heavy_only": not args.include_explicit_hydrogens,
            "time_limit_seconds_per_query": args.time_limit,
            "memory_limit_bytes": memory_limit,
            "max_mappings": args.max_mappings,
            "max_bijections": args.max_bijections,
            "symmetry_pruning": not args.no_symmetry_pruning,
            "maximum_workers": 1,
            "thread_limit": 1,
            "blind_seed": args.blind_seed,
        },
        "queries": queries,
        "resource": {
            "maximum_resident_set_kib": int(usage.ru_maxrss),
        },
    }
    payload["record_sha256"] = _payload_sha256(payload)
    _atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
