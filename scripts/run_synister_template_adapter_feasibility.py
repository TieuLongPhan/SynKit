"""Run the frozen full-atom source-replay template-adapter feasibility cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Chem.Mapper import (
    GlobalShellConfig,
    correspondence_from_export,
    enumerate_mapped_reaction_its_alternatives,
    replay_executable_rule_on_source,
)


def _sha256_bytes(payload: object) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_rows(dataset: Path, selection: dict[str, object]):
    expected = {int(record["r_id"]): record for record in selection["records"]}
    rows = {}
    opener = gzip.open if dataset.suffix == ".gz" else open
    with opener(dataset, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            r_id = int(row["r_id"])
            if r_id in expected:
                rows[r_id] = row
    if set(rows) != set(expected):
        raise ValueError("dataset does not contain every selected r_id")
    for r_id, selected in expected.items():
        row = rows[r_id]
        reaction = str(row["ground_truth"])
        if row["original_id"] != selected["original_id"]:
            raise ValueError(f"selected original ID mismatch for r_id {r_id}")
        if hashlib.sha256(reaction.encode("utf-8")).hexdigest() != selected[
            "ground_truth_sha256"
        ]:
            raise ValueError(f"selected reaction digest mismatch for r_id {r_id}")
    return rows, expected


def run(dataset: Path, selection_path: Path, *, time_limit_seconds: float = 30.0):
    """Enumerate complete full-atom shells and audit each executable class."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["elementary_dataset_sha256"] != _sha256_file(dataset):
        raise ValueError("elementary dataset digest does not match frozen selection")
    rows, expected = _selected_rows(dataset, selection)
    config = GlobalShellConfig(
        binary=False,
        time_limit_seconds=time_limit_seconds,
        max_bijections=None,
        max_mappings=None,
        symmetry_pruning=False,
        structure_analysis=True,
    )
    records = []
    for r_id, selected in sorted(expected.items()):
        row = rows[r_id]
        started = time.monotonic()
        result = enumerate_mapped_reaction_its_alternatives(
            row["ground_truth"],
            CD="minimal",
            seed_mode="none",
            heavy_only=False,
            config=config,
        )
        output = result.as_dict()
        shell = output["shell"]
        replays = []
        if shell["complete"]:
            for class_record in shell["classes"]:
                try:
                    replay = replay_executable_rule_on_source(
                        row["ground_truth"],
                        correspondence_from_export(class_record),
                        heavy_only=bool(output["heavy_only"]),
                    )
                    replays.append(
                        {
                            "its_class_id": class_record["its_class_id"],
                            "generated_product_count": len(replay.generated_products),
                            "recovered": replay.recovered,
                        }
                    )
                except Exception as error:  # recorded feasibility failure
                    replays.append(
                        {
                            "its_class_id": class_record["its_class_id"],
                            "conversion_error": f"{type(error).__name__}: {error}",
                            "recovered": False,
                        }
                    )
        records.append(
            {
                "r_id": r_id,
                "original_id": selected["original_id"],
                "shell_status": shell["status"],
                "shell_complete": shell["complete"],
                "shell_its_class_count": shell["shell_its_class_count"],
                "elapsed_seconds": shell["elapsed_seconds"],
                "end_to_end_seconds": time.monotonic() - started,
                "replays": replays,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "synister_template_adapter_feasibility_run",
        "selection_sha256": _sha256_file(selection_path),
        "options": {
            "CD": "minimal",
            "seed_mode": "none",
            "heavy_only": False,
            "binary": False,
            "time_limit_seconds": time_limit_seconds,
            "max_mappings": None,
            "symmetry_pruning": False,
        },
        "records": records,
        "maximum_resident_set_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    payload["record_sha256"] = _sha256_bytes(payload)
    return payload


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.time_limit_seconds < 0:
        raise ValueError("time limit must be non-negative")
    result = run(args.dataset, args.selection, time_limit_seconds=args.time_limit_seconds)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
