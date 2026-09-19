#!/usr/bin/env python3
"""Wait for a Synister campaign, then rerun its timeout cases at 120 seconds."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_KIND = "synister_reference_blinded_global_shell_case"


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


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, sort_keys=True), flush=True)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("rt", encoding="utf-8") as stream:
        return json.load(stream)


def _read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="ascii") as stream:
        return json.load(stream)


def _service_is_active(service: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", service],
        check=False,
    )
    return result.returncode == 0


def _wait_for_source(service: str, poll_seconds: float) -> None:
    polls = 0
    while _service_is_active(service):
        if polls % 10 == 0:
            _event("waiting_for_source_campaign", service=service)
        polls += 1
        time.sleep(poll_seconds)
    _event("source_campaign_stopped", service=service)


def _validate_completed_baseline(
    baseline: Path, dataset: Path
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = baseline / "manifest.json"
    summary_path = baseline / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("source campaign stopped without manifest and summary")
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    if int(summary.get("remaining_cases", -1)) != 0:
        raise RuntimeError("source campaign summary is incomplete")
    if int(summary.get("error_records", -1)) != 0:
        raise RuntimeError("source campaign contains error records")
    if summary.get("campaign_manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError("source summary and manifest do not match")
    if manifest.get("dataset_sha256") != _sha256_file(dataset):
        raise RuntimeError("source dataset does not match the campaign manifest")
    options = manifest.get("options", {})
    if float(options.get("time_limit_per_shell", -1)) != 60.0:
        raise RuntimeError("source campaign is not the expected 60-second run")
    if options.get("mode") != "both":
        raise RuntimeError("source campaign did not run both shell modes")
    return manifest, summary


def _timeout_source_lines(
    baseline: Path, manifest: dict[str, object]
) -> set[int]:
    expected_manifest = manifest["manifest_sha256"]
    expected_records = int(manifest["rows"])
    timeout_lines: set[int] = set()
    records = 0
    for path in sorted((baseline / "cases").glob("line_*.json.gz")):
        record = _read_gzip_json(path)
        claimed_digest = record.get("record_sha256")
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        if claimed_digest != _payload_sha256(unsigned):
            raise RuntimeError(f"invalid atomic record digest: {path}")
        if record.get("kind") != CASE_KIND:
            raise RuntimeError(f"unexpected atomic record kind: {path}")
        if record.get("campaign_manifest_sha256") != expected_manifest:
            raise RuntimeError(f"atomic record belongs to another campaign: {path}")
        if record.get("status") == "error":
            raise RuntimeError(f"source campaign error record: {path}")
        records += 1
        shells = record.get("shells", {})
        if any(shell.get("status") == "timeout" for shell in shells.values()):
            timeout_lines.add(int(record["source_line"]))
    if records != expected_records:
        raise RuntimeError(
            f"expected {expected_records} atomic records, found {records}"
        )
    return timeout_lines


def _write_timeout_dataset(
    source: Path, destination: Path, timeout_lines: set[int]
) -> int:
    opener = gzip.open if source.suffix == ".gz" else open
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    selected = 0
    seen: set[int] = set()
    try:
        with opener(source, "rt", encoding="utf-8", newline="") as input_stream:
            reader = csv.DictReader(input_stream)
            fieldnames = ["source_line", "reaction_id", "mapped_reaction"]
            if reader.fieldnames != fieldnames:
                raise RuntimeError("source dataset has an unexpected schema")
            with os.fdopen(descriptor, "wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
                ) as compressed:
                    with io.TextIOWrapper(
                        compressed, encoding="utf-8", newline=""
                    ) as output_stream:
                        writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in reader:
                            source_line = int(row["source_line"])
                            if source_line not in timeout_lines:
                                continue
                            writer.writerow(row)
                            selected += 1
                            seen.add(source_line)
        missing = timeout_lines - seen
        if missing:
            raise RuntimeError(
                f"timeout source lines are absent from the dataset: {sorted(missing)[:10]}"
            )
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return selected


def _write_provenance(
    output: Path,
    baseline: Path,
    manifest: dict[str, object],
    timeout_lines: set[int],
    timeout_dataset: Path,
) -> None:
    payload = {
        "kind": "synister_timeout_followup_selection",
        "source_output": str(baseline),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_time_limit_per_shell": 60,
        "followup_time_limit_per_shell": 120,
        "selection": "either reference_cd or minimal status equals timeout",
        "selected_cases": len(timeout_lines),
        "selected_source_lines": sorted(timeout_lines),
        "dataset": str(timeout_dataset),
        "dataset_sha256": _sha256_file(timeout_dataset),
    }
    payload["selection_sha256"] = _payload_sha256(payload)
    path = output / "timeout_selection.json"
    temporary = output / ".timeout_selection.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _launch_followup(args: argparse.Namespace, timeout_dataset: Path) -> None:
    unit = args.followup_service.removesuffix(".service")
    log_path = args.output / "campaign.log"
    subprocess.run(
        ["systemctl", "--user", "reset-failed", f"{unit}.service"], check=False
    )
    command = [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={unit}",
        "--description=Synister timeout follow-up: 16 workers, 120-second shells",
        f"--working-directory={ROOT}",
        "--property=CPUQuota=1600%",
        "--property=CPUWeight=200",
        "--property=MemoryHigh=infinity",
        "--property=MemoryMax=64G",
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=stop",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        sys.executable,
        str(ROOT / "scripts/run_synister_global_shells.py"),
        "--dataset",
        str(timeout_dataset),
        "--output",
        str(args.output),
        "--mode",
        "both",
        "--workers",
        str(args.workers),
        "--time-limit-per-shell",
        str(args.time_limit_per_shell),
        "--memory-limit-gib",
        str(args.memory_limit_gib),
    ]
    subprocess.run(command, check=True)
    _event(
        "followup_launched",
        service=f"{unit}.service",
        output=str(args.output),
        timeout_cases_dataset=str(timeout_dataset),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-service", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--followup-service", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--time-limit-per-shell", type=float, default=120.0)
    parser.add_argument("--memory-limit-gib", type=float, default=4.0)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.baseline_output = args.baseline_output.resolve()
    args.output = args.output.resolve()
    if args.poll_seconds <= 0 or args.workers < 1:
        raise ValueError("poll interval and worker count must be positive")
    if args.time_limit_per_shell != 120:
        raise ValueError("this follow-up must use a 120-second shell timeout")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    args.output.mkdir(parents=True, exist_ok=True)
    _event(
        "followup_queued",
        source_service=args.source_service,
        baseline_output=str(args.baseline_output),
        output=str(args.output),
    )
    _wait_for_source(args.source_service, args.poll_seconds)
    manifest, _ = _validate_completed_baseline(args.baseline_output, args.dataset)
    timeout_lines = _timeout_source_lines(args.baseline_output, manifest)
    _event("timeout_cases_selected", selected_cases=len(timeout_lines))
    if not timeout_lines:
        _event("followup_not_required")
        return 0

    timeout_dataset = args.output / "timeout_cases.csv.gz"
    selected = _write_timeout_dataset(args.dataset, timeout_dataset, timeout_lines)
    if selected != len(timeout_lines):
        raise RuntimeError("timeout dataset selection count mismatch")
    _write_provenance(
        args.output,
        args.baseline_output,
        manifest,
        timeout_lines,
        timeout_dataset,
    )
    _launch_followup(args, timeout_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
