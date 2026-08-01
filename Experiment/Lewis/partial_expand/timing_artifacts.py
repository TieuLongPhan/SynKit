"""Compact reaction-level timing metadata for partial-AAM benchmarks."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable, TextIO

SCHEMA = "synkit.partial-aam-reaction-timings/1"


def open_text(path: Path, mode: str) -> TextIO:
    """Open plain text or gzip based on the file content/name."""
    compressed = (
        path.suffix == ".gz" if "w" in mode else path.read_bytes()[:2] == b"\x1f\x8b"
    )
    opener = gzip.open if compressed else open
    return opener(path, mode, encoding="utf-8")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one dataset."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_timing_artifact(
    *,
    source: Path,
    output: Path,
    dataset: Path,
    method: str,
    repetition: int,
    normalize_status: Callable[[str], str] = str,
) -> None:
    """Extract compact per-reaction timings from a JSON-lines case file."""
    samples = []
    counts: Counter[str] = Counter()
    observed_records = set()
    with open_text(source, "rt") as handle:
        for line in handle:
            raw = json.loads(line)
            observed_method = str(raw["method"])
            if observed_method != method:
                raise ValueError(
                    f"Expected {method!r} timing row, found {observed_method!r}"
                )
            record_id = int(raw["record_id"])
            if record_id in observed_records:
                raise ValueError(f"Duplicate timing record {record_id} in {source}")
            observed_records.add(record_id)
            seconds = float(raw["generation_seconds"])
            if seconds < 0:
                raise ValueError("Generation time cannot be negative")
            status = normalize_status(str(raw["status"]))
            counts[status.lower()] += 1
            samples.append(
                {
                    "record_id": record_id,
                    "generation_seconds": seconds,
                    "status": status,
                }
            )
    payload = {
        "schema": SCHEMA,
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": sha256(dataset),
        },
        "method": method,
        "repetition": repetition,
        "unit": "seconds",
        "counts": dict(sorted(counts.items())),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_text(output, "wt") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
