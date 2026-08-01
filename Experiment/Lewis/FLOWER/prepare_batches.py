#!/usr/bin/env python3
"""Concatenate FLOWER splits into deterministic, balanced gzip batches."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterator, TextIO

HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = (HERE / "train.txt", HERE / "val.txt", HERE / "test.txt")
DEFAULT_OUTPUT = HERE / "batches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Input files in concatenation order (default: train, val, test).",
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated batch set.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def input_text(path: Path) -> Iterator[TextIO]:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    # Universal-newline input makes content digests and generated gzip bytes
    # independent of the platform that created the source split.
    with opener(path, "rt", encoding="utf-8", newline=None) as handle:
        yield handle


def inspect_input(path: Path) -> dict[str, object]:
    content_digest = hashlib.sha256()
    rows = 0
    with input_text(path) as handle:
        for line in handle:
            encoded = line.encode()
            content_digest.update(encoded)
            rows += 1
            if not line.endswith("\n"):
                raise ValueError(f"Input row {rows} has no trailing newline: {path}")
            reaction, separator, _label = line.rstrip("\n").rpartition("|")
            if not separator or reaction.count(">>") != 1:
                raise ValueError(f"Malformed FLOWER row {rows}: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": sha256(path),
        "content_sha256": content_digest.hexdigest(),
    }


def batch_sizes(total_rows: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("Batch count must be positive")
    quotient, remainder = divmod(total_rows, count)
    return [quotient + (1 if index < remainder else 0) for index in range(count)]


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text:
                yield text


def prepare_batches(
    inputs: list[Path],
    output_dir: Path,
    count: int,
    *,
    force: bool = False,
) -> dict[str, object]:
    if not inputs:
        raise ValueError("At least one input is required")
    resolved_inputs = [path.resolve() for path in inputs]
    missing = [path for path in resolved_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FLOWER inputs: {missing}")

    input_reports = [inspect_input(path) for path in resolved_inputs]
    total_rows = sum(int(report["rows"]) for report in input_reports)
    sizes = batch_sizes(total_rows, count)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [
        output_dir / f"batch-{index:02d}-of-{count:02d}.txt.gz"
        for index in range(1, count + 1)
    ]
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in [*final_paths, manifest_path] if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Generated output already exists; pass --force to replace it: "
            f"{existing[0]}"
        )

    temporary_paths = [path.with_suffix(path.suffix + ".part") for path in final_paths]
    for path in temporary_paths:
        if path.exists():
            path.unlink()

    batch_reports: list[dict[str, object]] = []
    combined_content_digest = hashlib.sha256()
    input_index = 0
    with ExitStack() as input_stack:
        for path in temporary_paths:
            input_stack.callback(path.unlink, missing_ok=True)
        input_handle = input_stack.enter_context(
            input_text(resolved_inputs[input_index])
        )
        for index, (temporary, final, size) in enumerate(
            zip(temporary_paths, final_paths, sizes),
            start=1,
        ):
            source_rows: Counter[str] = Counter()
            batch_content_digest = hashlib.sha256()
            with deterministic_gzip_text(temporary) as output:
                for _ in range(size):
                    line = input_handle.readline()
                    while not line:
                        input_index += 1
                        if input_index >= len(resolved_inputs):
                            raise RuntimeError(
                                "Input ended before the declared row total"
                            )
                        input_handle = input_stack.enter_context(
                            input_text(resolved_inputs[input_index])
                        )
                        line = input_handle.readline()
                    output.write(line)
                    encoded = line.encode("utf-8")
                    batch_content_digest.update(encoded)
                    combined_content_digest.update(encoded)
                    source_rows[resolved_inputs[input_index].name] += 1
            temporary.replace(final)
            batch_reports.append(
                {
                    "index": index,
                    "path": str(final.resolve()),
                    "rows": size,
                    "directional_replays": size * 2,
                    "compressed_bytes": final.stat().st_size,
                    "sha256": sha256(final),
                    "content_sha256": batch_content_digest.hexdigest(),
                    "source_rows": dict(sorted(source_rows.items())),
                }
            )

    manifest: dict[str, object] = {
        "schema": "synkit.flower-rule-replay-batches/1",
        "concatenation_order": [path.name for path in resolved_inputs],
        "inputs": input_reports,
        "batch_count": count,
        "total_rows": total_rows,
        "total_directional_replays": total_rows * 2,
        "combined_content_sha256": combined_content_digest.hexdigest(),
        "batch_rows": sizes,
        "batches": batch_reports,
    }
    temporary_manifest = manifest_path.with_suffix(".json.part")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare_batches(
        list(args.inputs),
        args.output_dir.resolve(),
        args.batches,
        force=args.force,
    )
    print(
        f"Wrote {manifest['batch_count']} batches with "
        f"{manifest['total_rows']} rows and "
        f"{manifest['total_directional_replays']} directional replays."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
