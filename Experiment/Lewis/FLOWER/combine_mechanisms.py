#!/usr/bin/env python3
"""Combine FLOWER step graphs into full source-to-terminal reactions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.FLOWER.prepare_batches import (  # noqa: E402
    deterministic_gzip_text,
    sha256,
)

DEFAULT_INPUTS = (HERE / "train.txt", HERE / "val.txt", HERE / "test.txt")
DEFAULT_OUTPUT = HERE / "full_reactions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Elementary-step split files (default: train, val, test).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Exclude the PC, PM, RC, and RS category blocks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated full-reaction files.",
    )
    return parser.parse_args()


def parse_row(line: str, row_number: int) -> tuple[str, str, str]:
    text = line.rstrip("\n")
    reaction, separator, label = text.rpartition("|")
    if not separator or reaction.count(">>") != 1:
        raise ValueError(f"Malformed FLOWER row {row_number}")
    reactants, products = reaction.split(">>", 1)
    return label, reactants, products


def iter_label_blocks(
    path: Path,
) -> Iterator[tuple[str, int, list[tuple[str, str]]]]:
    occurrences: Counter[str] = Counter()
    current_label: str | None = None
    edges: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, line in enumerate(handle, start=1):
            label, reactants, products = parse_row(line, row_number)
            if current_label is not None and label != current_label:
                occurrences[current_label] += 1
                yield current_label, occurrences[current_label], edges
                edges = []
            current_label = label
            edges.append((reactants, products))
    if current_label is not None:
        occurrences[current_label] += 1
        yield current_label, occurrences[current_label], edges


def _component_nodes(
    nodes: dict[str, int],
    undirected: dict[str, set[str]],
) -> Iterator[list[str]]:
    seen: set[str] = set()
    for root in nodes:
        if root in seen:
            continue
        component: list[str] = []
        stack = [root]
        seen.add(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in undirected[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        component.sort(key=nodes.__getitem__)
        yield component


def combine_block(
    label: str,
    occurrence: int,
    edges: Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    nodes: dict[str, int] = {}
    directed: dict[str, set[str]] = defaultdict(set)
    undirected: dict[str, set[str]] = defaultdict(set)
    indegree: Counter[str] = Counter()
    outdegree: Counter[str] = Counter()

    for reactants, products in edges:
        nodes.setdefault(reactants, len(nodes))
        nodes.setdefault(products, len(nodes))
        undirected[reactants].add(products)
        undirected[products].add(reactants)
        if reactants != products and products not in directed[reactants]:
            directed[reactants].add(products)
            indegree[products] += 1
            outdegree[reactants] += 1

    records: list[tuple[str, str]] = []
    report: dict[str, Any] = {
        "components": 0,
        "skipped_cyclic_components": 0,
        "source_terminal_pairs": 0,
        "skipped_samples": [],
    }
    for component_index, component in enumerate(
        _component_nodes(nodes, undirected),
        start=1,
    ):
        report["components"] += 1
        sources = [node for node in component if indegree[node] == 0]
        terminals = {node for node in component if outdegree[node] == 0}
        if not sources or not terminals:
            report["skipped_cyclic_components"] += 1
            if len(report["skipped_samples"]) < 10:
                report["skipped_samples"].append(
                    {
                        "label": label,
                        "occurrence": occurrence,
                        "component": component_index,
                        "states": len(component),
                        "sources": len(sources),
                        "terminals": len(terminals),
                    }
                )
            continue

        pair_index = 0
        for source in sources:
            reachable = {source}
            stack = [source]
            while stack:
                node = stack.pop()
                for product in directed[node]:
                    if product not in reachable:
                        reachable.add(product)
                        stack.append(product)
            for terminal in component:
                if terminal not in terminals or terminal not in reachable:
                    continue
                pair_index += 1
                provenance = f"{label}:b{occurrence}:c{component_index}:p{pair_index}"
                records.append((f"{source}>>{terminal}", provenance))
                report["source_terminal_pairs"] += 1
    return records, report


def combine_split(
    input_path: Path,
    output_path: Path,
    *,
    numeric_only: bool,
) -> dict[str, Any]:
    rows = 0
    blocks = 0
    components = 0
    skipped = 0
    output_rows = 0
    category_rows: Counter[str] = Counter()
    duplicate_reactions = 0
    reaction_digests: set[bytes] = set()
    skipped_samples: list[dict[str, Any]] = []
    content_digest = hashlib.sha256()

    with deterministic_gzip_text(output_path) as output:
        for label, occurrence, edges in iter_label_blocks(input_path):
            rows += len(edges)
            blocks += 1
            if numeric_only and not label.isdigit():
                continue
            records, block_report = combine_block(label, occurrence, edges)
            components += int(block_report["components"])
            skipped += int(block_report["skipped_cyclic_components"])
            room = 20 - len(skipped_samples)
            if room > 0:
                skipped_samples.extend(block_report["skipped_samples"][:room])
            category = "numeric" if label.isdigit() else label
            for reaction, provenance in records:
                line = f"{reaction}|{provenance}\n"
                output.write(line)
                content_digest.update(line.encode())
                output_rows += 1
                category_rows[category] += 1
                digest = hashlib.sha256(reaction.encode()).digest()
                if digest in reaction_digests:
                    duplicate_reactions += 1
                else:
                    reaction_digests.add(digest)

    return {
        "input": {
            "path": str(input_path.resolve()),
            "bytes": input_path.stat().st_size,
            "rows": rows,
            "sha256": sha256(input_path),
        },
        "output": {
            "path": str(output_path.resolve()),
            "rows": output_rows,
            "directional_replays": output_rows * 2,
            "compressed_bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "content_sha256": content_digest.hexdigest(),
        },
        "label_blocks": blocks,
        "weak_components": components,
        "skipped_cyclic_components": skipped,
        "skipped_samples": skipped_samples,
        "rows_by_category": dict(sorted(category_rows.items())),
        "duplicate_reaction_rows": duplicate_reactions,
        "unique_reactions": len(reaction_digests),
    }


def combine_mechanisms(
    inputs: list[Path],
    output_dir: Path,
    *,
    numeric_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    resolved = [path.resolve() for path in inputs]
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FLOWER inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{path.stem}.txt.gz" for path in resolved]
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in [*outputs, manifest_path] if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Generated output already exists; pass --force to replace it: "
            f"{existing[0]}"
        )

    split_reports = []
    for input_path, output_path in zip(resolved, outputs):
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        if temporary.exists():
            temporary.unlink()
        try:
            report = combine_split(
                input_path,
                temporary,
                numeric_only=numeric_only,
            )
            temporary.replace(output_path)
            report["output"]["path"] = str(output_path.resolve())
            report["output"]["sha256"] = sha256(output_path)
            split_reports.append(report)
        finally:
            if temporary.exists():
                temporary.unlink()

    manifest = {
        "schema": "synkit.flower-full-reactions/1",
        "policy": {
            "grouping": "contiguous label block, then weak state component",
            "reaction": "each reachable source-to-terminal pair",
            "self_loops": "ignored for source and terminal degree",
            "cyclic_components": "skipped and reported",
            "numeric_only": numeric_only,
            "duplicates": "retained and counted",
        },
        "splits": split_reports,
        "total_rows": sum(item["output"]["rows"] for item in split_reports),
        "total_directional_replays": sum(
            item["output"]["directional_replays"] for item in split_reports
        ),
        "total_skipped_cyclic_components": sum(
            item["skipped_cyclic_components"] for item in split_reports
        ),
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
    manifest = combine_mechanisms(
        list(args.inputs),
        args.output_dir.resolve(),
        numeric_only=args.numeric_only,
        force=args.force,
    )
    for report in manifest["splits"]:
        print(
            Path(report["output"]["path"]).name,
            report["output"]["rows"],
            "full reactions",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
