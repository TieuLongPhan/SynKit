#!/usr/bin/env python3
"""Reproduce forward/backward rule replay for tuple and typesGH rules."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import inspect
import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any

from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    POLAR_DATASET,
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    open_text,
    read_json,
    sha256,
    timing_summary,
    write_json,
)
from synkit.IO.chem_converter import rsmi_to_its  # noqa: E402
from synkit.Rule import SynRule  # noqa: E402
from synkit.Synthesis.Reactor.syn_reactor import SynReactor  # noqa: E402

REPRESENTATIONS = ("tuple", "typesGH")
DIRECTIONS = ("forward", "backward")
LEGACY_NODE_ATTRS = (
    "element",
    "aromatic",
    "hcount",
    "charge",
    "neighbors",
    "atom_map",
)
LEGACY_EDGE_ATTRS = ("order",)
HAS_FORMAT = "format" in inspect.signature(SynRule.__init__).parameters


class CaseTimeout(TimeoutError):
    """Raised when one replay direction exceeds its wall-time ceiling."""


def _raise_timeout(_signum, _frame) -> None:
    raise CaseTimeout("Replay direction exceeded the case timeout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=POLAR_DATASET)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=DIRECTIONS,
        default=list(DIRECTIONS),
    )
    parser.add_argument("--record-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-timeout",
        type=float,
        help="Optional per-direction timeout in seconds (default: no timeout)",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=int,
        help="Optional embedding cap (default: complete uncapped enumeration)",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "bidirectional-replay",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {"record_id": int(row["R-id"]), "reaction": str(row["smart"])}
        for row in read_json(path)
    ]


def extract_rule(reaction: str, representation: str) -> SynRule:
    if representation == "typesGH":
        graph = rsmi_to_its(
            reaction,
            core=True,
            drop_non_aam=False,
            use_index_as_atom_map=True,
            node_attrs=LEGACY_NODE_ATTRS,
            edge_attrs=LEGACY_EDGE_ATTRS,
            format="typesGH",
        )
        return SynRule(graph, canon=False, implicit_h=True, format="typesGH")
    graph = rsmi_to_its(
        reaction,
        core=True,
        drop_non_aam=False,
        use_index_as_atom_map=True,
        format="tuple",
    )
    return SynRule(graph, canon=False, implicit_h=True, format="tuple")


def make_reactor(
    host: str,
    rule: SynRule,
    representation: str,
    direction: str,
    embedding_threshold: int | None,
) -> SynReactor:
    return SynReactor(
        host,
        rule,
        invert=direction == "backward",
        explicit_h=False,
        implicit_temp=False,
        automorphism=True,
        embed_threshold=embedding_threshold,
        embed_pre_filter=True,
        template_format=representation,
        radical_policy="strict" if representation == "tuple" else "ignore",
        stereo_mode="ignore",
    )


def replay_direction(
    *,
    host: str,
    expected: str,
    rule: SynRule,
    representation: str,
    direction: str,
    embedding_threshold: int | None,
    case_timeout: float | None,
) -> dict[str, Any]:
    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    if case_timeout is not None:
        signal.setitimer(signal.ITIMER_REAL, case_timeout)
    started = time.perf_counter()
    try:
        reactor = make_reactor(
            host,
            rule,
            representation,
            direction,
            embedding_threshold,
        )
        mappings = reactor.mappings
        rewritten = reactor.its_list
        reactions = reactor.smarts_list
        generated = {canonical_unmapped_reaction(item) for item in reactions}
        recovered = expected in generated
        return {
            "status": "PASS" if recovered else "FAIL",
            "reference_recovered": recovered,
            "mapping_count": len(mappings),
            "rewrite_count": len(rewritten),
            "serialized_count": len(reactions),
            "unique_reaction_count": len(generated),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "stage": getattr(exc, "stage", None),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "seconds": time.perf_counter() - started,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def benchmark(
    representation: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    output = args.output_dir / f"{representation}-cases.jsonl.gz"
    wall_started = time.perf_counter()
    with open_text(output, "wt") as handle:
        for index, row in enumerate(rows, start=1):
            record_id = int(row["record_id"])
            reaction = str(row["reaction"])
            case: dict[str, Any] = {"record_id": record_id, "directions": {}}
            try:
                reactants, products = reaction.split(">>", 1)
                expected = canonical_unmapped_reaction(reaction)
                hosts = {
                    "forward": canonical_unmapped_side(reactants),
                    "backward": canonical_unmapped_side(products),
                }
                extraction_started = time.perf_counter()
                rule = extract_rule(reaction, representation)
                case["extraction_seconds"] = time.perf_counter() - extraction_started
                counts["rule_extracted"] += 1
            except Exception as exc:
                counts["rule_extraction_error"] += 1
                case.update(
                    status="RULE_EXTRACTION_ERROR",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                handle.write(json.dumps(case, sort_keys=True) + "\n")
                continue

            for direction in args.directions:
                result = replay_direction(
                    host=hosts[direction],
                    expected=expected,
                    rule=rule,
                    representation=representation,
                    direction=direction,
                    embedding_threshold=args.embedding_threshold,
                    case_timeout=args.case_timeout,
                )
                case["directions"][direction] = result
                durations[direction].append(float(result["seconds"]))
                counts[f"{direction}:attempted"] += 1
                counts[f"{direction}:{result['status'].lower()}"] += 1
                if result.get("error_type") == "CaseTimeout":
                    counts[f"{direction}:timeout"] += 1
            case["status"] = (
                "PASS"
                if all(item["status"] == "PASS" for item in case["directions"].values())
                else "INCOMPLETE"
            )
            handle.write(json.dumps(case, sort_keys=True) + "\n")
            if args.progress_every and index % args.progress_every == 0:
                print(
                    f"{representation}: {index}/{len(rows)}",
                    flush=True,
                )

    report = {
        "schema": "synkit.bidirectional-rule-replay/1",
        "representation": representation,
        "directions": args.directions,
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": sha256(args.dataset.resolve()),
        },
        "selection": {
            "rows": len(rows),
            "record_ids": args.record_ids,
            "limit": args.limit,
        },
        "policy": {
            "case_timeout_seconds": args.case_timeout,
            "embedding_threshold": args.embedding_threshold,
            "automorphism": True,
            "recovery": "canonical full reaction without AAM or stereo",
        },
        "counts": dict(sorted(counts.items())),
        "timing_seconds": {
            "wall": time.perf_counter() - wall_started,
            "directions": {
                name: timing_summary(values)
                for name, values in sorted(durations.items())
            },
        },
        "case_file": str(output.resolve()),
    }
    write_json(args.output_dir / f"{representation}-summary.json", report)
    return report


def main() -> int:
    args = parse_args()
    if not HAS_FORMAT:
        raise RuntimeError("Current tuple/typesGH format selection is unavailable")
    if args.case_timeout is not None and args.case_timeout <= 0:
        raise ValueError("Timeout must be positive")
    if args.embedding_threshold is not None and args.embedding_threshold < 1:
        raise ValueError("Embedding threshold must be positive")
    RDLogger.DisableLog("rdApp.*")
    logging.disable(logging.INFO)
    rows = load_rows(args.dataset.resolve())
    if args.record_ids:
        selected = set(args.record_ids)
        rows = [row for row in rows if row["record_id"] in selected]
        missing = selected - {int(row["record_id"]) for row in rows}
        if missing:
            raise ValueError(f"Unknown record IDs: {sorted(missing)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [benchmark(name, rows, args) for name in args.representations]
    write_json(
        args.output_dir / "summary.json",
        {
            "schema": "synkit.bidirectional-rule-replay-matrix/1",
            "reports": reports,
        },
    )
    for report in reports:
        print(report["representation"], json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
