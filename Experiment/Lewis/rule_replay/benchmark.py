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

import networkx as nx
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import (  # noqa: E402
    POLAR_DATASET,
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    open_text,
    read_json,
    sha256,
    timing_summary,
    unique_standardized_reactions,
    write_json,
)
from synkit.Graph.ITS.its_reverter import ITSReverter  # noqa: E402
from synkit.Graph.ITS.rc_extractor import RCExtractor  # noqa: E402
from synkit.IO.chem_converter import (  # noqa: E402
    DEFAULT_EDGE_ATTRS,
    DEFAULT_NODE_ATTRS,
    rsmi_to_its,
)
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
RESULTS_ROOT = HERE / "Data"


class CaseTimeout(BaseException):
    """Cancel a replay direction that exceeds its wall-time ceiling.

    This deliberately does not inherit from :class:`Exception`.  Candidate-
    level chemistry code uses broad ``except Exception`` handlers to reject
    malformed products and continue with the remaining embeddings.  A replay
    timeout is control flow rather than a malformed candidate and must cross
    those recovery boundaries unchanged.
    """


def _raise_timeout(_signum, _frame) -> None:
    raise CaseTimeout("Replay direction exceeded the case timeout")


def _supports_interval_timer() -> bool:
    return all(
        hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")
    )


def _set_timeout(seconds: float | None) -> None:
    if seconds is not None and _supports_interval_timer():
        signal.setitimer(signal.ITIMER_REAL, seconds)


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
    parser.add_argument(
        "--product-deduplication",
        choices=("structural", "deferred"),
        default="structural",
        help=(
            "Endpoint quotient policy: structural runs exact pre-rewrite "
            "application and post-rewrite attributed-ITS quotients (default); "
            "deferred is an output-set diagnostic and is excluded from the "
            "primary efficiency comparison"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "bidirectional-replay",
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
        include_context_edges=False,
    )
    return SynRule(graph, canon=False, implicit_h=True, format="tuple")


def extract_rule_and_hosts(
    reaction: str,
    representation: str,
) -> tuple[SynRule, dict[str, str | nx.Graph]]:
    """Parse one reaction once and reuse its endpoint graphs as replay hosts."""
    if representation == "typesGH":
        reactants, products = reaction.split(">>", 1)
        return extract_rule(reaction, representation), {
            "forward": canonical_unmapped_side(reactants),
            "backward": canonical_unmapped_side(products),
        }

    full = rsmi_to_its(
        reaction,
        core=False,
        drop_non_aam=False,
        use_index_as_atom_map=True,
        format="tuple",
    )
    core = RCExtractor(
        node_attrs=DEFAULT_NODE_ATTRS,
        edge_attrs=DEFAULT_EDGE_ATTRS,
        preserve_full_attrs=False,
    ).extract(full, include_context_edges=False)
    rule = SynRule(core, canon=False, implicit_h=True, format="tuple")
    reverter = ITSReverter(full)
    hosts = {
        "forward": reverter.to_graph(
            "reactant",
            node_attrs=DEFAULT_NODE_ATTRS,
            edge_attrs=DEFAULT_EDGE_ATTRS,
        ),
        "backward": reverter.to_graph(
            "product",
            node_attrs=DEFAULT_NODE_ATTRS,
            edge_attrs=DEFAULT_EDGE_ATTRS,
        ),
    }
    for direction, graph in hosts.items():
        # The benchmark ignores stereo and AAM in both matching and endpoint
        # identity. Relabel in insertion order because downstream RDKit
        # re-perception returns consecutive node IDs in that same order.
        # Retaining source atom-map IDs here could attach refreshed Kekule
        # fields to the wrong bonds when map order and insertion order differ.
        graph = nx.convert_node_labels_to_integers(
            graph,
            first_label=1,
            ordering="default",
        )
        graph.graph.pop("stereo_descriptors", None)
        for _, attrs in graph.nodes(data=True):
            attrs["atom_map"] = 0
        hosts[direction] = graph
    return rule, hosts


def make_reactor(
    host: str | nx.Graph,
    rule: SynRule,
    representation: str,
    direction: str,
    embedding_threshold: int | None,
    product_deduplication: str = "structural",
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
        product_deduplication=product_deduplication,
    )


def replay_direction(
    *,
    host: str | nx.Graph,
    expected: str,
    rule: SynRule,
    representation: str,
    direction: str,
    embedding_threshold: int | None,
    case_timeout: float | None,
    product_deduplication: str = "structural",
    failure_sample_limit: int = 0,
    standardized_reaction_sink: set[str] | None = None,
) -> dict[str, Any]:
    previous_handler = None
    if case_timeout is not None and _supports_interval_timer():
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    _set_timeout(case_timeout)
    started = time.perf_counter()
    stage = "reactor_construction"
    stage_started = started
    stage_seconds: dict[str, float] = {}
    mappings = None
    rewritten = None
    reactions = None
    try:
        reactor = make_reactor(
            host,
            rule,
            representation,
            direction,
            embedding_threshold,
            product_deduplication,
        )
        stage_seconds[stage] = time.perf_counter() - stage_started
        stage = "matching"
        stage_started = time.perf_counter()
        _set_timeout(case_timeout)
        mappings = reactor.mappings
        stage_seconds[stage] = time.perf_counter() - stage_started
        stage = "rewriting"
        stage_started = time.perf_counter()
        _set_timeout(case_timeout)
        rewritten = reactor.its_list
        stage_seconds[stage] = time.perf_counter() - stage_started
        # The timeout measures reaction expansion, not benchmark-only string
        # conversion and endpoint canonicalization. Keep those costs visible
        # below without misclassifying a completed reactor application.
        _set_timeout(0.0)
        stage = "serialization"
        stage_started = time.perf_counter()
        reactions = reactor.smarts_list
        stage_seconds[stage] = time.perf_counter() - stage_started
        stage = "canonicalization"
        stage_started = time.perf_counter()
        generated = unique_standardized_reactions(reactions)
        if standardized_reaction_sink is not None:
            standardized_reaction_sink.update(generated)
        stage_seconds[stage] = time.perf_counter() - stage_started
        recovered = expected in generated
        result = {
            "status": "PASS" if recovered else "FAIL",
            "reference_recovered": recovered,
            "mapping_count": len(mappings),
            "rewrite_count": len(rewritten),
            "serialized_count": len(reactions),
            "unique_reaction_count": len(generated),
            "unique_standardized_reaction_count": len(generated),
            "duplicate_reaction_count": len(reactions) - len(generated),
            "seconds": time.perf_counter() - started,
            "expansion_seconds": sum(
                stage_seconds[name]
                for name in ("reactor_construction", "matching", "rewriting")
            ),
            "stage_seconds": stage_seconds,
        }
        if not recovered and failure_sample_limit:
            ordered = sorted(generated)
            result.update(
                expected_reaction=expected,
                generated_sample=ordered[:failure_sample_limit],
                generated_sample_truncated=(len(ordered) > failure_sample_limit),
            )
        return result
    except (CaseTimeout, Exception) as exc:
        stage_seconds[stage] = time.perf_counter() - stage_started
        result = {
            "status": "ERROR",
            "stage": getattr(exc, "stage", None) or stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "seconds": time.perf_counter() - started,
            "stage_seconds": stage_seconds,
        }
        if mappings is not None:
            result["mapping_count"] = len(mappings)
        if rewritten is not None:
            result["rewrite_count"] = len(rewritten)
        if reactions is not None:
            result["serialized_count"] = len(reactions)
        return result
    finally:
        _set_timeout(0.0)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)


def benchmark(
    representation: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    global_unique: dict[str, set[str]] = {
        direction: set() for direction in args.directions
    }
    output = args.output_dir / f"{representation}-cases.jsonl.gz"
    wall_started = time.perf_counter()
    with open_text(output, "wt") as handle:
        for index, row in enumerate(rows, start=1):
            record_id = int(row["record_id"])
            reaction = str(row["reaction"])
            case: dict[str, Any] = {"record_id": record_id, "directions": {}}
            try:
                expected = canonical_unmapped_reaction(reaction)
                extraction_started = time.perf_counter()
                rule, hosts = extract_rule_and_hosts(reaction, representation)
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
                    product_deduplication=args.product_deduplication,
                    standardized_reaction_sink=global_unique[direction],
                )
                case["directions"][direction] = result
                durations[direction].append(float(result["seconds"]))
                counts[f"{direction}:attempted"] += 1
                counts[f"{direction}:{result['status'].lower()}"] += 1
                counts[f"{direction}:serialized"] += int(
                    result.get("serialized_count", 0)
                )
                counts[f"{direction}:unique_standardized"] += int(
                    result.get("unique_standardized_reaction_count", 0)
                )
                counts[f"{direction}:duplicates_removed"] += int(
                    result.get("duplicate_reaction_count", 0)
                )
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
            "timeout_scope": "each expansion stage",
            "embedding_threshold": args.embedding_threshold,
            "automorphism": True,
            "host_preparation": (
                "reuse parsed reaction-side graphs with insertion-order node normalization"
                if representation == "tuple"
                else "canonical endpoint SMILES"
            ),
            "product_deduplication": args.product_deduplication,
            "application_equivalence": (
                "complete valid mappings modulo exact transition-rule "
                "automorphisms; simple graphs retain every representative "
                "until endpoint certification"
                if args.product_deduplication == "structural"
                else "not computed; diagnostic output-set semantics only"
            ),
            "product_equivalence": (
                "injective canonical attributed-graph certificate after "
                "electron-state finalization, with exact tree/VF2 fallback"
                if args.product_deduplication == "structural"
                else "not computed; diagnostic output-set semantics only"
            ),
            "reaction_center_edge_policy": "changed",
            "recovery": "canonical full reaction without AAM or stereo",
            "output_deduplication": (
                "standardize both sides, canonicalize and sort components, "
                "then remove exact duplicates"
            ),
        },
        "counts": dict(sorted(counts.items())),
        "output_population": {
            direction: {
                "serialized_total": counts[f"{direction}:serialized"],
                "per_case_unique_standardized_total": counts[
                    f"{direction}:unique_standardized"
                ],
                "duplicates_removed_within_cases": counts[
                    f"{direction}:duplicates_removed"
                ],
                "global_unique_standardized": len(global_unique[direction]),
            }
            for direction in args.directions
        },
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


def retained_results(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a path- and timing-independent result suitable for versioning."""
    dataset = reports[0]["dataset"] if reports else {}
    entries = []
    for report in reports:
        representation = str(report["representation"])
        entries.append(
            {
                "name": "llg" if representation == "tuple" else representation,
                "format": representation,
                "directions": report["directions"],
                "selection": report["selection"],
                "policy": report["policy"],
                "counts": report["counts"],
                "output_population": report.get("output_population", {}),
            }
        )
    return {
        "schema": "synkit.bidirectional-rule-replay-results/1",
        "dataset_sha256": dataset.get("sha256"),
        "representations": entries,
    }


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
    write_json(args.output_dir / "results.json", retained_results(reports))
    for report in reports:
        print(
            report["representation"],
            json.dumps(report["counts"], sort_keys=True),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
