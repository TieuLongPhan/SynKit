#!/usr/bin/env python3
"""Replay the Lewis benchmark with the historical SynKit 1.0 MØD path.

This is an efficiency and endpoint-recovery baseline for the maintained LLG
runner in :mod:`Experiment.Lewis.rule_replay.benchmark`.  MØD consumes the
legacy atom/bond GML projection of each reaction; it does not receive the full
Lewis-state labels carried by the LLG tuple representation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib
import json
import logging
from pathlib import Path
import signal
import sys
import time
from types import ModuleType
from typing import Any

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
from synkit.Chem.Molecule.standardize import (  # noqa: E402
    sanitize_and_canonicalize_smiles,
)
from synkit.IO.chem_converter import smart_to_gml  # noqa: E402

DIRECTIONS = ("forward", "backward")
STRATEGIES = ("bt", "comp", "all")
RESULTS_ROOT = HERE / "Data"
HISTORICAL_SYNKIT_TAG = "v1.0.0"
HISTORICAL_ADAPTER_BLOB = "cc153040879f987bfb7bc431b187b651fa150616"


class CaseTimeout(BaseException):
    """Cancel a replay stage that exceeds its diagnostic wall-time ceiling."""


def _raise_timeout(_signum, _frame) -> None:
    raise CaseTimeout("MØD replay stage exceeded the case timeout")


def _supports_interval_timer() -> bool:
    return all(
        hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")
    )


def _set_timeout(seconds: float | None) -> None:
    if seconds is not None and _supports_interval_timer():
        signal.setitimer(signal.ITIMER_REAL, seconds)


def require_mod() -> ModuleType:
    """Import the optional PyMØD module with an actionable error."""
    try:
        return importlib.import_module("mod")
    except ImportError as exc:
        raise RuntimeError(
            "PyMØD is required for this baseline. Install it on Linux with "
            "`conda install -c jakobandersen -c conda-forge mod`."
        ) from exc


def mod_version(mod_module: ModuleType) -> str:
    """Return the most precise version exposed by the loaded MØD build."""
    version = getattr(mod_module, "version", None)
    if callable(version):
        return str(version())
    return str(getattr(mod_module, "__version__", "unknown"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=POLAR_DATASET)
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=DIRECTIONS,
        default=list(DIRECTIONS),
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="bt",
        help=(
            "Historical MODReactor strategy: bt tries proper component replay "
            "then relaxed replay; comp requires every host component; all "
            "permits unused components (default: bt)."
        ),
    )
    parser.add_argument("--record-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-timeout",
        type=float,
        help=(
            "Optional best-effort per-stage timeout in seconds. Native MØD "
            "calls may not be interruptible on every build."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "mod-replay",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {"record_id": int(row["R-id"]), "reaction": str(row["smart"])}
        for row in read_json(path)
    ]


def extract_gml_rule(reaction: str) -> str:
    """Project a mapped reaction onto the SynKit 1.0 atom/bond GML rule."""
    return smart_to_gml(
        reaction,
        core=True,
        sanitize=True,
        explicit_hydrogen=False,
        useSmiles=True,
    )


def _reuse_isomorphic_graphs(graphs: list[Any]) -> list[Any]:
    """Preserve multiplicity while reusing MØD objects for isomorphic inputs.

    This reproduces ``_deduplicateGraphs`` from SynKit 1.0.  Reusing the same
    object is required because a MØD derivation graph rejects distinct but
    isomorphic objects in its graph database.
    """
    prepared: list[Any] = []
    for candidate in graphs:
        for existing in prepared:
            if candidate.isomorphism(existing) != 0:
                prepared.append(existing)
                break
        else:
            prepared.append(candidate)
    return prepared


def _prepare_molecules(mod_module: ModuleType, host: str) -> list[Any]:
    molecules = [
        mod_module.smiles(component, add=False) for component in host.split(".")
    ]
    molecules = _reuse_isomorphic_graphs(molecules)
    molecules.sort(key=lambda molecule: getattr(molecule, "numVertices", 0))
    return molecules


def _apply_once(
    mod_module: ModuleType,
    molecules: list[Any],
    rule: Any,
    *,
    only_proper: bool,
) -> list[Any]:
    dg = mod_module.DG(graphDatabase=molecules)
    mod_module.config.dg.doRuleIsomorphismDuringBinding = False
    dg.build().apply(
        molecules,
        rule,
        onlyProper=only_proper,
        verbosity=0,
    )
    return list(dg.edges)


def _apply_strategy(
    mod_module: ModuleType,
    molecules: list[Any],
    rule: Any,
    strategy: str,
) -> tuple[list[Any], bool, bool, bool]:
    """Apply one historical strategy.

    :return: Edges, whether unused host components must be reattached, whether
        target SMILES must be sanitized, and whether ``bt`` used its relaxed
        fallback.
    """
    if strategy == "comp":
        return (
            _apply_once(mod_module, molecules, rule, only_proper=True),
            False,
            False,
            False,
        )
    if strategy == "all":
        return (
            _apply_once(mod_module, molecules, rule, only_proper=False),
            True,
            False,
            False,
        )
    if strategy != "bt":
        raise ValueError(f"Unsupported MØD strategy: {strategy!r}")

    proper = _apply_once(mod_module, molecules, rule, only_proper=True)
    if proper:
        return proper, False, True, False
    relaxed = _apply_once(mod_module, molecules, rule, only_proper=False)
    return relaxed, True, False, True


def _canonical_component(smiles: str) -> str | None:
    return sanitize_and_canonicalize_smiles(smiles)


def _serialize_edges(
    edges: list[Any],
    initial_components: list[str],
    *,
    reattach_unused: bool,
    sanitize_targets: bool,
) -> list[list[str | None]]:
    base = (
        Counter(_canonical_component(item) for item in initial_components)
        if reattach_unused
        else Counter()
    )
    batches: list[list[str | None]] = []
    for edge in edges:
        targets: list[str | None] = [vertex.graph.smiles for vertex in edge.targets]
        if sanitize_targets:
            targets = [_canonical_component(item) for item in targets if item]
        if reattach_unused:
            used = Counter(
                _canonical_component(vertex.graph.smiles) for vertex in edge.sources
            )
            targets.extend((base - used).elements())
        batches.append(targets)
    return batches


def _reaction_strings(
    batches: list[list[str | None]],
    host: str,
    direction: str,
) -> list[str]:
    reactions: list[str] = []
    for batch in batches:
        if any(item is None for item in batch):
            continue
        endpoint = ".".join(str(item) for item in batch)
        if direction == "backward":
            reactions.append(f"{endpoint}>>{host}")
        else:
            reactions.append(f"{host}>>{endpoint}")
    return reactions


def replay_direction(
    *,
    mod_module: ModuleType,
    host: str,
    expected: str,
    gml_rule: str,
    direction: str,
    strategy: str,
    case_timeout: float | None,
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
    edges: list[Any] | None = None
    batches: list[list[str | None]] | None = None
    reactions: list[str] | None = None
    try:
        initial_components = host.split(".")
        molecules = _prepare_molecules(mod_module, host)
        rule = mod_module.ruleGMLString(
            gml_rule,
            invert=direction == "backward",
            add=False,
        )
        stage_seconds[stage] = time.perf_counter() - stage_started

        stage = "matching_rewriting"
        stage_started = time.perf_counter()
        _set_timeout(case_timeout)
        edges, reattach_unused, sanitize_targets, relaxed_fallback = _apply_strategy(
            mod_module, molecules, rule, strategy
        )
        stage_seconds[stage] = time.perf_counter() - stage_started

        stage = "serialization"
        stage_started = time.perf_counter()
        _set_timeout(case_timeout)
        batches = _serialize_edges(
            edges,
            initial_components,
            reattach_unused=reattach_unused,
            sanitize_targets=sanitize_targets,
        )
        reactions = _reaction_strings(batches, host, direction)
        stage_seconds[stage] = time.perf_counter() - stage_started

        stage = "canonicalization"
        stage_started = time.perf_counter()
        _set_timeout(0.0)
        generated = unique_standardized_reactions(reactions)
        if standardized_reaction_sink is not None:
            standardized_reaction_sink.update(generated)
        stage_seconds[stage] = time.perf_counter() - stage_started
        recovered = expected in generated
        result: dict[str, Any] = {
            "status": "PASS" if recovered else "FAIL",
            "reference_recovered": recovered,
            "derivation_count": len(edges),
            "serialized_count": len(reactions),
            "unique_reaction_count": len(generated),
            "unique_standardized_reaction_count": len(generated),
            "duplicate_reaction_count": len(reactions) - len(generated),
            "relaxed_fallback": relaxed_fallback,
            "seconds": time.perf_counter() - started,
            "expansion_seconds": sum(
                stage_seconds[name]
                for name in ("reactor_construction", "matching_rewriting")
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
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "seconds": time.perf_counter() - started,
            "stage_seconds": stage_seconds,
        }
        if edges is not None:
            result["derivation_count"] = len(edges)
        if batches is not None:
            result["batch_count"] = len(batches)
        if reactions is not None:
            result["serialized_count"] = len(reactions)
        return result
    finally:
        _set_timeout(0.0)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)


def benchmark(
    mod_module: ModuleType,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    global_unique: dict[str, set[str]] = {
        direction: set() for direction in args.directions
    }
    output = args.output_dir / "mod-cases.jsonl.gz"
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
                gml_rule = extract_gml_rule(reaction)
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
                    mod_module=mod_module,
                    host=hosts[direction],
                    expected=expected,
                    gml_rule=gml_rule,
                    direction=direction,
                    strategy=args.strategy,
                    case_timeout=args.case_timeout,
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
                if result.get("relaxed_fallback"):
                    counts[f"{direction}:relaxed_fallback"] += 1
            case["status"] = (
                "PASS"
                if all(item["status"] == "PASS" for item in case["directions"].values())
                else "INCOMPLETE"
            )
            handle.write(json.dumps(case, sort_keys=True) + "\n")
            if args.progress_every and index % args.progress_every == 0:
                print(f"mod: {index}/{len(rows)}", flush=True)

    report = {
        "schema": "synkit.mod-bidirectional-rule-replay/1",
        "engine": {
            "name": "MØD",
            "version": mod_version(mod_module),
            "historical_synkit_tag": HISTORICAL_SYNKIT_TAG,
            "historical_adapter_blob": HISTORICAL_ADAPTER_BLOB,
        },
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
            "strategy": args.strategy,
            "case_timeout_seconds": args.case_timeout,
            "timeout_scope": "each stage; native calls are best-effort",
            "rule_projection": "legacy atom/bond GML reaction center",
            "lewis_state_labels": False,
            "rule_isomorphism_during_binding": False,
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
    write_json(args.output_dir / "mod-summary.json", report)
    return report


def retained_results(report: dict[str, Any]) -> dict[str, Any]:
    """Return path- and timing-independent evidence suitable for versioning."""
    return {
        "schema": "synkit.mod-bidirectional-rule-replay-results/1",
        "dataset_sha256": report["dataset"]["sha256"],
        "engine": report["engine"],
        "directions": report["directions"],
        "selection": report["selection"],
        "policy": report["policy"],
        "counts": report["counts"],
        "output_population": report.get("output_population", {}),
    }


def main() -> int:
    args = parse_args()
    if args.case_timeout is not None and args.case_timeout <= 0:
        raise ValueError("Timeout must be positive")
    mod_module = require_mod()
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
    report = benchmark(mod_module, rows, args)
    write_json(args.output_dir / "results.json", retained_results(report))
    print("mod", json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
