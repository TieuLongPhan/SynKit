#!/usr/bin/env python3
"""Secondary whole-graph atom-relabeling stress tests.

.. rubric:: Examples

Run all ten internal families::

    python Experiment/Stereo/Canonicalization/run.py internal

Run ACS configured records with 100 deterministic relabelings per large case::

    python Experiment/Stereo/Canonicalization/atom_relabel.py         acs --permutations 100

Run RotA support invariance (not configured handedness)::

    python Experiment/Stereo/Canonicalization/atom_relabel.py         rota --permutations 100

Run the external CIP suite after supplying its exact audited source file::

    python Experiment/Stereo/Canonicalization/atom_relabel.py cip         --cip-path /path/to/compounds.smi --permutations 100
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Perception.axis_loci import (  # noqa: E402
    _axis_predictions,
)
from Experiment.Stereo.datasets import (  # noqa: E402
    ROTA,
    load_cip,
    load_rota,
)
from Experiment.Stereo.Canonicalization.global_local import (  # noqa: E402
    _case_time_limit,
    benchmark_global_local_canonicalization,
)
from Experiment.Stereo.Canonicalization.inventory import (  # noqa: E402
    DEFAULT_CSV,
    DEFAULT_JSON,
    build_inventory,
    write_inventory,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    DATASET as ACS_DATASET,
    load_dataset,
)
from synkit.Graph.Stereo import (  # noqa: E402
    canonicalize_rdkit_configured_stereograph,
    descriptors_from_rdkit,
)

CANON_DATA_ROOT = ROOT / "Experiment" / "Stereo" / "Data" / "Canonicalization"
_TASKS = ("global-local", "internal", "acs", "rota", "cip")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(seed: int, record_id: str) -> int:
    payload = f"{seed}:{record_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sampled_orders(
    size: int,
    *,
    samples: int,
    seed: int,
    record_id: str,
) -> tuple[tuple[int, ...], ...]:
    if samples < 1:
        raise ValueError("--permutations must be at least one.")
    identity = tuple(range(size))
    target = min(samples, math.factorial(size))
    orders = {identity}
    reversed_order = tuple(reversed(identity))
    if len(orders) < target:
        orders.add(reversed_order)
    generator = random.Random(_stable_seed(seed, record_id))
    while len(orders) < target:
        order = list(identity)
        generator.shuffle(order)
        orders.add(tuple(order))
    return tuple(sorted(orders))


def _permutation_orders(
    size: int,
    *,
    samples: int,
    exhaustive_max_atoms: int,
    force_exhaustive: bool,
    seed: int,
    record_id: str,
) -> tuple[str, int, Iterable[tuple[int, ...]]]:
    if force_exhaustive or size <= exhaustive_max_atoms:
        planned = math.factorial(size)
        return "exhaustive", planned, permutations(range(size))
    orders = _sampled_orders(
        size,
        samples=samples,
        seed=seed,
        record_id=record_id,
    )
    return "fixed_seed_sample", len(orders), orders


def _selected(
    records: list[dict[str, Any]],
    *,
    record_ids: tuple[str, ...],
    limit: int | None,
) -> list[dict[str, Any]]:
    if record_ids:
        by_id = {record["record_id"]: record for record in records}
        missing = sorted(set(record_ids) - by_id.keys())
        if missing:
            raise ValueError(f"Unknown record identifiers: {missing}")
        selected = [by_id[record_id] for record_id in record_ids]
    else:
        selected = list(records)
    return selected[:limit] if limit is not None else selected


def _configured_record_audit(
    molecule: Chem.Mol,
    *,
    record_id: str,
    samples: int,
    exhaustive_max_atoms: int,
    force_exhaustive: bool,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    registry = descriptors_from_rdkit(
        molecule,
        require_atom_maps=False,
    )
    family_counts = Counter(
        descriptor.descriptor_class for descriptor in registry.values()
    )
    if not registry:
        return {
            "id": record_id,
            "atoms": molecule.GetNumAtoms(),
            "configured_elements": 0,
            "family_counts": {},
            "status": "no_extractable_configured_stereo",
            "permutations_planned": 0,
            "permutations_passed": 0,
            "permutation_mismatches": 0,
            "timeouts": 0,
            "errors": 0,
            "seconds": time.perf_counter() - started,
        }
    try:
        with _case_time_limit(timeout_seconds):
            baseline = canonicalize_rdkit_configured_stereograph(molecule)
    except TimeoutError:
        return {
            "id": record_id,
            "atoms": molecule.GetNumAtoms(),
            "configured_elements": len(registry),
            "family_counts": dict(sorted(family_counts.items())),
            "status": "baseline_timeout",
            "permutations_planned": 0,
            "permutations_passed": 0,
            "permutation_mismatches": 0,
            "timeouts": 1,
            "errors": 0,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return {
            "id": record_id,
            "atoms": molecule.GetNumAtoms(),
            "configured_elements": len(registry),
            "family_counts": dict(sorted(family_counts.items())),
            "status": "baseline_error",
            "error": f"{type(error).__name__}: {error}",
            "permutations_planned": 0,
            "permutations_passed": 0,
            "permutation_mismatches": 0,
            "timeouts": 0,
            "errors": 1,
            "seconds": time.perf_counter() - started,
        }

    protocol, planned, orders = _permutation_orders(
        molecule.GetNumAtoms(),
        samples=samples,
        exhaustive_max_atoms=exhaustive_max_atoms,
        force_exhaustive=force_exhaustive,
        seed=seed,
        record_id=record_id,
    )
    passed = 0
    mismatches = 0
    timeouts = 0
    errors = 0
    mismatch_examples = []
    error_examples = []
    for order in orders:
        try:
            renumbered = Chem.RenumberAtoms(molecule, order)
            with _case_time_limit(timeout_seconds):
                result = canonicalize_rdkit_configured_stereograph(renumbered)
            if baseline.same_stereograph(result):
                passed += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(list(order))
        except TimeoutError:
            timeouts += 1
        except Exception as error:  # pragma: no cover - diagnostic boundary
            errors += 1
            if len(error_examples) < 5:
                error_examples.append(
                    {
                        "order": list(order),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    complete = passed + mismatches + timeouts + errors == planned
    fully_invariant = complete and passed == planned
    return {
        "id": record_id,
        "atoms": molecule.GetNumAtoms(),
        "configured_elements": len(registry),
        "family_counts": dict(sorted(family_counts.items())),
        "configured_certificate_digest": baseline.canonical_digest,
        "permutation_protocol": protocol,
        "permutations_planned": planned,
        "permutations_passed": passed,
        "permutation_mismatches": mismatches,
        "timeouts": timeouts,
        "errors": errors,
        "mismatch_examples": mismatch_examples,
        "error_examples": error_examples,
        "status": "passed" if fully_invariant else "incomplete_or_failed",
        "fully_invariant": fully_invariant,
        "seconds": time.perf_counter() - started,
    }


def _invariance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    planned = sum(record["permutations_planned"] for record in records)
    passed = sum(record["permutations_passed"] for record in records)
    mismatches = sum(record["permutation_mismatches"] for record in records)
    timeouts = sum(record["timeouts"] for record in records)
    errors = sum(record["errors"] for record in records)
    definitive = passed + mismatches
    return {
        "records_selected": len(records),
        "fully_invariant_records": sum(
            record.get("fully_invariant", False) for record in records
        ),
        "permutations_planned": planned,
        "permutations_passed": passed,
        "permutation_mismatches": mismatches,
        "timeouts": timeouts,
        "errors": errors,
        "definitive_invariance_accuracy": _ratio(passed, definitive),
        "strict_invariance_accuracy": _ratio(passed, planned),
        "completion_rate": _ratio(definitive, planned),
    }


def _configured_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _invariance_summary(records)
    summary["records_with_extractable_configured_stereo"] = sum(
        record["configured_elements"] > 0 for record in records
    )
    return summary


def _run_acs(
    inventory: dict[str, Any],
    *,
    path: Path,
    record_ids: tuple[str, ...],
    limit: int | None,
    max_atoms: int | None,
    samples: int,
    exhaustive_max_atoms: int,
    force_exhaustive: bool,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    candidates = _selected(
        [
            record
            for record in inventory["records"]
            if record["source"] == "acs_stereomolgraph"
        ],
        record_ids=record_ids,
        limit=limit,
    )
    source = {row["ID"]: row for row in load_dataset(path)}
    records = []
    excluded_by_size = []
    started = time.perf_counter()
    for candidate in candidates:
        identifier = candidate["record_id"]
        molecule = Chem.MolFromSmiles(source[identifier]["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected ACS case {identifier}.")
        if max_atoms is not None and molecule.GetNumAtoms() > max_atoms:
            excluded_by_size.append(identifier)
            continue
        records.append(
            _configured_record_audit(
                molecule,
                record_id=identifier,
                samples=samples,
                exhaustive_max_atoms=exhaustive_max_atoms,
                force_exhaustive=force_exhaustive,
                seed=seed,
                timeout_seconds=timeout_seconds,
            )
        )
    summary = _configured_summary(records)
    summary["records_excluded_by_max_atoms"] = len(excluded_by_size)
    return {
        "task": "acs",
        "canonicalization_scope": "configured_stereograph",
        "summary": summary,
        "records_excluded_by_max_atoms": excluded_by_size,
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "ACS manual chiral/achiral labels are not used. Accuracy measures "
            "only certificate invariance for the same supplied configuration "
            "under atom relabeling."
        ),
    }


def _axis_set(molecule: Chem.Mol) -> set[tuple[str, tuple[int, ...]]]:
    return {(item["type"], tuple(item["path"])) for item in _axis_predictions(molecule)}


def _transport_axis_set(
    axes: set[tuple[str, tuple[int, ...]]],
    old_to_new: dict[int, int],
) -> set[tuple[str, tuple[int, ...]]]:
    transported = set()
    for axis_type, path in axes:
        mapped = tuple(old_to_new[index] for index in path)
        transported.add((axis_type, min(mapped, tuple(reversed(mapped)))))
    return transported


def _rota_record_audit(
    molecule: Chem.Mol,
    *,
    record_id: str,
    reference_loci: list[list[int]],
    samples: int,
    exhaustive_max_atoms: int,
    force_exhaustive: bool,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    with _case_time_limit(timeout_seconds):
        baseline = _axis_set(molecule)
    predicted_pairs = {
        tuple(sorted((path[0], path[-1]))) for _axis_type, path in baseline
    }
    reference_pairs = {tuple(pair) for pair in reference_loci}
    protocol, planned, orders = _permutation_orders(
        molecule.GetNumAtoms(),
        samples=samples,
        exhaustive_max_atoms=exhaustive_max_atoms,
        force_exhaustive=force_exhaustive,
        seed=seed,
        record_id=record_id,
    )
    passed = 0
    mismatches = 0
    timeouts = 0
    errors = 0
    mismatch_examples = []
    for order in orders:
        old_to_new = {old: new for new, old in enumerate(order)}
        expected = _transport_axis_set(baseline, old_to_new)
        try:
            renumbered = Chem.RenumberAtoms(molecule, order)
            with _case_time_limit(timeout_seconds):
                observed = _axis_set(renumbered)
            if observed == expected:
                passed += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        {
                            "order": list(order),
                            "missing": sorted(expected - observed),
                            "extra": sorted(observed - expected),
                        }
                    )
        except TimeoutError:
            timeouts += 1
        except Exception:  # pragma: no cover - diagnostic boundary
            errors += 1
    fully_invariant = passed == planned
    return {
        "id": record_id,
        "atoms": molecule.GetNumAtoms(),
        "reference_loci": len(reference_pairs),
        "detected_axis_supports": len(baseline),
        "reference_loci_recovered": len(reference_pairs & predicted_pairs),
        "permutation_protocol": protocol,
        "permutations_planned": planned,
        "permutations_passed": passed,
        "permutation_mismatches": mismatches,
        "timeouts": timeouts,
        "errors": errors,
        "mismatch_examples": mismatch_examples,
        "fully_invariant": fully_invariant,
        "status": "passed" if fully_invariant else "incomplete_or_failed",
        "seconds": time.perf_counter() - started,
    }


def _run_rota(
    inventory: dict[str, Any],
    *,
    path: Path,
    record_ids: tuple[str, ...],
    limit: int | None,
    max_atoms: int | None,
    samples: int,
    exhaustive_max_atoms: int,
    force_exhaustive: bool,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    candidates = _selected(
        [
            record
            for record in inventory["records"]
            if record["source"] == "chiralfinder_rota"
        ],
        record_ids=record_ids,
        limit=limit,
    )
    source = {f"RotA-{index:04d}": row for index, row in enumerate(load_rota(path))}
    records = []
    excluded_by_size = []
    started = time.perf_counter()
    for candidate in candidates:
        identifier = candidate["record_id"]
        molecule = Chem.MolFromSmiles(source[identifier]["SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected RotA case {identifier}.")
        if max_atoms is not None and molecule.GetNumAtoms() > max_atoms:
            excluded_by_size.append(identifier)
            continue
        records.append(
            _rota_record_audit(
                molecule,
                record_id=identifier,
                reference_loci=candidate["reference_loci"],
                samples=samples,
                exhaustive_max_atoms=exhaustive_max_atoms,
                force_exhaustive=force_exhaustive,
                seed=seed,
                timeout_seconds=timeout_seconds,
            )
        )
    summary = _invariance_summary(records)
    summary.update(
        {
            "records_excluded_by_max_atoms": len(excluded_by_size),
            "records_with_detected_axis_support": sum(
                record["detected_axis_supports"] > 0 for record in records
            ),
            "reference_loci": sum(record["reference_loci"] for record in records),
            "reference_loci_recovered": sum(
                record["reference_loci_recovered"] for record in records
            ),
            "source_locus_recall": _ratio(
                sum(record["reference_loci_recovered"] for record in records),
                sum(record["reference_loci"] for record in records),
            ),
        }
    )
    return {
        "task": "rota",
        "canonicalization_scope": "axis_support_only",
        "summary": summary,
        "records_excluded_by_max_atoms": excluded_by_size,
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "RotA accuracy combines source-locus recall with atom-relabeling "
            "invariance of detected axis supports. It does not test a "
            "configured axial certificate because handedness is absent."
        ),
    }


def _run_cip(
    inventory: dict[str, Any],
    *,
    path: Path,
    record_ids: tuple[str, ...],
    limit: int | None,
    max_atoms: int | None,
    samples: int,
    exhaustive_max_atoms: int,
    force_exhaustive: bool,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    candidates = _selected(
        [
            record
            for record in inventory["records"]
            if record["source"] == "cip_validation_suite"
        ],
        record_ids=record_ids,
        limit=limit,
    )
    source = {row["ID"]: row for row in load_cip(path)}
    records = []
    excluded_by_size = []
    started = time.perf_counter()
    for candidate in candidates:
        identifier = candidate["record_id"]
        molecule = Chem.MolFromSmiles(source[identifier]["SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected CIP case {identifier}.")
        if max_atoms is not None and molecule.GetNumAtoms() > max_atoms:
            excluded_by_size.append(identifier)
            continue
        result = _configured_record_audit(
            molecule,
            record_id=identifier,
            samples=samples,
            exhaustive_max_atoms=exhaustive_max_atoms,
            force_exhaustive=force_exhaustive,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        result["source_unit_tags"] = candidate["source_categories"]
        records.append(result)
    summary = _configured_summary(records)
    summary["records_excluded_by_max_atoms"] = len(excluded_by_size)
    summary["configured_extraction_coverage"] = _ratio(
        summary["records_with_extractable_configured_stereo"],
        summary["records_selected"],
    )
    return {
        "task": "cip",
        "canonicalization_scope": "configured_local_stereo_elements",
        "summary": summary,
        "records_excluded_by_max_atoms": excluded_by_size,
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "CIP recommended labels establish local stereo-unit provenance, "
            "not global molecular chirality. Invariance is scored only where "
            "a configured descriptor can be extracted from the audited "
            "external structure."
        ),
    }


def _run_global_local(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report = benchmark_global_local_canonicalization(
        arguments.acs_path,
        family_names=tuple(arguments.family or ()),
        class_relabelings=arguments.class_relabelings,
        representative_samples=arguments.permutations,
        exhaustive_atom_limit=arguments.exhaustive_max_atoms,
        exhaustive_all_classes=arguments.exhaustive_atom_relabelings,
        timeout_seconds=arguments.timeout,
        include_public=False,
    )
    totals = report["totals"]
    families_passed = totals["families"] - totals["family_failures"]
    return {
        "task": "global_local",
        "canonicalization_scope": "ten_configured_stereo_families",
        "summary": {
            "families": totals["families"],
            "families_passed": families_passed,
            "family_accuracy": _ratio(families_passed, totals["families"]),
            "expected_configuration_classes": totals["expected_configuration_classes"],
            "exact_certificate_classes": totals["exact_certificate_classes"],
            "certificate_class_accuracy": _ratio(
                totals["exact_certificate_classes"],
                totals["expected_configuration_classes"],
            ),
            "raw_local_arrangements_checked": totals["raw_local_arrangements_checked"],
            "class_relabelings_checked": totals["class_relabelings_checked"],
            "representative_relabelings_checked": totals[
                "representative_relabelings_checked"
            ],
            "timeouts": totals["synthetic_timeouts"],
        },
        "internal_report": report,
        "seconds": time.perf_counter() - started,
    }


def _result_document(
    task_result: dict[str, Any],
    *,
    arguments: argparse.Namespace,
    inventory_json: Path,
    inventory_csv: Path,
) -> dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    return {
        "schema": "synkit.stereo-canonicalization-run/1",
        "task": task_result["task"],
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "inputs": {
            "inventory_json": display_path(inventory_json),
            "inventory_json_sha256": _sha256(inventory_json),
            "inventory_csv": display_path(inventory_csv),
            "inventory_csv_sha256": _sha256(inventory_csv),
        },
        "protocol": {
            "permutations_for_nonexhaustive_records": arguments.permutations,
            "exhaustive_max_atoms": arguments.exhaustive_max_atoms,
            "force_exhaustive_atom_relabeling": (arguments.exhaustive_atom_relabelings),
            "seed": arguments.seed,
            "timeout_seconds_per_canonicalization": arguments.timeout,
            "record_ids": list(arguments.record_id or ()),
            "record_limit": arguments.limit,
            "maximum_selected_atom_count": arguments.max_atoms,
        },
        **task_result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=_TASKS)
    parser.add_argument("--permutations", type=int, default=32)
    parser.add_argument("--exhaustive-max-atoms", type=int, default=6)
    parser.add_argument(
        "--exhaustive-atom-relabelings",
        dest="exhaustive_atom_relabelings",
        action="store_true",
        help=(
            "Secondary stress test: enumerate all n! whole-graph atom "
            "relabelings for every selected record."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--record-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-atoms",
        type=int,
        help=(
            "Exclude public records larger than this before permutation; "
            "use with --exhaustive for a feasible exact tier."
        ),
    )
    parser.add_argument("--family", action="append")
    parser.add_argument("--class-relabelings", type=int, default=3)
    parser.add_argument("--acs-path", type=Path, default=ACS_DATASET)
    parser.add_argument("--rota-path", type=Path, default=ROTA)
    parser.add_argument("--cip-path", type=Path)
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be at least one.")
    if arguments.max_atoms is not None and arguments.max_atoms < 1:
        parser.error("--max-atoms must be at least one.")
    if arguments.exhaustive_max_atoms < 0:
        parser.error("--exhaustive-max-atoms cannot be negative.")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive.")
    if arguments.task == "cip" and arguments.cip_path is None:
        parser.error("the cip task requires --cip-path")
    if (
        arguments.exhaustive_atom_relabelings
        and arguments.task in {"acs", "rota", "cip"}
        and not arguments.record_id
        and arguments.max_atoms is None
        and arguments.limit is None
    ):
        print(
            "warning: exhaustive enumeration over the complete public "
            "corpus includes factorially impossible large records; use "
            "--max-atoms or --record-id for a finishable exact tier",
            file=sys.stderr,
        )
    RDLogger.DisableLog("rdApp.*")

    inventory = build_inventory()
    write_inventory(
        inventory,
        json_path=arguments.inventory_json,
        csv_path=arguments.inventory_csv,
    )
    if arguments.inventory_only:
        print(json.dumps(inventory["summary"], indent=2, sort_keys=True))
        return 0

    if arguments.task in {"global-local", "internal"}:
        task_result = _run_global_local(arguments)
    elif arguments.task == "acs":
        task_result = _run_acs(
            inventory,
            path=arguments.acs_path,
            record_ids=tuple(arguments.record_id or ()),
            limit=arguments.limit,
            max_atoms=arguments.max_atoms,
            samples=arguments.permutations,
            exhaustive_max_atoms=arguments.exhaustive_max_atoms,
            force_exhaustive=arguments.exhaustive_atom_relabelings,
            seed=arguments.seed,
            timeout_seconds=arguments.timeout,
        )
    elif arguments.task == "rota":
        task_result = _run_rota(
            inventory,
            path=arguments.rota_path,
            record_ids=tuple(arguments.record_id or ()),
            limit=arguments.limit,
            max_atoms=arguments.max_atoms,
            samples=arguments.permutations,
            exhaustive_max_atoms=arguments.exhaustive_max_atoms,
            force_exhaustive=arguments.exhaustive_atom_relabelings,
            seed=arguments.seed,
            timeout_seconds=arguments.timeout,
        )
    else:
        task_result = _run_cip(
            inventory,
            path=arguments.cip_path,
            record_ids=tuple(arguments.record_id or ()),
            limit=arguments.limit,
            max_atoms=arguments.max_atoms,
            samples=arguments.permutations,
            exhaustive_max_atoms=arguments.exhaustive_max_atoms,
            force_exhaustive=arguments.exhaustive_atom_relabelings,
            seed=arguments.seed,
            timeout_seconds=arguments.timeout,
        )

    result = _result_document(
        task_result,
        arguments=arguments,
        inventory_json=arguments.inventory_json,
        inventory_csv=arguments.inventory_csv,
    )
    output = arguments.output or (
        CANON_DATA_ROOT / f"{arguments.task}_canonicalization_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task": arguments.task,
                "output": str(output),
                "seconds": result["seconds"],
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
