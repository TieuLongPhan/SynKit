#!/usr/bin/env python3
"""Exhaustively canonicalize every local ordering of detected RotA axes.

RotA supplies molecular structures and positive locus annotations, but this
experiment deliberately does not use those annotations or any handedness
label. For each detected axis support it enumerates every admissible raw local
ordering and canonicalizes the fixed molecular graph. Every representation in
one formal local class must collapse to one certificate; whole-molecule
symmetry may further quotient the formal classes. Only one focal carrier is
varied at a time, while all other extracted configurations remain fixed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Canonicalization.local_permutations import (  # noqa: E402
    all_local_arrangements,
)
from Experiment.Stereo.Canonicalization.run import _canonicalize  # noqa: E402
from Experiment.Stereo.datasets import ROTA, ROTA_SHA256, load_rota  # noqa: E402
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    PotentialStereoElement,
    StereoElementType,
    detect_potential_stereo_elements,
)
from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    CumuleneAxisStereo,
    descriptor_id,
)
from synkit.Graph.Stereo.canonical import (  # noqa: E402
    _rdkit_graph_and_registry,
)
from synkit.Graph.Stereo.enumeration import (  # noqa: E402
    _rdkit_identifier_map,
    _translate_reference,
)
from synkit.Graph.Stereo.supports import AxisStereoSupport  # noqa: E402

_AXIS_TYPES = frozenset(
    {
        StereoElementType.ATROP_AXIS,
        StereoElementType.CUMULENE_AXIS,
    }
)
_TABLE_FIELDS = (
    "record_id",
    "carrier_id",
    "family",
    "carrier_status",
    "raw_local_orderings_expected",
    "raw_local_orderings_checked",
    "canonicalizations_completed",
    "formal_configuration_classes",
    "canonical_classes_observed",
    "global_symmetry_quotient",
    "passed",
    "seconds",
    "issues",
)


def _graph_support(
    support: AxisStereoSupport,
    identifiers: dict[int, int],
) -> AxisStereoSupport:
    return AxisStereoSupport(
        tuple(identifiers[atom] for atom in support.path),
        tuple(
            tuple(_translate_reference(reference, identifiers) for reference in frame)
            for frame in support.terminal_frames
        ),  # type: ignore[arg-type]
    )


def _configured_seed(
    element: PotentialStereoElement,
    identifiers: dict[int, int],
) -> AtropBondStereo | CumuleneAxisStereo:
    if not isinstance(element.support, AxisStereoSupport):
        raise TypeError("RotA local canonicalization requires an axis support.")
    support = _graph_support(element.support, identifiers)
    provenance = "rota_configuration_free_local_enumeration"
    if element.element_type is StereoElementType.ATROP_AXIS:
        if len(support.path) != 2:
            raise ValueError("Atrop local enumeration requires a two-atom axis.")
        left, right = support.terminal_frames
        return AtropBondStereo(
            (*left, support.path[0], support.path[-1], *right),
            1,
            provenance,
        )
    if element.element_type is StereoElementType.CUMULENE_AXIS:
        return CumuleneAxisStereo(
            support.path,
            support.terminal_frames,
            1,
            provenance,
        )
    raise TypeError(f"Unsupported RotA carrier type: {element.element_type.value}.")


def _focal_context(
    registry: dict[str, Any],
    seed: AtropBondStereo | CumuleneAxisStereo,
) -> dict[str, Any]:
    """Remove any supplied configuration attached to the focal support."""
    return {
        identifier: descriptor
        for identifier, descriptor in registry.items()
        if getattr(descriptor, "support", None) != seed.support
    }


def _carrier_result(
    *,
    graph: Any,
    registry: dict[str, Any],
    seed: AtropBondStereo | CumuleneAxisStereo,
    record_id: str,
    carrier_id: str,
    carrier_status: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    arrangements = all_local_arrangements(seed)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for descriptor in arrangements:
        grouped[repr(descriptor.canonical_form())].append(descriptor)

    context = _focal_context(registry, seed)
    focal_id = descriptor_id(seed)
    class_results = []
    all_digests = []
    all_issues = []
    checked = 0
    for configuration_index, key in enumerate(sorted(grouped)):
        digests = []
        issues = []
        for descriptor in grouped[key]:
            variant = dict(context)
            variant[focal_id] = descriptor
            result, issue, _duration = _canonicalize(
                graph,
                variant,
                timeout_seconds=timeout_seconds,
                enumerate_automorphism_group=False,
            )
            checked += 1
            if issue is not None or result is None:
                issues.append(issue or "missing_result")
            else:
                digests.append(result.canonical_digest)
        all_digests.extend(digests)
        all_issues.extend(issues)
        class_results.append(
            {
                "configuration_index": configuration_index,
                "raw_local_orderings": len(grouped[key]),
                "canonicalizations_completed": len(digests),
                "canonical_digests": sorted(set(digests)),
                "issues": issues,
                "collapsed_within_class": len(set(digests)) == 1 and not issues,
            }
        )

    formal_classes = len(grouped)
    observed_classes = len(set(all_digests))
    within_class_passed = all(
        result["collapsed_within_class"] for result in class_results
    )
    complete = checked == len(arrangements) and len(all_digests) == len(arrangements)
    passed = complete and within_class_passed and 0 < observed_classes <= formal_classes
    issues = list(all_issues)
    if observed_classes > formal_classes:
        issues.append("representation_invariance_failure")
    return {
        "record_id": record_id,
        "carrier_id": carrier_id,
        "family": seed.descriptor_class,
        "carrier_status": carrier_status,
        "raw_local_orderings_expected": len(arrangements),
        "raw_local_orderings_checked": checked,
        "canonicalizations_completed": len(all_digests),
        "formal_configuration_classes": formal_classes,
        "canonical_classes_observed": observed_classes,
        "global_symmetry_quotient": (
            complete and within_class_passed and 0 < observed_classes < formal_classes
        ),
        "passed": passed,
        "seconds": time.perf_counter() - started,
        "issues": issues,
        "configuration_results": class_results,
    }


def _record_result(
    item: tuple[int, dict[str, str], float],
) -> dict[str, Any]:
    index, row, timeout_seconds = item
    RDLogger.DisableLog("rdApp.*")
    record_id = f"RotA-{index:04d}"
    molecule = Chem.MolFromSmiles(row["SMILES"])
    if molecule is None:
        return {
            "id": record_id,
            "parse_failure": True,
            "status": "parse_failure",
            "carriers": [],
        }
    graph, extracted = _rdkit_graph_and_registry(molecule)
    registry = dict(extracted)
    identifiers = _rdkit_identifier_map(molecule)
    elements = tuple(
        element
        for element in detect_potential_stereo_elements(molecule)
        if element.element_type in _AXIS_TYPES
    )
    unique: dict[tuple[str, str], PotentialStereoElement] = {}
    for element in elements:
        unique[(element.element_type.value, repr(element.support))] = element

    carriers = []
    for carrier_index, element in enumerate(unique[key] for key in sorted(unique)):
        seed = _configured_seed(element, identifiers)
        carriers.append(
            _carrier_result(
                graph=graph,
                registry=registry,
                seed=seed,
                record_id=record_id,
                carrier_id=f"{record_id}:axis:{carrier_index}",
                carrier_status=element.carrier_status.value,
                timeout_seconds=timeout_seconds,
            )
        )
    return {
        "id": record_id,
        "parse_failure": False,
        "status": (
            "no_axis_support"
            if not carriers
            else (
                "passed" if all(carrier["passed"] for carrier in carriers) else "failed"
            )
        ),
        "extracted_context_configurations": len(registry),
        "carriers": carriers,
    }


def _run_records(
    items: list[tuple[int, dict[str, str], float]],
    *,
    jobs: int,
) -> tuple[dict[str, Any], ...]:
    if jobs == 1:
        return tuple(_record_result(item) for item in items)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return tuple(executor.map(_record_result, items))


def benchmark_rota_local_canonicalization(
    *,
    path: Path = ROTA,
    record_ids: tuple[str, ...] = (),
    limit: int | None = None,
    jobs: int = 1,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    rows = load_rota(path)
    selected = [
        (index, row, timeout_seconds)
        for index, row in enumerate(rows)
        if not record_ids or f"RotA-{index:04d}" in record_ids
    ]
    if limit is not None:
        selected = selected[:limit]
    started = time.perf_counter()
    records = list(_run_records(selected, jobs=jobs))
    carriers = [carrier for record in records for carrier in record["carriers"]]
    checked = sum(carrier["raw_local_orderings_checked"] for carrier in carriers)
    completed = sum(carrier["canonicalizations_completed"] for carrier in carriers)
    expected = sum(carrier["raw_local_orderings_expected"] for carrier in carriers)
    return {
        "schema": "synkit.rota-local-canonicalization/1",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "dataset": {
            "path": str(path),
            "audited_sha256": ROTA_SHA256,
            "records_available": len(rows),
            "records_selected": len(selected),
            "source_annotations_used": False,
        },
        "protocol": {
            "graph_fixed": True,
            "one_focal_carrier_at_a_time": True,
            "raw_local_orderings": "exhaustive",
            "configuration_labels": "unnamed_formal_orbits",
            "other_extracted_configurations": "fixed",
            "timeout_seconds_per_canonicalization": timeout_seconds,
            "worker_processes": jobs,
            "record_ids": list(record_ids),
            "record_limit": limit,
        },
        "summary": {
            "records_evaluated": len(records),
            "parse_failures": sum(record["parse_failure"] for record in records),
            "records_with_axis_support": sum(
                bool(record["carriers"]) for record in records
            ),
            "record_statuses": dict(
                sorted(Counter(record["status"] for record in records).items())
            ),
            "detected_axis_carriers": len(carriers),
            "families": dict(
                sorted(Counter(carrier["family"] for carrier in carriers).items())
            ),
            "carrier_statuses": dict(
                sorted(
                    Counter(carrier["carrier_status"] for carrier in carriers).items()
                )
            ),
            "carriers_passed": sum(carrier["passed"] for carrier in carriers),
            "carriers_failed": sum(not carrier["passed"] for carrier in carriers),
            "global_symmetry_quotients": sum(
                carrier["global_symmetry_quotient"] for carrier in carriers
            ),
            "raw_local_orderings_expected": expected,
            "raw_local_orderings_checked": checked,
            "canonicalizations_completed": completed,
            "complete": completed == expected,
            "strict_carrier_accuracy": (
                sum(carrier["passed"] for carrier in carriers) / len(carriers)
                if carriers
                else None
            ),
        },
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is configuration-free local representation canonicalization "
            "over detected atrop and cumulene supports. RotA annotations, "
            "handedness, stability, and global chirality are not inputs or "
            "accuracy references."
        ),
    }


def _write_table(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_FIELDS)
        writer.writeheader()
        for record in report["records"]:
            for carrier in record["carriers"]:
                writer.writerow(
                    {
                        field: (
                            ";".join(carrier[field])
                            if field == "issues"
                            else (
                                str(carrier[field]).lower()
                                if isinstance(carrier[field], bool)
                                else carrier[field]
                            )
                        )
                        for field in _TABLE_FIELDS
                    }
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rota-path", type=Path, default=ROTA)
    parser.add_argument("--record-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be at least one")
    if arguments.jobs < 1:
        parser.error("--jobs must be at least one")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    report = benchmark_rota_local_canonicalization(
        path=arguments.rota_path,
        record_ids=tuple(arguments.record_id or ()),
        limit=arguments.limit,
        jobs=arguments.jobs,
        timeout_seconds=arguments.timeout,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(arguments.table, report)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "table": str(arguments.table),
                "summary": report["summary"],
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
