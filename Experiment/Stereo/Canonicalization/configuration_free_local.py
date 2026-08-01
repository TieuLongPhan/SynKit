#!/usr/bin/env python3
"""Benchmark exhaustive, configuration-free local stereo canonicalization.

For every perceived carrier, this benchmark removes every supplied stereo
configuration, enumerates every raw local reference ordering for that carrier,
and canonicalizes each ordering on the fixed molecular constitution.  Carriers
are tested independently: this is not a Cartesian product over a molecule's
stereocentres.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
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

from Experiment.Stereo.Canonicalization.global_local import (  # noqa: E402
    _case_time_limit,
    _fixture_catalog,
)
from Experiment.Stereo.Canonicalization.local_permutations import (  # noqa: E402
    all_local_arrangements,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    DATASET as ACS_DATASET,
    EXPECTED_SHA256 as ACS_SHA256,
    load_dataset,
)
from Experiment.Stereo.datasets import (  # noqa: E402
    CIP_SHA256,
    ROTA,
    ROTA_SHA256,
    load_cip,
    load_rota,
)
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    PotentialStereoElement,
    StereoElementType,
    detect_potential_stereo_elements,
)
from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    PlanarBondStereo,
    TetrahedralStereo,
    canonicalize_configured_stereograph,
    descriptor_id,
)
from synkit.Graph.Stereo.canonical import (  # noqa: E402
    _rdkit_graph_and_registry,
)
from synkit.Graph.Stereo.enumeration import (  # noqa: E402
    _double_bond_unknown,
    _fixed_seed as _fixed_descriptor_seed,
    _rdkit_identifier_map,
    _tetrahedral_unknown,
    _translate_reference,
)

_TASKS = ("internal", "acs", "cip", "rota")
_TABLE_FIELDS = (
    "source",
    "record_id",
    "carrier_id",
    "family",
    "carrier_status",
    "raw_local_permutations_expected",
    "raw_local_permutations_checked",
    "canonicalizations_completed",
    "theoretical_configuration_classes",
    "theoretical_class_multiplicities",
    "canonical_classes_observed",
    "representation_invariance_passed",
    "formal_class_separation_passed",
    "global_symmetry_quotient",
    "passed",
    "seconds",
    "issues",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stereo_neutral(molecule: Chem.Mol) -> Chem.Mol:
    """Return the same constitution with all supplied configurations erased."""
    neutral = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(neutral)
    for atom in neutral.GetAtoms():
        for property_name in ("_CIPCode", "_CIPRank"):
            if atom.HasProp(property_name):
                atom.ClearProp(property_name)
    return neutral


def _seed_from_potential(
    element: PotentialStereoElement,
    molecule: Chem.Mol,
    identifiers: dict[int, int],
) -> Any:
    """Build an arbitrary fixed representative without reading orientation."""
    provenance = "configuration_free_local_enumeration"
    if element.element_type is StereoElementType.TETRAHEDRAL:
        if element.constitutional_evidence is None:
            raise ValueError("Tetrahedral carrier lacks constitutional evidence.")
        unknown = _tetrahedral_unknown(
            element.constitutional_evidence,
            identifiers,
        )
        return TetrahedralStereo(unknown.atoms, 1, provenance)
    if element.element_type is StereoElementType.DOUBLE_BOND:
        unknown = _double_bond_unknown(molecule, element.support, identifiers)
        return PlanarBondStereo(unknown.atoms, 0, provenance)
    if element.element_type in {
        StereoElementType.CUMULENE_AXIS,
        StereoElementType.EXTENDED_CIS_TRANS,
    }:
        support = element.support
        path = tuple(identifiers[atom] for atom in support.path)
        frames = tuple(
            tuple(_translate_reference(reference, identifiers) for reference in frame)
            for frame in support.terminal_frames
        )
        if element.element_type is StereoElementType.CUMULENE_AXIS:
            return CumuleneAxisStereo(path, frames, 1, provenance)
        return ExtendedCisTransStereo(path, frames, 0, provenance)
    if element.element_type is StereoElementType.ATROP_AXIS:
        support = element.support
        frames = tuple(
            tuple(_translate_reference(reference, identifiers) for reference in frame)
            for frame in support.terminal_frames
        )
        return AtropBondStereo(
            (
                *frames[0],
                identifiers[support.path[0]],
                identifiers[support.path[-1]],
                *frames[1],
            ),
            1,
            provenance,
        )
    if element.element_type is StereoElementType.HELICAL:
        support = element.support
        return HelicalStereo(
            tuple(identifiers[atom] for atom in support.path),
            1,
            provenance,
            support.cyclic,
        )
    raise TypeError(f"Unsupported carrier type: {element.element_type.value}.")


def _canonicalize(
    graph: Any,
    descriptor: Any,
    *,
    timeout_seconds: float,
    internal_colors: bool,
) -> tuple[str | None, str | None, float]:
    started = time.perf_counter()
    options: dict[str, Any] = {"enumerate_automorphism_group": False}
    if internal_colors:
        options.update(atom_color="color", bond_color="color")
    try:
        with _case_time_limit(timeout_seconds):
            result = canonicalize_configured_stereograph(
                graph,
                (descriptor,),
                **options,
            )
    except TimeoutError:
        return None, "timeout", time.perf_counter() - started
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return (
            None,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
        )
    return result.canonical_digest, None, time.perf_counter() - started


def _carrier_result(
    *,
    source: str,
    record_id: str,
    carrier_id: str,
    carrier_status: str,
    graph: Any,
    seed: Any,
    timeout_seconds: float,
    internal_colors: bool = False,
) -> dict[str, Any]:
    """Evaluate every raw ordering and its theoretical local orbit."""
    started = time.perf_counter()
    arrangements = all_local_arrangements(seed)
    theoretical: dict[str, list[Any]] = defaultdict(list)
    for arrangement in arrangements:
        theoretical[repr(arrangement.canonical_form())].append(arrangement)

    class_results = []
    all_digests: list[str] = []
    all_issues: list[str] = []
    checked = 0
    for class_index, class_key in enumerate(sorted(theoretical)):
        descriptors = theoretical[class_key]
        digests: list[str] = []
        issues: list[str] = []
        class_seconds = 0.0
        for descriptor in descriptors:
            digest, issue, duration = _canonicalize(
                graph,
                descriptor,
                timeout_seconds=timeout_seconds,
                internal_colors=internal_colors,
            )
            checked += 1
            class_seconds += duration
            if issue is not None or digest is None:
                issues.append(issue or "missing_result")
            else:
                digests.append(digest)
        distinct = sorted(set(digests))
        collapsed = (
            not issues and len(digests) == len(descriptors) and len(distinct) == 1
        )
        class_results.append(
            {
                "theoretical_class_index": class_index,
                "raw_permutation_multiplicity": len(descriptors),
                "canonicalizations_completed": len(digests),
                "canonical_digests": distinct,
                "collapsed_within_theoretical_class": collapsed,
                "seconds": class_seconds,
                "issues": issues,
            }
        )
        all_digests.extend(digests)
        all_issues.extend(issues)

    expected = len(arrangements)
    completed = len(all_digests)
    theoretical_count = len(theoretical)
    observed = len(set(all_digests))
    invariant = all(
        result["collapsed_within_theoretical_class"] for result in class_results
    )
    complete = checked == expected and completed == expected and not all_issues
    separated = (
        complete
        and invariant
        and len(
            {
                result["canonical_digests"][0]
                for result in class_results
                if result["canonical_digests"]
            }
        )
        == theoretical_count
    )
    quotient = complete and invariant and 0 < observed < theoretical_count
    passed = complete and invariant and 0 < observed <= theoretical_count
    issues = list(all_issues)
    if complete and not invariant:
        issues.append("within_theoretical_class_did_not_collapse")
    if observed > theoretical_count:
        issues.append("more_canonical_classes_than_theoretical_classes")
    return {
        "source": source,
        "record_id": record_id,
        "carrier_id": carrier_id,
        "family": seed.descriptor_class,
        "carrier_status": carrier_status,
        "raw_local_permutations_expected": expected,
        "raw_local_permutations_checked": checked,
        "canonicalizations_completed": completed,
        "theoretical_configuration_classes": theoretical_count,
        "theoretical_class_multiplicities": sorted(
            len(values) for values in theoretical.values()
        ),
        "canonical_classes_observed": observed,
        "representation_invariance_passed": invariant,
        "formal_class_separation_passed": separated,
        "global_symmetry_quotient": quotient,
        "passed": passed,
        "seconds": time.perf_counter() - started,
        "issues": issues,
        "theoretical_class_results": class_results,
    }


def _molecular_record(
    item: tuple[str, str, str, float],
) -> dict[str, Any]:
    source, record_id, smiles, timeout_seconds = item
    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            "id": record_id,
            "status": "parse_failure",
            "parse_failure": True,
            "carriers": [],
            "seconds": time.perf_counter() - started,
        }

    _input_graph, supplied_registry = _rdkit_graph_and_registry(molecule)
    neutral = _stereo_neutral(molecule)
    graph, extracted_after_erasure = _rdkit_graph_and_registry(neutral)
    identifiers = _rdkit_identifier_map(neutral)
    elements = detect_potential_stereo_elements(neutral)
    unique: dict[str, tuple[str, Any]] = {
        descriptor_id(seed): ("supplied_support", seed)
        for descriptor in supplied_registry.values()
        for seed in (_fixed_descriptor_seed(descriptor),)
    }
    perception_issues = []
    for element in elements:
        try:
            seed = _seed_from_potential(element, neutral, identifiers)
        except (TypeError, ValueError) as error:
            perception_issues.append(
                f"{element.source_identifier}: {type(error).__name__}: {error}"
            )
            continue
        unique.setdefault(
            descriptor_id(seed),
            (element.carrier_status.value, seed),
        )

    carriers = []
    for carrier_index, (carrier_status, seed) in enumerate(
        unique[key] for key in sorted(unique)
    ):
        carriers.append(
            _carrier_result(
                source=source,
                record_id=record_id,
                carrier_id=f"{record_id}:local:{carrier_index}",
                carrier_status=carrier_status,
                graph=graph,
                seed=seed,
                timeout_seconds=timeout_seconds,
            )
        )
    passed = not perception_issues and all(carrier["passed"] for carrier in carriers)
    return {
        "id": record_id,
        "status": (
            "no_stereo_carrier"
            if not carriers and not perception_issues
            else ("passed" if passed else "failed")
        ),
        "parse_failure": False,
        "supplied_configurations_retained": 0,
        "known_carrier_supports_from_input": len(supplied_registry),
        "configurations_extracted_after_erasure": len(extracted_after_erasure),
        "potential_carriers_detected": len(elements),
        "perception_issues": perception_issues,
        "carriers": carriers,
        "seconds": time.perf_counter() - started,
    }


def _run_molecular_records(
    items: list[tuple[str, str, str, float]],
    *,
    jobs: int,
) -> list[dict[str, Any]]:
    if jobs == 1:
        return [_molecular_record(item) for item in items]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_molecular_record, items))


def _internal_records(
    *,
    families: tuple[str, ...],
    record_ids: tuple[str, ...],
    limit: int | None,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    fixtures = _fixture_catalog()
    known = {fixture.family for fixture in fixtures}
    unknown = sorted(set(families) - known)
    if unknown:
        raise ValueError(f"Unknown internal stereo families: {unknown}")
    missing = sorted(set(record_ids) - known)
    if missing:
        raise ValueError(f"Unknown internal record identifiers: {missing}")
    selected = [
        fixture
        for fixture in fixtures
        if (not families or fixture.family in families)
        and (not record_ids or fixture.family in record_ids)
    ]
    if limit is not None:
        selected = selected[:limit]
    records = []
    for fixture in selected:
        carrier = _carrier_result(
            source="internal",
            record_id=fixture.family,
            carrier_id=f"{fixture.family}:local:0",
            carrier_status="fixture",
            graph=fixture.graph,
            seed=fixture.seed,
            timeout_seconds=timeout_seconds,
            internal_colors=True,
        )
        records.append(
            {
                "id": fixture.family,
                "status": "passed" if carrier["passed"] else "failed",
                "parse_failure": False,
                "supplied_configurations_retained": 0,
                "potential_carriers_detected": 1,
                "perception_issues": [],
                "carriers": [carrier],
                "seconds": carrier["seconds"],
            }
        )
    return records


def _select_rows(
    task: str,
    *,
    acs_path: Path,
    cip_path: Path | None,
    rota_path: Path,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    if task == "acs":
        rows = load_dataset(acs_path)
        return (
            [(str(row["ID"]), str(row["Input SMILES"])) for row in rows],
            {
                "path": str(acs_path),
                "audited_sha256": ACS_SHA256,
                "records_available": len(rows),
            },
        )
    if task == "cip":
        if cip_path is None:
            raise ValueError("The CIP task requires --cip-path.")
        rows = load_cip(cip_path)
        return (
            [(str(row["ID"]), str(row["SMILES"])) for row in rows],
            {
                "path": str(cip_path),
                "audited_sha256": CIP_SHA256,
                "records_available": len(rows),
            },
        )
    rows = load_rota(rota_path)
    return (
        [(f"RotA-{index:04d}", str(row["SMILES"])) for index, row in enumerate(rows)],
        {
            "path": str(rota_path),
            "audited_sha256": ROTA_SHA256,
            "records_available": len(rows),
        },
    )


def benchmark(
    task: str,
    *,
    acs_path: Path = ACS_DATASET,
    cip_path: Path | None = None,
    rota_path: Path = ROTA,
    record_ids: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
    limit: int | None = None,
    jobs: int = 1,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    if task == "internal":
        records = _internal_records(
            families=families,
            record_ids=record_ids,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )
        dataset = {"records_available": len(_fixture_catalog())}
    else:
        rows, dataset = _select_rows(
            task,
            acs_path=acs_path,
            cip_path=cip_path,
            rota_path=rota_path,
        )
        known = {record_id for record_id, _smiles in rows}
        missing = sorted(set(record_ids) - known)
        if missing:
            raise ValueError(f"Unknown record identifiers: {missing}")
        selected = [
            (task, record_id, smiles, timeout_seconds)
            for record_id, smiles in rows
            if not record_ids or record_id in record_ids
        ]
        if limit is not None:
            selected = selected[:limit]
        records = _run_molecular_records(selected, jobs=jobs)
    carriers = [carrier for record in records for carrier in record["carriers"]]
    expected = sum(carrier["raw_local_permutations_expected"] for carrier in carriers)
    completed = sum(carrier["canonicalizations_completed"] for carrier in carriers)
    dataset["records_selected"] = len(records)
    return {
        "schema": "synkit.configuration-free-local-canonicalization/1",
        "task": task,
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "dataset": dataset,
        "protocol": {
            "fixed_molecular_constitution": True,
            "supplied_stereo_removed_before_perception": True,
            "original_configuration_used": False,
            "known_input_carrier_supports_retained": True,
            "one_carrier_at_a_time": True,
            "other_carrier_configurations": "omitted",
            "joint_carrier_cartesian_product": False,
            "raw_local_permutations": "exhaustive",
            "theoretical_classes": "local_descriptor_group_orbits",
            "timeout_seconds_per_canonicalization": timeout_seconds,
            "worker_processes": jobs,
        },
        "summary": {
            "records_evaluated": len(records),
            "record_statuses": dict(
                sorted(Counter(record["status"] for record in records).items())
            ),
            "parse_failures": sum(record["parse_failure"] for record in records),
            "records_with_carriers": sum(
                bool(record["carriers"]) for record in records
            ),
            "carriers": len(carriers),
            "families": dict(
                sorted(Counter(carrier["family"] for carrier in carriers).items())
            ),
            "raw_local_permutations_expected": expected,
            "raw_local_permutations_checked": sum(
                carrier["raw_local_permutations_checked"] for carrier in carriers
            ),
            "canonicalizations_completed": completed,
            "theoretical_configuration_classes": sum(
                carrier["theoretical_configuration_classes"] for carrier in carriers
            ),
            "canonical_classes_observed": sum(
                carrier["canonical_classes_observed"] for carrier in carriers
            ),
            "carriers_passed": sum(carrier["passed"] for carrier in carriers),
            "carriers_failed": sum(not carrier["passed"] for carrier in carriers),
            "formal_class_separation_passed": sum(
                carrier["formal_class_separation_passed"] for carrier in carriers
            ),
            "global_symmetry_quotients": sum(
                carrier["global_symmetry_quotient"] for carrier in carriers
            ),
            "complete": completed == expected,
            "carrier_accuracy": (
                sum(carrier["passed"] for carrier in carriers) / len(carriers)
                if carriers
                else None
            ),
        },
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This tests exhaustive local representation collapse for each "
            "known or constitutionally perceived carrier. Supplied descriptors "
            "locate supports only; their configurations are discarded. The "
            "benchmark neither recovers input labels nor enumerates the joint "
            "N-carrier product."
        ),
    }


def _write_table(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_FIELDS)
        writer.writeheader()
        for record in report["records"]:
            for carrier in record["carriers"]:
                row = {}
                for field in _TABLE_FIELDS:
                    value = carrier[field]
                    if isinstance(value, bool):
                        value = str(value).lower()
                    elif isinstance(value, list):
                        value = ";".join(map(str, value))
                    row[field] = value
                writer.writerow(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=_TASKS)
    parser.add_argument("--acs-path", type=Path, default=ACS_DATASET)
    parser.add_argument("--cip-path", type=Path)
    parser.add_argument("--rota-path", type=Path, default=ROTA)
    parser.add_argument("--record-id", action="append")
    parser.add_argument("--family", action="append")
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
    try:
        report = benchmark(
            arguments.task,
            acs_path=arguments.acs_path,
            cip_path=arguments.cip_path,
            rota_path=arguments.rota_path,
            record_ids=tuple(arguments.record_id or ()),
            families=tuple(arguments.family or ()),
            limit=arguments.limit,
            jobs=arguments.jobs,
            timeout_seconds=arguments.timeout,
        )
    except ValueError as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(arguments.table, report)
    print(
        json.dumps(
            {
                "task": arguments.task,
                "output": str(arguments.output),
                "table": str(arguments.table),
                "summary": report["summary"],
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not report["summary"]["carriers_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
