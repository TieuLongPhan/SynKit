#!/usr/bin/env python3
"""Exhaustively canonicalize local stereo permutations with a fixed graph.

The primary experiment changes only the local reference ordering of one
configured stereo element.  It does not renumber every atom in the molecule.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    load_cip,
)
from Experiment.Stereo.Canonicalization.global_local import (  # noqa: E402
    _case_time_limit,
    _fixture_catalog,
)
from Experiment.Stereo.Canonicalization.inventory import (  # noqa: E402
    DEFAULT_CSV,
    DEFAULT_JSON,
    build_inventory,
    write_inventory,
)
from Experiment.Stereo.Canonicalization.local_permutations import (  # noqa: E402
    all_local_arrangements,
    same_configuration_representations,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    DATASET as ACS_DATASET,
    load_dataset,
)
from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    TetrahedralStereo,
    canonicalize_configured_registry,
    canonicalize_configured_stereograph,
    descriptor_id,
    local_configuration_classes,
)
from synkit.Graph.Stereo.canonical import (  # noqa: E402
    _rdkit_graph_and_registry,
)
from synkit.Graph.Stereo.enumeration import (  # noqa: E402
    _double_bond_unknown,
    _rdkit_identifier_map,
    _translate_reference,
)
from synkit.Graph.Stereo.rdkit_adapter import _virtual_ligand  # noqa: E402
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    StereoElementType,
    detect_potential_stereo_elements,
)

CANON_DATA_ROOT = ROOT / "Experiment" / "Stereo" / "Data" / "Canonicalization"
_TASKS = ("internal", "acs", "cip")
_TABLE_FIELDS = (
    "source",
    "record_id",
    "element_id",
    "family",
    "configuration_index",
    "configured_elements_in_graph",
    "local_permutations_expected",
    "local_permutations_checked",
    "permutations_passed",
    "permutation_failures",
    "distinct_canonical_stereographs",
    "canonical_digest",
    "passed",
    "seconds",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _canonicalize(
    graph: Any,
    registry: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[Any | None, str | None, float]:
    started = time.perf_counter()
    try:
        with _case_time_limit(timeout_seconds):
            result = canonicalize_configured_registry(graph, registry)
    except TimeoutError:
        return None, "timeout", time.perf_counter() - started
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return (
            None,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
        )
    return result, None, time.perf_counter() - started


def _internal_canonicalize(
    graph: Any,
    descriptor: Any,
    *,
    timeout_seconds: float,
) -> tuple[Any | None, str | None, float]:
    started = time.perf_counter()
    try:
        with _case_time_limit(timeout_seconds):
            result = canonicalize_configured_stereograph(
                graph,
                (descriptor,),
                atom_color="color",
                bond_color="color",
            )
    except TimeoutError:
        return None, "timeout", time.perf_counter() - started
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return (
            None,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
        )
    return result, None, time.perf_counter() - started


def _internal_task(
    *,
    families: tuple[str, ...],
    timeout_seconds: float,
) -> dict[str, Any]:
    fixtures = _fixture_catalog()
    known = {fixture.family for fixture in fixtures}
    unknown = sorted(set(families) - known)
    if unknown:
        raise ValueError(f"Unknown internal stereo families: {unknown}")
    selected = [
        fixture for fixture in fixtures if not families or fixture.family in families
    ]
    rows = []
    family_reports = []
    started_task = time.perf_counter()
    for fixture in selected:
        started_family = time.perf_counter()
        arrangements = all_local_arrangements(fixture.seed)
        grouped: dict[str, list[Any]] = defaultdict(list)
        for descriptor in arrangements:
            grouped[repr(descriptor.canonical_form())].append(descriptor)

        class_results = []
        digest_to_classes: dict[str, set[int]] = defaultdict(set)
        for configuration_index, descriptors in enumerate(
            grouped[key] for key in sorted(grouped)
        ):
            digests = []
            issues = []
            seconds = 0.0
            for descriptor in descriptors:
                result, issue, duration = _internal_canonicalize(
                    fixture.graph,
                    descriptor,
                    timeout_seconds=timeout_seconds,
                )
                seconds += duration
                if issue is not None or result is None:
                    issues.append(issue or "missing_result")
                else:
                    digests.append(result.canonical_digest)
            for digest in set(digests):
                digest_to_classes[digest].add(configuration_index)
            class_results.append(
                {
                    "configuration_index": configuration_index,
                    "expected": len(descriptors),
                    "checked": len(descriptors),
                    "digests": digests,
                    "issues": issues,
                    "seconds": seconds,
                }
            )

        collided = {
            digest
            for digest, configuration_indices in digest_to_classes.items()
            if len(configuration_indices) > 1
        }
        family_rows = []
        for record in class_results:
            distinct = sorted(set(record["digests"]))
            passed = (
                not record["issues"]
                and len(distinct) == 1
                and not (set(distinct) & collided)
            )
            passed_permutations = record["checked"] if passed else 0
            row = {
                "source": "internal",
                "record_id": fixture.family,
                "element_id": f"{fixture.family}:0",
                "family": fixture.family,
                "configuration_index": record["configuration_index"],
                "configured_elements_in_graph": 1,
                "local_permutations_expected": record["expected"],
                "local_permutations_checked": record["checked"],
                "permutations_passed": passed_permutations,
                "permutation_failures": (record["checked"] - passed_permutations),
                "distinct_canonical_stereographs": len(distinct),
                "canonical_digest": distinct[0] if len(distinct) == 1 else "",
                "passed": passed,
                "seconds": record["seconds"],
                "issues": record["issues"],
            }
            rows.append(row)
            family_rows.append(row)

        family_reports.append(
            {
                "family": fixture.family,
                "raw_local_permutations": len(arrangements),
                "configuration_classes": len(grouped),
                "distinct_canonical_stereographs": len(digest_to_classes),
                "certificate_collisions": len(collided),
                "permutations_passed": sum(
                    row["permutations_passed"] for row in family_rows
                ),
                "passed": all(row["passed"] for row in family_rows),
                "seconds": time.perf_counter() - started_family,
            }
        )
    total_permutations = sum(row["local_permutations_checked"] for row in rows)
    passed_permutations = sum(row["permutations_passed"] for row in rows)
    return {
        "task": "internal",
        "scope": "exhaustive_local_stereo_permutations_fixed_graph",
        "summary": {
            "families": len(family_reports),
            "families_passed": sum(report["passed"] for report in family_reports),
            "configuration_classes": len(rows),
            "distinct_canonical_stereographs": sum(
                report["distinct_canonical_stereographs"] for report in family_reports
            ),
            "local_permutations_checked": total_permutations,
            "permutations_passed": passed_permutations,
            "permutation_failures": (total_permutations - passed_permutations),
            "local_canonicalization_accuracy": _ratio(
                passed_permutations,
                total_permutations,
            ),
        },
        "families": family_reports,
        "table_rows": rows,
        "seconds": time.perf_counter() - started_task,
    }


def _select_ids(
    candidates: list[dict[str, Any]],
    *,
    record_ids: tuple[str, ...],
    limit: int | None,
) -> list[dict[str, Any]]:
    if record_ids:
        by_id = {record["record_id"]: record for record in candidates}
        missing = sorted(set(record_ids) - by_id.keys())
        if missing:
            raise ValueError(f"Unknown record identifiers: {missing}")
        selected = [by_id[record_id] for record_id in record_ids]
    else:
        selected = candidates
    return selected[:limit] if limit is not None else selected


def _source_enumeration_seeds(
    molecule: Chem.Mol,
    registry: dict[str, Any],
    *,
    source_unit_tags: tuple[str, ...],
    source_labels: tuple[str, ...],
    source_smiles: str | None,
) -> dict[str, Any]:
    """Recover only source-declared supports whose orientation was erased."""
    tags = set(source_unit_tags)
    identifiers = _rdkit_identifier_map(molecule)
    seeds: dict[str, Any] = {}
    recover_unextractable = not registry
    for element in detect_potential_stereo_elements(molecule):
        support = element.support
        seed = None
        if (
            element.element_type is StereoElementType.EXTENDED_CIS_TRANS
            and "CT4" in tags
        ):
            seed = ExtendedCisTransStereo(
                tuple(identifiers[atom] for atom in support.path),
                tuple(
                    tuple(
                        _translate_reference(reference, identifiers)
                        for reference in frame
                    )
                    for frame in support.terminal_frames
                ),  # type: ignore[arg-type]
                None,
                "cip_ct4_topology_enumeration",
            )
        elif (
            recover_unextractable
            and element.element_type is StereoElementType.DOUBLE_BOND
            and "CT" in tags
        ):
            seed = _double_bond_unknown(
                molecule,
                support,
                identifiers,
            )
        elif (
            recover_unextractable
            and element.element_type is StereoElementType.CUMULENE_AXIS
            and tags & {"TH3", "TH5"}
        ):
            seed = CumuleneAxisStereo(
                tuple(identifiers[atom] for atom in support.path),
                tuple(
                    tuple(
                        _translate_reference(reference, identifiers)
                        for reference in frame
                    )
                    for frame in support.terminal_frames
                ),  # type: ignore[arg-type]
                None,
                "cip_extended_tetrahedral_topology_enumeration",
            )
        elif (
            recover_unextractable
            and element.element_type is StereoElementType.ATROP_AXIS
            and "AT" in tags
        ):
            seed = AtropBondStereo(
                (
                    *tuple(
                        _translate_reference(reference, identifiers)
                        for reference in support.terminal_frames[0]
                    ),
                    identifiers[support.path[0]],
                    identifiers[support.path[-1]],
                    *tuple(
                        _translate_reference(reference, identifiers)
                        for reference in support.terminal_frames[1]
                    ),
                ),  # type: ignore[arg-type]
                None,
                "cip_atrop_topology_enumeration",
            )
        if seed is not None:
            seeds.setdefault(descriptor_id(seed), seed)

    if recover_unextractable and "TH" in tags and source_smiles is not None:
        raw = Chem.MolFromSmiles(source_smiles, sanitize=False)
        if raw is not None:
            tetrahedral_tags = {
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            }
            for raw_atom in raw.GetAtoms():
                if raw_atom.GetChiralTag() not in tetrahedral_tags:
                    continue
                atom_index = raw_atom.GetIdx()
                atom = molecule.GetAtomWithIdx(atom_index)
                center = identifiers[atom_index]
                references = [
                    identifiers[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()
                ]
                if len(references) == 3:
                    references.append(_virtual_ligand(atom, center))
                if len(references) != 4:
                    # Extended-tetrahedral source tags also use ``@`` on the
                    # internal cumulene atom; their path support was handled
                    # above and must not be miscast as local tetrahedral.
                    continue
                seed = TetrahedralStereo(
                    (center, *references),
                    None,
                    "cip_raw_tetrahedral_carrier_enumeration",
                )
                seeds.setdefault(descriptor_id(seed), seed)

    if recover_unextractable and "HE" in tags:
        helical_labels = [
            (int(label[:-1]), label[-1])
            for label in source_labels
            if len(label) > 1 and label[:-1].isdigit() and label[-1] in {"P", "M"}
        ]
        if (
            len(helical_labels) == 2
            and len({position for position, _label in helical_labels}) == 2
            and len({_label for _position, _label in helical_labels}) == 1
            and all(
                1 <= position <= molecule.GetNumAtoms()
                for position, _label in helical_labels
            )
        ):
            start, end = (position - 1 for position, _label in sorted(helical_labels))
            path = _unique_shortest_atom_path(molecule, start, end)
            if path is not None and len(path) >= 4:
                reported_positions = tuple(
                    identifiers[position - 1]
                    for position, _label in sorted(helical_labels)
                )
                seed = HelicalStereo(
                    tuple(identifiers[atom] for atom in path),
                    None,
                    "cip_helical_reported_positions_enumeration",
                    False,
                    reported_positions,
                )
                seeds.setdefault(descriptor_id(seed), seed)
    return seeds


def _unique_shortest_atom_path(
    molecule: Chem.Mol,
    start: int,
    end: int,
) -> tuple[int, ...] | None:
    """Return one shortest atom path only when the graph makes it unique."""
    distances = {start: 0}
    path_counts = {start: 1}
    predecessors: dict[int, int] = {}
    pending = deque([start])
    while pending:
        atom_index = pending.popleft()
        distance = distances[atom_index] + 1
        atom = molecule.GetAtomWithIdx(atom_index)
        for neighbor in atom.GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor_index not in distances:
                distances[neighbor_index] = distance
                path_counts[neighbor_index] = path_counts[atom_index]
                predecessors[neighbor_index] = atom_index
                pending.append(neighbor_index)
            elif distances[neighbor_index] == distance:
                path_counts[neighbor_index] += path_counts[atom_index]
    if end not in distances or path_counts[end] != 1:
        return None
    reversed_path = [end]
    while reversed_path[-1] != start:
        reversed_path.append(predecessors[reversed_path[-1]])
    return tuple(reversed(reversed_path))


def _public_record(
    molecule: Chem.Mol,
    *,
    source: str,
    record_id: str,
    timeout_seconds: float,
    source_unit_tags: tuple[str, ...] = (),
    source_labels: tuple[str, ...] = (),
    source_smiles: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_record = time.perf_counter()
    graph, registry = _rdkit_graph_and_registry(molecule)
    baseline, baseline_issue, baseline_seconds = _canonicalize(
        graph,
        dict(registry),
        timeout_seconds=timeout_seconds,
    )
    if baseline_issue is not None or baseline is None:
        return [], {
            "id": record_id,
            "configured_elements": len(registry),
            "extracted_configured_elements": len(registry),
            "status": "baseline_failed",
            "issue": baseline_issue,
            "seconds": time.perf_counter() - started_record,
        }

    rows = []
    for element_index, (element_id, descriptor) in enumerate(sorted(registry.items())):
        representations = same_configuration_representations(descriptor)
        digests = []
        issues = []
        seconds = 0.0
        for representation in representations:
            variant_registry = dict(registry)
            variant_registry[element_id] = representation
            result, issue, duration = _canonicalize(
                graph,
                variant_registry,
                timeout_seconds=timeout_seconds,
            )
            seconds += duration
            if issue is not None or result is None:
                issues.append(issue or "missing_result")
            else:
                digests.append(result.canonical_digest)
        passed_checks = sum(digest == baseline.canonical_digest for digest in digests)
        passed = (
            not issues
            and passed_checks == len(representations)
            and len(set(digests)) == 1
        )
        rows.append(
            {
                "source": source,
                "record_id": record_id,
                "element_id": element_id,
                "family": descriptor.descriptor_class,
                "configuration_index": 0,
                "configured_elements_in_graph": len(registry),
                "local_permutations_expected": len(representations),
                "local_permutations_checked": len(representations),
                "permutations_passed": (passed_checks if not issues else passed_checks),
                "permutation_failures": (len(representations) - passed_checks),
                "distinct_canonical_stereographs": len(set(digests)),
                "canonical_digest": (
                    next(iter(set(digests))) if len(set(digests)) == 1 else ""
                ),
                "passed": passed,
                "seconds": seconds,
                "issues": issues,
                "element_index": element_index,
            }
        )
    source_seeds = _source_enumeration_seeds(
        molecule,
        dict(registry),
        source_unit_tags=source_unit_tags,
        source_labels=source_labels,
        source_smiles=source_smiles,
    )
    source_classes = {
        element_id: local_configuration_classes(seed)
        for element_id, seed in source_seeds.items()
    }
    source_context = dict(registry)
    source_context.update(
        {element_id: classes[0] for element_id, classes in source_classes.items()}
    )
    enumerated_families: Counter[str] = Counter()
    source_class_count = 0
    source_elements_with_collapse = 0
    source_collapsed_classes = 0
    extended_elements = 0
    extended_classes = 0
    extended_certificates: set[str] = set()
    extended_collisions = 0
    for source_index, (element_id, seed) in enumerate(sorted(source_seeds.items())):
        classes = source_classes[element_id]
        class_rows = []
        class_digests = []
        for configuration_index, descriptor in enumerate(classes):
            variant_registry = dict(source_context)
            variant_registry[element_id] = descriptor
            class_baseline, class_issue, class_seconds = _canonicalize(
                graph,
                variant_registry,
                timeout_seconds=timeout_seconds,
            )
            representations = same_configuration_representations(descriptor)
            digests = []
            issues = []
            seconds = class_seconds
            if class_issue is not None or class_baseline is None:
                issues.append(class_issue or "missing_class_baseline")
            else:
                for representation in representations:
                    representation_registry = dict(variant_registry)
                    representation_registry[element_id] = representation
                    result, issue, duration = _canonicalize(
                        graph,
                        representation_registry,
                        timeout_seconds=timeout_seconds,
                    )
                    seconds += duration
                    if issue is not None or result is None:
                        issues.append(issue or "missing_result")
                    else:
                        digests.append(result.canonical_digest)
                class_digests.append(class_baseline.canonical_digest)
            passed_checks = (
                0
                if class_baseline is None
                else sum(
                    digest == class_baseline.canonical_digest for digest in digests
                )
            )
            passed = (
                not issues
                and passed_checks == len(representations)
                and len(set(digests)) == 1
            )
            row = {
                "source": source,
                "record_id": record_id,
                "element_id": element_id,
                "family": descriptor.descriptor_class,
                "configuration_index": configuration_index,
                "configured_elements_in_graph": len(source_context),
                "local_permutations_expected": len(representations),
                "local_permutations_checked": len(representations),
                "permutations_passed": passed_checks,
                "permutation_failures": (len(representations) - passed_checks),
                "distinct_canonical_stereographs": len(set(digests)),
                "canonical_digest": (
                    next(iter(set(digests))) if len(set(digests)) == 1 else ""
                ),
                "passed": passed,
                "seconds": seconds,
                "issues": issues,
                "element_index": len(registry) + source_index,
                "enumeration_source": seed.provenance,
            }
            rows.append(row)
            class_rows.append(row)
        collided = len(class_digests) == len(classes) and len(
            set(class_digests)
        ) != len(classes)
        if collided:
            source_elements_with_collapse += 1
            source_collapsed_classes += len(classes) - len(set(class_digests))
        enumerated_families[seed.descriptor_class] += 1
        source_class_count += len(classes)
        if isinstance(seed, ExtendedCisTransStereo):
            extended_elements += 1
            extended_classes += len(classes)
            extended_certificates.update(class_digests)
            if collided:
                extended_collisions += 1
                # The CIP CT4 fixtures selected for this benchmark have
                # constitutionally distinct terminal substituents, so their
                # two configurations must remain distinguishable. Other CIP
                # source markers can intentionally denote non-stereogenic
                # supports, where whole-graph symmetry collapse is valid.
                for row in class_rows:
                    row["passed"] = False
                    row["permutations_passed"] = 0
                    row["permutation_failures"] = row["local_permutations_checked"]
                    row["issues"].append("extended_configuration_certificate_collision")
    element_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        element_rows[row["element_id"]].append(row)
    return rows, {
        "id": record_id,
        "configured_elements": len(element_rows),
        "extracted_configured_elements": len(registry),
        "elements_passed": sum(
            all(row["passed"] for row in values) for values in element_rows.values()
        ),
        "configuration_classes": len(rows),
        "configuration_classes_passed": sum(row["passed"] for row in rows),
        "enumerated_source_elements": len(source_seeds),
        "enumerated_source_configuration_classes": source_class_count,
        "enumerated_source_families": dict(sorted(enumerated_families.items())),
        "source_elements_with_configuration_collapse": (source_elements_with_collapse),
        "source_configuration_classes_collapsed": source_collapsed_classes,
        "enumerated_extended_cis_trans_elements": extended_elements,
        "enumerated_extended_cis_trans_classes": extended_classes,
        "distinct_extended_configuration_certificates": len(extended_certificates),
        "extended_configuration_certificate_collisions": (extended_collisions),
        "status": (
            "passed"
            if rows and all(row["passed"] for row in rows)
            else "failed" if rows else "no_supported_stereo"
        ),
        "baseline_seconds": baseline_seconds,
        "seconds": time.perf_counter() - started_record,
    }


def _public_record_from_smiles(
    item: tuple[
        str,
        str,
        str,
        float,
        tuple[str, ...],
        tuple[str, ...],
    ],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one public record in the current worker process."""
    (
        source,
        record_id,
        smiles,
        timeout_seconds,
        source_unit_tags,
        source_labels,
    ) = item
    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected {source.upper()} case {record_id}.")
    return _public_record(
        molecule,
        source=source,
        record_id=record_id,
        timeout_seconds=timeout_seconds,
        source_unit_tags=source_unit_tags,
        source_labels=source_labels,
        source_smiles=smiles,
    )


def _run_public_records(
    items: list[
        tuple[
            str,
            str,
            str,
            float,
            tuple[str, ...],
            tuple[str, ...],
        ]
    ],
    *,
    jobs: int,
) -> tuple[tuple[list[dict[str, Any]], dict[str, Any]], ...]:
    if jobs == 1:
        return tuple(_public_record_from_smiles(item) for item in items)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return tuple(executor.map(_public_record_from_smiles, items))


def _public_summary(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    source_records_selected: int,
) -> dict[str, Any]:
    checked = sum(row["local_permutations_checked"] for row in rows)
    passed = sum(row["permutations_passed"] for row in rows)
    testable_records = sum(record["configured_elements"] > 0 for record in records)
    extracted_records = sum(
        record.get("extracted_configured_elements", 0) > 0 for record in records
    )
    source_enumerated_records = sum(
        record.get("enumerated_source_elements", 0) > 0 for record in records
    )
    element_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        element_groups[(row["record_id"], row["element_id"])].append(row)
    family_elements = {
        (record_id, element_id, values[0]["family"])
        for (record_id, element_id), values in element_groups.items()
    }
    enumerated_source_families: Counter[str] = Counter()
    for record in records:
        enumerated_source_families.update(record.get("enumerated_source_families", {}))
    return {
        "source_records_selected": source_records_selected,
        "records_with_rdkit_extracted_configured_stereo": extracted_records,
        "records_with_source_enumerated_stereo": source_enumerated_records,
        "records_with_testable_stereo_support": testable_records,
        "records_without_testable_stereo_support": (
            source_records_selected - testable_records
        ),
        "record_statuses": dict(
            sorted(Counter(record["status"] for record in records).items())
        ),
        "records_passed": sum(record["status"] == "passed" for record in records),
        "configured_elements": len(element_groups),
        "elements_passed": sum(
            all(row["passed"] for row in values) for values in element_groups.values()
        ),
        "configuration_classes": len(rows),
        "configuration_classes_passed": sum(row["passed"] for row in rows),
        "enumerated_source_elements": sum(
            record.get("enumerated_source_elements", 0) for record in records
        ),
        "enumerated_source_configuration_classes": sum(
            record.get("enumerated_source_configuration_classes", 0)
            for record in records
        ),
        "enumerated_source_families": dict(sorted(enumerated_source_families.items())),
        "source_elements_with_configuration_collapse": sum(
            record.get(
                "source_elements_with_configuration_collapse",
                0,
            )
            for record in records
        ),
        "source_configuration_classes_collapsed": sum(
            record.get(
                "source_configuration_classes_collapsed",
                0,
            )
            for record in records
        ),
        "enumerated_extended_cis_trans_elements": sum(
            record.get("enumerated_extended_cis_trans_elements", 0)
            for record in records
        ),
        "enumerated_extended_cis_trans_classes": sum(
            record.get("enumerated_extended_cis_trans_classes", 0) for record in records
        ),
        "extended_configuration_certificate_collisions": sum(
            record.get(
                "extended_configuration_certificate_collisions",
                0,
            )
            for record in records
        ),
        "local_permutations_checked": checked,
        "permutations_passed": passed,
        "permutation_failures": checked - passed,
        "distinct_families": dict(
            sorted(
                Counter(
                    family for _record_id, _element_id, family in family_elements
                ).items()
            )
        ),
        "rdkit_direct_extraction_coverage": _ratio(
            extracted_records,
            source_records_selected,
        ),
        "stereo_support_coverage": _ratio(
            testable_records,
            source_records_selected,
        ),
        "coverage_status": (
            "complete" if testable_records == source_records_selected else "partial"
        ),
        "local_canonicalization_accuracy": _ratio(passed, checked),
    }


def _acs_task(
    inventory: dict[str, Any],
    *,
    path: Path,
    record_ids: tuple[str, ...],
    limit: int | None,
    timeout_seconds: float,
    jobs: int = 1,
) -> dict[str, Any]:
    candidates = _select_ids(
        [
            record
            for record in inventory["records"]
            if record["source"] == "acs_stereomolgraph"
        ],
        record_ids=record_ids,
        limit=limit,
    )
    source_rows = {row["ID"]: row for row in load_dataset(path)}
    table_rows = []
    records = []
    started = time.perf_counter()
    items = [
        (
            "acs",
            candidate["record_id"],
            source_rows[candidate["record_id"]]["Input SMILES"],
            timeout_seconds,
            (),
            (),
        )
        for candidate in candidates
    ]
    for rows, record in _run_public_records(items, jobs=jobs):
        table_rows.extend(rows)
        records.append(record)
    return {
        "task": "acs",
        "scope": "exhaustive_local_same_configuration_permutations",
        "summary": _public_summary(
            table_rows,
            records,
            source_records_selected=len(candidates),
        ),
        "records": records,
        "table_rows": table_rows,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Each configured stereo element is permuted locally while every "
            "other element and the molecular graph remain fixed. ACS global "
            "chiral/achiral labels are not used."
        ),
    }


def _cip_task(
    inventory: dict[str, Any],
    *,
    path: Path,
    record_ids: tuple[str, ...],
    limit: int | None,
    timeout_seconds: float,
    jobs: int = 1,
) -> dict[str, Any]:
    candidates = _select_ids(
        [
            record
            for record in inventory["records"]
            if record["source"] == "cip_validation_suite"
        ],
        record_ids=record_ids,
        limit=limit,
    )
    source_rows = {row["ID"]: row for row in load_cip(path)}
    table_rows = []
    records = []
    started = time.perf_counter()
    items = [
        (
            "cip",
            candidate["record_id"],
            source_rows[candidate["record_id"]]["SMILES"],
            timeout_seconds,
            tuple(candidate["source_categories"]),
            tuple(
                source_rows[candidate["record_id"]].get(
                    "recommended_labels",
                    (),
                )
            ),
        )
        for candidate in candidates
    ]
    for candidate, (rows, record) in zip(
        candidates,
        _run_public_records(items, jobs=jobs),
    ):
        record["source_unit_tags"] = candidate["source_categories"]
        table_rows.extend(rows)
        records.append(record)
    return {
        "task": "cip",
        "scope": "exhaustive_local_same_configuration_permutations",
        "summary": _public_summary(
            table_rows,
            records,
            source_records_selected=len(candidates),
        ),
        "records": records,
        "table_rows": table_rows,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "CIP supplies local configured-unit provenance, not whole-"
            "molecule chirality truth. For source-declared TH, CT, AT, TH3, "
            "TH5, CT4, and HE supports whose orientation RDKit erased, the "
            "runner enumerates both formal configurations and tests every "
            "equivalent local representation; this does not recover source "
            "orientation. HE recovery additionally requires two consistently "
            "labelled source positions joined by one unique shortest path. "
            "Whole-graph symmetry may validly collapse opposite formal "
            "classes for non-stereogenic source markers. CT4 fixtures require "
            "two distinct certificates."
        ),
    }


def _write_table(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        str(row.get(field)).lower()
                        if isinstance(row.get(field), bool)
                        else row.get(field, "")
                    )
                    for field in _TABLE_FIELDS
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=_TASKS)
    parser.add_argument("--family", action="append")
    parser.add_argument("--record-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--acs-path", type=Path, default=ACS_DATASET)
    parser.add_argument("--cip-path", type=Path)
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be at least one.")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive.")
    if arguments.jobs < 1:
        parser.error("--jobs must be at least one.")
    if arguments.task == "cip" and arguments.cip_path is None:
        parser.error("the cip task requires --cip-path")
    RDLogger.DisableLog("rdApp.*")

    inventory = build_inventory()
    write_inventory(
        inventory,
        json_path=arguments.inventory_json,
        csv_path=arguments.inventory_csv,
    )
    if arguments.task == "internal":
        task_result = _internal_task(
            families=tuple(arguments.family or ()),
            timeout_seconds=arguments.timeout,
        )
    elif arguments.task == "acs":
        task_result = _acs_task(
            inventory,
            path=arguments.acs_path,
            record_ids=tuple(arguments.record_id or ()),
            limit=arguments.limit,
            timeout_seconds=arguments.timeout,
            jobs=arguments.jobs,
        )
    else:
        task_result = _cip_task(
            inventory,
            path=arguments.cip_path,
            record_ids=tuple(arguments.record_id or ()),
            limit=arguments.limit,
            timeout_seconds=arguments.timeout,
            jobs=arguments.jobs,
        )

    table_rows = task_result.pop("table_rows")
    result = {
        "schema": "synkit.local-stereo-canonicalization/1",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "protocol": {
            "graph_fixed": True,
            "local_permutations": "exhaustive",
            "multiple_elements": "one_element_at_a_time_others_fixed",
            "timeout_seconds_per_canonicalization": arguments.timeout,
            "worker_processes": arguments.jobs,
            "record_ids": list(arguments.record_id or ()),
            "record_limit": arguments.limit,
        },
        **task_result,
    }
    output = arguments.output or (
        CANON_DATA_ROOT / f"{arguments.task}_local_canonicalization_report.json"
    )
    table = arguments.table or (
        CANON_DATA_ROOT / f"{arguments.task}_local_canonicalization_table.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(table, table_rows)
    print(
        json.dumps(
            {
                "task": arguments.task,
                "report": str(output),
                "table": str(table),
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
