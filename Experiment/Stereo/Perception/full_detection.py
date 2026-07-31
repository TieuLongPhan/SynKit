#!/usr/bin/env python3
"""Run neutral carrier detection over all three stereo datasets."""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import Counter
import json
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Iterable

import networkx as nx
from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.published import (  # noqa: E402
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
from Experiment.Stereo.Perception.family_executors import (  # noqa: E402
    neutralize_rdkit_configuration,
)
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    PotentialStereoElement,
    StereoCarrierStatus,
    StereoElementType,
    analyze_tetrahedral_carriers,
    detect_potential_stereo_elements,
)
from synkit.Chem.Molecule._stereo_orientation_constraints import (  # noqa: E402
    extract_local_orientation_constraints,
)
from synkit.Graph.Stereo.supports import (  # noqa: E402
    AtomStereoSupport,
    AxisStereoSupport,
    BondStereoSupport,
    PathStereoSupport,
    Reference,
)

DEFAULT_REPORT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "full_detection_report.json"
)
_LABEL_PATTERN = re.compile(r"^(\d+)([A-Za-z]+)$")
_AXIS_TYPES = frozenset(
    {
        StereoElementType.ATROP_AXIS,
        StereoElementType.CUMULENE_AXIS,
    }
)
_DETECTION_CACHE: dict[
    tuple[Any, ...],
    tuple[tuple[PotentialStereoElement, ...], bool | None, tuple[str, ...]],
] = {}


class _DetectionTimeout(TimeoutError):
    """Raised when one full carrier-detection case exceeds its budget."""


class _case_time_limit:
    """Bound one benchmark case while restoring the process signal state."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        if not hasattr(signal, "setitimer"):
            return
        self.previous_handler = signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_error: Any) -> None:
        if not hasattr(signal, "setitimer"):
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    @staticmethod
    def _expired(_signum: int, _frame: Any) -> None:
        raise _DetectionTimeout


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _reference_key(reference: Reference) -> tuple[int, str]:
    return (0, str(reference)) if type(reference) is int else (1, reference)


def _normalized_axis(
    support: AxisStereoSupport,
) -> tuple[tuple[int, ...], tuple[Reference, ...], tuple[Reference, ...]]:
    forward = (
        support.path,
        tuple(sorted(support.terminal_frames[0], key=_reference_key)),
        tuple(sorted(support.terminal_frames[1], key=_reference_key)),
    )
    reverse = (
        tuple(reversed(support.path)),
        tuple(sorted(support.terminal_frames[1], key=_reference_key)),
        tuple(sorted(support.terminal_frames[0], key=_reference_key)),
    )
    return min((forward, reverse), key=repr)


def _element_signature(element: PotentialStereoElement) -> tuple[Any, ...]:
    support = element.support
    if isinstance(support, AtomStereoSupport):
        locus: tuple[Any, ...] = ("atom", support.center)
    elif isinstance(support, BondStereoSupport):
        locus = ("bond", *sorted(support.endpoints))
    elif isinstance(support, AxisStereoSupport):
        path, left, right = _normalized_axis(support)
        locus = ("axis", path, left, right)
    elif isinstance(support, PathStereoSupport):
        path = min(support.path, tuple(reversed(support.path)))
        locus = ("path", path, support.cyclic)
    else:  # pragma: no cover
        raise TypeError(f"Unsupported carrier support: {type(support).__name__}")
    return (
        element.element_type.value,
        element.carrier_status.value,
        element.carrier_reason,
        locus,
    )


def _transported_elements(
    elements: Iterable[PotentialStereoElement],
    mapping: dict[int, int],
) -> tuple[PotentialStereoElement, ...]:
    transported = []
    for element in elements:
        clone = PotentialStereoElement(
            element.element_type,
            element.support.relabel(mapping),
            element.configuration_state,
            element.evidence_provenance,
            element.source_identifier,
            carrier_status=element.carrier_status,
            carrier_reason=element.carrier_reason,
        )
        transported.append(clone)
    return tuple(transported)


def _transported_signatures(
    elements: Iterable[PotentialStereoElement],
    mapping: dict[int, int],
) -> set[tuple[Any, ...]]:
    return {
        _element_signature(element)
        for element in _transported_elements(elements, mapping)
    }


def _serialize_element(element: PotentialStereoElement) -> dict[str, Any]:
    support = element.support
    if isinstance(support, AtomStereoSupport):
        payload: dict[str, Any] = {"kind": "atom", "center": support.center}
    elif isinstance(support, BondStereoSupport):
        payload = {"kind": "bond", "endpoints": sorted(support.endpoints)}
    elif isinstance(support, AxisStereoSupport):
        path, left, right = _normalized_axis(support)
        payload = {
            "kind": (
                "path"
                if element.element_type is StereoElementType.EXTENDED_CIS_TRANS
                else "axis"
            ),
            "path": list(path),
            "terminal_frames": [list(left), list(right)],
        }
    elif isinstance(support, PathStereoSupport):
        payload = {
            "kind": "path",
            "path": list(min(support.path, tuple(reversed(support.path)))),
            "cyclic": support.cyclic,
        }
    else:  # pragma: no cover
        raise TypeError(f"Unsupported carrier support: {type(support).__name__}")
    return {
        "family": element.element_type.value,
        "carrier_status": element.carrier_status.value,
        "carrier_reason": element.carrier_reason,
        "support": payload,
    }


def _detect(
    molecule: Chem.Mol,
    *,
    check_renumbering: bool,
    timeout_seconds: float,
    include_extended_ring_axes: bool = False,
    retain_neighbor_configuration: bool = False,
) -> tuple[tuple[PotentialStereoElement, ...], bool | None, list[str]]:
    working = (
        Chem.Mol(molecule)
        if retain_neighbor_configuration
        else neutralize_rdkit_configuration(molecule)
    )
    cache_key = (
        Chem.MolToSmiles(working, canonical=False, isomericSmiles=True),
        tuple(
            (atom.GetIdx(), int(atom.GetChiralTag()))
            for atom in working.GetAtoms()
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ),
        tuple(
            (
                bond.GetIdx(),
                int(bond.GetStereo()),
                tuple(bond.GetStereoAtoms()),
            )
            for bond in working.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE
        ),
        check_renumbering,
        include_extended_ring_axes,
        retain_neighbor_configuration,
    )
    cached = _DETECTION_CACHE.get(cache_key)
    if cached is not None:
        elements, invariant, issues = cached
        return elements, invariant, list(issues)
    with _case_time_limit(timeout_seconds):
        elements = detect_potential_stereo_elements(
            working,
            include_extended_ring_axes=include_extended_ring_axes,
        )
        if not check_renumbering:
            _DETECTION_CACHE[cache_key] = (elements, None, ())
            return elements, None, []
        order = tuple(reversed(range(working.GetNumAtoms())))
        mapping = {old: new for new, old in enumerate(order)}
        renumbered = Chem.RenumberAtoms(working, order)
        observed = {
            _element_signature(element)
            for element in detect_potential_stereo_elements(
                renumbered,
                include_extended_ring_axes=include_extended_ring_axes,
            )
        }
        expected = _transported_signatures(elements, mapping)
    issues = [
        *(f"missing:{value!r}" for value in sorted(expected - observed, key=repr)),
        *(f"extra:{value!r}" for value in sorted(observed - expected, key=repr)),
    ]
    result = (elements, not issues, tuple(issues))
    _DETECTION_CACHE[cache_key] = result
    return elements, not issues, issues


def _confirmed(
    elements: Iterable[PotentialStereoElement],
) -> tuple[PotentialStereoElement, ...]:
    return tuple(
        element
        for element in elements
        if element.carrier_status is StereoCarrierStatus.CONFIRMED
    )


def _prediction_counts(
    records: Iterable[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                carrier["family"]
                for record in records
                for carrier in record.get(field, ())
            ).items()
        )
    )


def _renumbering_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(records)
    checked = [
        record for record in values if record["renumbering_invariant"] is not None
    ]
    failures = [
        {"id": record["id"], "issues": record["renumbering_issues"]}
        for record in checked
        if record["renumbering_invariant"] is False
    ]
    return {
        "checked_records": len(checked),
        "invariant_records": sum(
            record["renumbering_invariant"] is True for record in checked
        ),
        "failures": failures,
    }


def _set_score(
    records: Iterable[dict[str, Any]],
    *,
    reference_field: str,
    prediction_field: str,
) -> dict[str, Any]:
    values = tuple(records)
    reference = sum(len(record[reference_field]) for record in values)
    predicted = sum(len(record[prediction_field]) for record in values)
    true_positive = sum(
        len(set(record[reference_field]) & set(record[prediction_field]))
        for record in values
    )
    exact = sum(
        set(record[reference_field]) == set(record[prediction_field])
        for record in values
    )
    return {
        "records": len(values),
        "reference_loci": reference,
        "predicted_loci": predicted,
        "true_positive_loci": true_positive,
        "recall": _ratio(true_positive, reference),
        "annotation_precision": _ratio(true_positive, predicted),
        "exact_records": exact,
        "exact_record_rate": _ratio(exact, len(values)),
    }


def _reference_recall(
    records: Iterable[dict[str, Any]],
    *,
    reference_field: str,
    prediction_field: str,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Score recovery for a reference subset without inventing negatives."""
    values = tuple(records)
    references = [
        {
            locus
            for locus in record[reference_field]
            if prefix is None or locus.startswith(prefix)
        }
        for record in values
    ]
    predictions = [set(record[prediction_field]) for record in values]
    reference_count = sum(map(len, references))
    recovered = sum(
        len(reference & prediction)
        for reference, prediction in zip(references, predictions)
    )
    return {
        "reference_loci": reference_count,
        "recovered_loci": recovered,
        "recall": _ratio(recovered, reference_count),
    }


def _setting_summary(
    records: Iterable[dict[str, Any]],
    *,
    reference_field: str,
    prediction_field: str,
    category_field: str | None = None,
) -> dict[str, Any]:
    """Summarize one detection setting without treating extras as negatives."""
    values = tuple(records)
    missed = []
    additional = []
    for record in values:
        reference = set(record[reference_field])
        prediction = set(record[prediction_field])
        category = str(record[category_field]) if category_field is not None else None
        missed.extend(
            (
                category or locus.split(":", 1)[0],
                locus,
            )
            for locus in reference - prediction
        )
        additional.extend(
            (
                category or locus.split(":", 1)[0],
                locus,
            )
            for locus in prediction - reference
        )
    return {
        **_set_score(
            values,
            reference_field=reference_field,
            prediction_field=prediction_field,
        ),
        "missed_loci": len(missed),
        "additional_detected_loci": len(additional),
        "missed_by_family": dict(sorted(Counter(item[0] for item in missed).items())),
        "additional_by_family": dict(
            sorted(Counter(item[0] for item in additional).items())
        ),
        "records_with_additional_detected_loci": sum(
            bool(set(record[prediction_field]) - set(record[reference_field]))
            for record in values
        ),
    }


def _erased_miss_classification(
    records: Iterable[dict[str, Any]],
    *,
    reference_field: str,
    erased_prediction_field: str,
    retained_prediction_field: str,
    category_field: str | None = None,
) -> dict[str, Any]:
    """Classify erased-input misses by their retained-orientation outcome."""
    recovered = Counter()
    remaining = Counter()
    for record in records:
        reference = set(record[reference_field])
        erased = set(record[erased_prediction_field])
        retained = set(record[retained_prediction_field])
        for locus in reference - erased:
            category = (
                str(record[category_field])
                if category_field is not None
                else locus.split(":", 1)[0]
            )
            target = recovered if locus in retained else remaining
            target[category] += 1
    return {
        "requires_supplied_local_neighbor_orientation": sum(recovered.values()),
        "requires_supplied_local_neighbor_orientation_by_family": dict(
            sorted(recovered.items())
        ),
        "unrecovered_with_supplied_local_neighbor_orientation": sum(remaining.values()),
        "unrecovered_with_supplied_local_neighbor_orientation_by_family": dict(
            sorted(remaining.items())
        ),
    }


def _erased_tetrahedral_miss_details(
    configured_molecule: Chem.Mol,
    neutral_molecule: Chem.Mol,
    missed_loci: Iterable[str],
    *,
    one_based: bool,
) -> list[dict[str, Any]]:
    """Explain strict tetrahedral misses from topology and nearby frames."""
    evidence = {
        item.support.center: item
        for item in analyze_tetrahedral_carriers(neutral_molecule)
    }
    constraints = extract_local_orientation_constraints(configured_molecule)
    graph = nx.Graph(
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
        )
        for bond in configured_molecule.GetBonds()
    )
    graph.add_nodes_from(range(configured_molecule.GetNumAtoms()))
    details = []
    for locus in sorted(missed_loci):
        if not locus.startswith("tetrahedral:"):
            continue
        source_position = int(locus.rsplit(":", 1)[1])
        center = source_position - 1 if one_based else source_position
        component = nx.node_connected_component(graph, center)
        frame_families = []
        if any(
            constraint.center != center and constraint.center in component
            for constraint in constraints.tetrahedral
        ):
            frame_families.append("tetrahedral")
        if any(constraint.bond <= component for constraint in constraints.planar):
            frame_families.append("planar")
        if any(
            set(constraint.path) <= component for constraint in constraints.cumulene
        ):
            frame_families.append("cumulene")
        atom = configured_molecule.GetAtomWithIdx(center)
        item = evidence.get(center)
        details.append(
            {
                "locus": locus,
                "atom_element": atom.GetSymbol(),
                "ring_membership": ("ring" if atom.IsInRing() else "acyclic"),
                "configuration_erased_status": (
                    "absent_carrier" if item is None else item.status.value
                ),
                "available_local_neighbor_frame_families": frame_families,
            }
        )
    return details


def _erased_miss_detail_summary(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-locus structural explanations."""
    details = [
        detail
        for record in records
        for detail in record.get(
            "configuration_erased_tetrahedral_miss_details",
            (),
        )
    ]
    return {
        "tetrahedral_misses_audited": len(details),
        "by_configuration_erased_status": dict(
            sorted(
                Counter(
                    detail["configuration_erased_status"] for detail in details
                ).items()
            )
        ),
        "by_atom_element": dict(
            sorted(Counter(detail["atom_element"] for detail in details).items())
        ),
        "by_ring_membership": dict(
            sorted(Counter(detail["ring_membership"] for detail in details).items())
        ),
        "by_available_local_neighbor_frame_families": dict(
            sorted(
                Counter(
                    "+".join(detail["available_local_neighbor_frame_families"])
                    or "none"
                    for detail in details
                ).items()
            )
        ),
    }


def _acs_reference_loci(molecule: Chem.Mol) -> set[str]:
    loci = set()
    for info in Chem.FindPotentialStereo(molecule):
        if info.specified != Chem.StereoSpecified.Specified:
            continue
        centered_on = int(info.centeredOn)
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            loci.add(f"tetrahedral:atom:{centered_on}")
        elif info.type == Chem.StereoType.Bond_Double:
            bond = molecule.GetBondWithIdx(centered_on)
            endpoints = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            loci.add(f"double_bond:bond:{endpoints[0]}-{endpoints[1]}")
    return loci


def _exact_locus(element: PotentialStereoElement) -> str:
    support = element.support
    if isinstance(support, AtomStereoSupport):
        return f"{element.element_type.value}:atom:{support.center}"
    if isinstance(support, BondStereoSupport):
        left, right = sorted(support.endpoints)
        return f"{element.element_type.value}:bond:{left}-{right}"
    if isinstance(support, PathStereoSupport):
        path = min(support.path, tuple(reversed(support.path)))
        return f"{element.element_type.value}:path:" + "-".join(map(str, path))
    path, _left, _right = _normalized_axis(support)
    return f"{element.element_type.value}:axis:" + "-".join(map(str, path))


def _benchmark_acs(
    *,
    check_renumbering: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    rows = load_dataset()
    results = []
    constitution_keys = []
    started = time.perf_counter()
    for position, row in enumerate(rows, start=1):
        reference_molecule = Chem.MolFromSmiles(row["Input SMILES"])
        molecule = _molecule_with_local_orientation(row["Input SMILES"])
        if reference_molecule is None or molecule is None:
            results.append({"id": row["ID"], "parse_failure": True})
            continue
        neutral = neutralize_rdkit_configuration(molecule)
        constitution_keys.append(
            Chem.MolToSmiles(neutral, canonical=True, isomericSmiles=False)
        )
        reference = _acs_reference_loci(reference_molecule)
        try:
            elements, invariant, issues = _detect(
                molecule,
                check_renumbering=check_renumbering,
                timeout_seconds=timeout_seconds,
            )
            retained_elements, retained_invariant, retained_issues = _detect(
                molecule,
                check_renumbering=check_renumbering,
                timeout_seconds=timeout_seconds,
                retain_neighbor_configuration=True,
            )
        except _DetectionTimeout:
            results.append(
                {
                    "id": row["ID"],
                    "manual_global_label": row["manual"],
                    "parse_failure": False,
                    "detection_error": "case_timeout",
                    "reference_supplied_loci": sorted(reference),
                }
            )
            continue
        broad_predictions = {_exact_locus(element) for element in elements}
        confirmed_predictions = {
            _exact_locus(element) for element in _confirmed(elements)
        }
        retained_predictions = {_exact_locus(element) for element in retained_elements}
        retained_confirmed_predictions = {
            _exact_locus(element) for element in _confirmed(retained_elements)
        }
        results.append(
            {
                "id": row["ID"],
                "manual_global_label": row["manual"],
                "parse_failure": False,
                "detection_error": None,
                "reference_supplied_loci": sorted(reference),
                "broad_predicted_loci": sorted(broad_predictions),
                "confirmed_predicted_loci": sorted(confirmed_predictions),
                "broad_missed_supplied_loci": sorted(reference - broad_predictions),
                "confirmed_missed_supplied_loci": sorted(
                    reference - confirmed_predictions
                ),
                "broad_additional_detected_loci": sorted(broad_predictions - reference),
                "confirmed_additional_detected_loci": sorted(
                    confirmed_predictions - reference
                ),
                "local_neighbor_orientation_retained_predicted_loci": sorted(
                    retained_predictions
                ),
                (
                    "local_neighbor_orientation_retained_confirmed_" "predicted_loci"
                ): sorted(retained_confirmed_predictions),
                "local_neighbor_orientation_retained_missed_loci": sorted(
                    reference - retained_predictions
                ),
                "local_neighbor_orientation_retained_additional_loci": sorted(
                    retained_predictions - reference
                ),
                "configuration_erased_tetrahedral_miss_details": (
                    _erased_tetrahedral_miss_details(
                        molecule,
                        neutral,
                        reference - broad_predictions,
                        one_based=False,
                    )
                ),
                "detected_carriers": [
                    _serialize_element(element) for element in elements
                ],
                "local_neighbor_orientation_retained_detected_carriers": [
                    _serialize_element(element) for element in retained_elements
                ],
                "renumbering_invariant": invariant,
                "renumbering_issues": issues,
                "local_neighbor_orientation_retained_renumbering_invariant": (
                    retained_invariant
                ),
                "local_neighbor_orientation_retained_renumbering_issues": (
                    retained_issues
                ),
            }
        )
        if position % 100 == 0:
            print(f"ACS detection: {position}/{len(rows)}", file=sys.stderr)
    parsed = [record for record in results if not record["parse_failure"]]
    valid = [record for record in parsed if record["detection_error"] is None]
    all_reference = sum(len(record["reference_supplied_loci"]) for record in parsed)
    modes = {
        mode: {
            **_reference_recall(
                valid,
                reference_field="reference_supplied_loci",
                prediction_field=f"{mode}_predicted_loci",
            ),
            "records_recovering_every_supplied_locus": sum(
                set(record["reference_supplied_loci"])
                <= set(record[f"{mode}_predicted_loci"])
                for record in valid
            ),
        }
        for mode in ("broad", "confirmed")
    }
    by_family = {
        family: {
            mode: _reference_recall(
                valid,
                reference_field="reference_supplied_loci",
                prediction_field=f"{mode}_predicted_loci",
                prefix=f"{family}:",
            )
            for mode in ("broad", "confirmed")
        }
        for family in ("tetrahedral", "double_bond")
    }
    by_global_label = {}
    for label in sorted({record["manual_global_label"] for record in valid}):
        subset = [record for record in valid if record["manual_global_label"] == label]
        by_global_label[label] = {
            "records": len(subset),
            **{
                mode: _reference_recall(
                    subset,
                    reference_field="reference_supplied_loci",
                    prediction_field=f"{mode}_predicted_loci",
                )
                for mode in ("broad", "confirmed")
            },
        }
    retained_prediction_field = "local_neighbor_orientation_retained_predicted_loci"
    return (
        {
            "task": ("two-setting recovery of supplied configured local loci"),
            "dataset_records": len(rows),
            "parsed_records": len(parsed),
            "evaluated_records": len(valid),
            "parse_failure_ids": [
                record["id"] for record in results if record["parse_failure"]
            ],
            "detection_errors": [
                {"id": record["id"], "error": record["detection_error"]}
                for record in parsed
                if record["detection_error"] is not None
            ],
            "case_timeout_seconds": timeout_seconds,
            "audited_sha256": ACS_SHA256,
            "dataset_reference_supplied_loci": all_reference,
            **modes,
            "detected_carriers_by_family": _prediction_counts(
                valid,
                "detected_carriers",
            ),
            "by_reference_family": by_family,
            "by_manual_global_label": by_global_label,
            "renumbering": _renumbering_summary(valid),
            "two_setting_comparison": {
                "configuration_erased": _setting_summary(
                    valid,
                    reference_field="reference_supplied_loci",
                    prediction_field="broad_predicted_loci",
                ),
                "local_neighbor_orientation_retained": _setting_summary(
                    valid,
                    reference_field="reference_supplied_loci",
                    prediction_field=retained_prediction_field,
                ),
            },
            "configuration_erased_unrecovered_classification": (
                _erased_miss_classification(
                    valid,
                    reference_field="reference_supplied_loci",
                    erased_prediction_field="broad_predicted_loci",
                    retained_prediction_field=retained_prediction_field,
                )
            ),
            "configuration_erased_miss_structure": (_erased_miss_detail_summary(valid)),
            "local_neighbor_orientation_retained_renumbering": {
                "checked_records": sum(
                    record[
                        ("local_neighbor_orientation_retained_" "renumbering_invariant")
                    ]
                    is not None
                    for record in valid
                ),
                "invariant_records": sum(
                    record[
                        ("local_neighbor_orientation_retained_" "renumbering_invariant")
                    ]
                    is True
                    for record in valid
                ),
                "failures": [
                    {
                        "id": record["id"],
                        "issues": record[
                            (
                                "local_neighbor_orientation_retained_"
                                "renumbering_issues"
                            )
                        ],
                    }
                    for record in valid
                    if record[
                        ("local_neighbor_orientation_retained_" "renumbering_invariant")
                    ]
                    is False
                ],
            },
            "records": results,
            "seconds": time.perf_counter() - started,
            "claim_boundary": (
                "ACS supplies global chiral/achiral truth, not complete local "
                "carrier annotations. Recall is scored only for configured "
                "atom/bond loci present in the source SMILES. Additional "
                "candidates in either setting are not false positives. The "
                "retained setting uses other supplied local orientation "
                "frames but excludes the focal center's own frame."
            ),
        },
        constitution_keys,
    )


def _pair(path: Iterable[int]) -> tuple[int, int]:
    values = tuple(path)
    return tuple(sorted((values[0], values[-1])))


def _pair_text(pair: Iterable[int]) -> str:
    return "-".join(str(value) for value in sorted(pair))


def _path_text(path: Iterable[int]) -> str:
    values = tuple(path)
    normalized = min(values, tuple(reversed(values)))
    return "-".join(map(str, normalized))


def _rota_predicted_loci(
    molecule: Chem.Mol,
    elements: Iterable[PotentialStereoElement],
    chiral_type: str,
) -> set[str]:
    """Project typed carriers onto each RotA annotation convention."""
    values = tuple(elements)
    if chiral_type == "Chiral atom pair":
        centers = {
            element.support.center
            for element in values
            if element.element_type is StereoElementType.TETRAHEDRAL
        }
        return {
            _pair_text((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for bond in molecule.GetBonds()
            if bond.GetBeginAtomIdx() in centers and bond.GetEndAtomIdx() in centers
        }
    if chiral_type == "Spiral atom and chain":
        rings = tuple(frozenset(ring) for ring in molecule.GetRingInfo().AtomRings())
        centers = {
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP3
            and atom.GetDegree() == 4
            and any(
                left & right == {atom.GetIdx()}
                for position, left in enumerate(rings)
                for right_position, right in enumerate(rings)
                if right_position > position
            )
        }
        chain_edges = {
            tuple(sorted((left, right)))
            for ring in rings
            if len(ring) == 4
            for left in ring
            for right in ring
            if left < right
            and molecule.GetBondBetweenAtoms(left, right) is None
            and molecule.GetAtomWithIdx(left).GetHybridization()
            == Chem.HybridizationType.SP3
            and molecule.GetAtomWithIdx(right).GetHybridization()
            == Chem.HybridizationType.SP3
            and molecule.GetAtomWithIdx(left).GetDegree() == 4
            and molecule.GetAtomWithIdx(right).GetDegree() == 4
        }
        chain_graph = nx.Graph()
        chain_graph.add_edges_from(chain_edges)
        chain_pairs = set()
        for component in nx.connected_components(chain_graph):
            terminals = sorted(
                center for center in component if chain_graph.degree(center) == 1
            )
            if len(terminals) == 2:
                chain_pairs.add(tuple(terminals))
        return {
            *(_pair_text((center,)) for center in centers),
            *(_pair_text(pair) for pair in chain_pairs),
        }
    return {
        _pair_text(_pair(element.support.path))
        for element in values
        if element.element_type in _AXIS_TYPES
    }


def _rota_projection_issues(
    molecule: Chem.Mol,
    elements: tuple[PotentialStereoElement, ...],
    chiral_type: str,
    broad_loci: set[str],
    confirmed_loci: set[str],
) -> list[str]:
    order = tuple(reversed(range(molecule.GetNumAtoms())))
    mapping = {old: new for new, old in enumerate(order)}
    renumbered = Chem.RenumberAtoms(molecule, order)
    transported = _transported_elements(elements, mapping)

    def remap(loci: set[str]) -> set[str]:
        return {
            _pair_text(mapping[int(value)] for value in locus.split("-"))
            for locus in loci
        }

    issues = []
    for mode, source, candidates in (
        ("broad", transported, broad_loci),
        ("confirmed", _confirmed(transported), confirmed_loci),
    ):
        expected = remap(candidates)
        observed = _rota_predicted_loci(
            renumbered,
            source,
            chiral_type,
        )
        issues.extend(
            f"{mode}_projection_missing:{value}"
            for value in sorted(expected - observed)
        )
        issues.extend(
            f"{mode}_projection_extra:{value}" for value in sorted(observed - expected)
        )
    return issues


def _benchmark_rota(
    *,
    check_renumbering: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    rows = load_rota(ROTA)
    results = []
    constitution_keys = []
    started = time.perf_counter()
    for position, row in enumerate(rows, start=1):
        identifier = f"RotA-{position:04d}"
        molecule = _molecule_with_local_orientation(row["SMILES"])
        if molecule is None:
            results.append({"id": identifier, "parse_failure": True})
            continue
        neutral = neutralize_rdkit_configuration(molecule)
        constitution_keys.append(
            Chem.MolToSmiles(neutral, canonical=True, isomericSmiles=False)
        )
        reference = {_pair_text(pair) for pair in literal_eval(row["label"])}
        reference_paths = {
            _path_text(path) for path in literal_eval(row["label_expanded"])
        }
        try:
            elements, invariant, issues = _detect(
                molecule,
                check_renumbering=check_renumbering,
                timeout_seconds=timeout_seconds,
                include_extended_ring_axes=True,
            )
            retained_elements, retained_invariant, retained_issues = _detect(
                molecule,
                check_renumbering=check_renumbering,
                timeout_seconds=timeout_seconds,
                include_extended_ring_axes=True,
                retain_neighbor_configuration=True,
            )
        except _DetectionTimeout:
            results.append(
                {
                    "id": identifier,
                    "chiral_type": row["chiral_type"],
                    "parse_failure": False,
                    "detection_error": "case_timeout",
                    "reference_pairs": sorted(reference),
                    "reference_expanded_paths": sorted(reference_paths),
                }
            )
            continue
        confirmed_elements = _confirmed(elements)
        retained_confirmed_elements = _confirmed(retained_elements)
        axes = tuple(
            element for element in elements if element.element_type in _AXIS_TYPES
        )
        confirmed = _confirmed(axes)
        retained_axes = tuple(
            element
            for element in retained_elements
            if element.element_type in _AXIS_TYPES
        )
        retained_confirmed_axes = _confirmed(retained_axes)
        broad_pairs = _rota_predicted_loci(
            neutral,
            elements,
            row["chiral_type"],
        )
        confirmed_pairs = _rota_predicted_loci(
            neutral,
            confirmed_elements,
            row["chiral_type"],
        )
        retained_pairs = _rota_predicted_loci(
            molecule,
            retained_elements,
            row["chiral_type"],
        )
        retained_confirmed_pairs = _rota_predicted_loci(
            molecule,
            retained_confirmed_elements,
            row["chiral_type"],
        )
        if invariant is not None:
            projection_issues = _rota_projection_issues(
                neutral,
                elements,
                row["chiral_type"],
                broad_pairs,
                confirmed_pairs,
            )
            issues.extend(projection_issues)
            invariant = invariant and not projection_issues
        if retained_invariant is not None:
            retained_projection_issues = _rota_projection_issues(
                molecule,
                retained_elements,
                row["chiral_type"],
                retained_pairs,
                retained_confirmed_pairs,
            )
            retained_issues.extend(retained_projection_issues)
            retained_invariant = retained_invariant and not retained_projection_issues
        broad_paths = {
            _path_text(element.support.path)
            for element in axes
            if element.element_type is StereoElementType.CUMULENE_AXIS
        }
        confirmed_paths = {
            _path_text(element.support.path)
            for element in confirmed
            if element.element_type is StereoElementType.CUMULENE_AXIS
        }
        retained_paths = {
            _path_text(element.support.path)
            for element in retained_axes
            if element.element_type is StereoElementType.CUMULENE_AXIS
        }
        retained_confirmed_paths = {
            _path_text(element.support.path)
            for element in retained_confirmed_axes
            if element.element_type is StereoElementType.CUMULENE_AXIS
        }
        results.append(
            {
                "id": identifier,
                "chiral_type": row["chiral_type"],
                "parse_failure": False,
                "detection_error": None,
                "reference_pairs": sorted(reference),
                "broad_predicted_pairs": sorted(broad_pairs),
                "confirmed_predicted_pairs": sorted(confirmed_pairs),
                "broad_missed_pairs": sorted(reference - broad_pairs),
                "confirmed_missed_pairs": sorted(reference - confirmed_pairs),
                "local_neighbor_orientation_retained_predicted_pairs": sorted(
                    retained_pairs
                ),
                (
                    "local_neighbor_orientation_retained_confirmed_" "predicted_pairs"
                ): sorted(retained_confirmed_pairs),
                "local_neighbor_orientation_retained_missed_pairs": sorted(
                    reference - retained_pairs
                ),
                "local_neighbor_orientation_retained_additional_pairs": sorted(
                    retained_pairs - reference
                ),
                "reference_expanded_paths": sorted(reference_paths),
                "broad_predicted_expanded_paths": sorted(broad_paths),
                "confirmed_predicted_expanded_paths": sorted(confirmed_paths),
                "broad_missed_expanded_paths": sorted(reference_paths - broad_paths),
                "confirmed_missed_expanded_paths": sorted(
                    reference_paths - confirmed_paths
                ),
                (
                    "local_neighbor_orientation_retained_predicted_" "expanded_paths"
                ): sorted(retained_paths),
                (
                    "local_neighbor_orientation_retained_confirmed_"
                    "predicted_expanded_paths"
                ): sorted(retained_confirmed_paths),
                (
                    "local_neighbor_orientation_retained_missed_" "expanded_paths"
                ): sorted(reference_paths - retained_paths),
                "detected_axes": [_serialize_element(element) for element in axes],
                "detected_carriers": [
                    _serialize_element(element) for element in elements
                ],
                "local_neighbor_orientation_retained_detected_carriers": [
                    _serialize_element(element) for element in retained_elements
                ],
                "renumbering_invariant": invariant,
                "renumbering_issues": issues,
                "local_neighbor_orientation_retained_renumbering_invariant": (
                    retained_invariant
                ),
                "local_neighbor_orientation_retained_renumbering_issues": (
                    retained_issues
                ),
            }
        )
        if position % 100 == 0:
            print(f"RotA detection: {position}/{len(rows)}", file=sys.stderr)
    parsed = [record for record in results if not record["parse_failure"]]
    valid = [record for record in parsed if record["detection_error"] is None]
    all_reference_loci = sum(len(record["reference_pairs"]) for record in parsed)
    all_reference_paths = sum(
        len(record["reference_expanded_paths"]) for record in parsed
    )
    by_type = {}
    for chiral_type in sorted({record["chiral_type"] for record in valid}):
        subset = [record for record in valid if record["chiral_type"] == chiral_type]
        by_type[chiral_type] = {
            "broad": _set_score(
                subset,
                reference_field="reference_pairs",
                prediction_field="broad_predicted_pairs",
            ),
            "confirmed": _set_score(
                subset,
                reference_field="reference_pairs",
                prediction_field="confirmed_predicted_pairs",
            ),
            "local_neighbor_orientation_retained": _set_score(
                subset,
                reference_field="reference_pairs",
                prediction_field=(
                    "local_neighbor_orientation_retained_predicted_pairs"
                ),
            ),
        }
    retained_pair_field = "local_neighbor_orientation_retained_predicted_pairs"
    return (
        {
            "task": ("two-setting positive typed stereo-carrier locus detection"),
            "dataset_records": len(rows),
            "parsed_records": len(parsed),
            "evaluated_records": len(valid),
            "parse_failure_ids": [
                record["id"] for record in results if record["parse_failure"]
            ],
            "detection_errors": [
                {"id": record["id"], "error": record["detection_error"]}
                for record in parsed
                if record["detection_error"] is not None
            ],
            "case_timeout_seconds": timeout_seconds,
            "audited_sha256": ROTA_SHA256,
            "dataset_reference_loci": all_reference_loci,
            "dataset_expanded_reference_paths": all_reference_paths,
            "broad": _set_score(
                valid,
                reference_field="reference_pairs",
                prediction_field="broad_predicted_pairs",
            ),
            "confirmed": _set_score(
                valid,
                reference_field="reference_pairs",
                prediction_field="confirmed_predicted_pairs",
            ),
            "broad_expanded_paths": _set_score(
                valid,
                reference_field="reference_expanded_paths",
                prediction_field="broad_predicted_expanded_paths",
            ),
            "confirmed_expanded_paths": _set_score(
                valid,
                reference_field="reference_expanded_paths",
                prediction_field="confirmed_predicted_expanded_paths",
            ),
            "annotation_shape": {
                "two_atom_loci": sum(
                    "-" in locus
                    for record in parsed
                    for locus in record["reference_pairs"]
                ),
                "single_atom_loci": sum(
                    "-" not in locus
                    for record in parsed
                    for locus in record["reference_pairs"]
                ),
            },
            "detected_axes_by_family": _prediction_counts(valid, "detected_axes"),
            "detected_carriers_by_family": _prediction_counts(
                valid, "detected_carriers"
            ),
            "by_chiral_type": by_type,
            "renumbering": _renumbering_summary(valid),
            "two_setting_comparison": {
                "configuration_erased": _setting_summary(
                    valid,
                    reference_field="reference_pairs",
                    prediction_field="broad_predicted_pairs",
                    category_field="chiral_type",
                ),
                "local_neighbor_orientation_retained": _setting_summary(
                    valid,
                    reference_field="reference_pairs",
                    prediction_field=retained_pair_field,
                    category_field="chiral_type",
                ),
            },
            "configuration_erased_unrecovered_classification": (
                _erased_miss_classification(
                    valid,
                    reference_field="reference_pairs",
                    erased_prediction_field="broad_predicted_pairs",
                    retained_prediction_field=retained_pair_field,
                    category_field="chiral_type",
                )
            ),
            "local_neighbor_orientation_retained_expanded_paths": (
                _set_score(
                    valid,
                    reference_field="reference_expanded_paths",
                    prediction_field=(
                        "local_neighbor_orientation_retained_predicted_"
                        "expanded_paths"
                    ),
                )
            ),
            "local_neighbor_orientation_retained_renumbering": {
                "checked_records": sum(
                    record[
                        ("local_neighbor_orientation_retained_" "renumbering_invariant")
                    ]
                    is not None
                    for record in valid
                ),
                "invariant_records": sum(
                    record[
                        ("local_neighbor_orientation_retained_" "renumbering_invariant")
                    ]
                    is True
                    for record in valid
                ),
                "failures": [
                    {
                        "id": record["id"],
                        "issues": record[
                            (
                                "local_neighbor_orientation_retained_"
                                "renumbering_issues"
                            )
                        ],
                    }
                    for record in valid
                    if record[
                        ("local_neighbor_orientation_retained_" "renumbering_invariant")
                    ]
                    is False
                ],
            },
            "records": results,
            "seconds": time.perf_counter() - started,
            "claim_boundary": (
                "RotA is positive-only. Annotation precision is agreement "
                "with "
                "the supplied locus list, not true-negative specificity. No "
                "handedness or configurational-stability claim is made. "
                "Axis families use extended ring-axis candidates; chiral "
                "atom-pair and spiro/spiral records use their typed topology "
                "projections instead of being forced through an axis-only "
                "scorer. Both erased and local-neighbor-orientation-retained "
                "settings are reported independently."
            ),
        },
        constitution_keys,
    )


def _label_group(descriptor: str) -> str | None:
    if descriptor in {"R", "S", "r", "s"}:
        return "tetrahedral"
    if descriptor in {"E", "Z", "e", "z"}:
        return "planar"
    if descriptor in {"M", "P", "m", "p"}:
        return "axial"
    return None


def _cip_reference_positions(record: dict[str, Any]) -> set[str]:
    positions = set()
    for label in record["recommended_labels"]:
        match = _LABEL_PATTERN.fullmatch(str(label))
        if match is None:
            continue
        position, descriptor = match.groups()
        group = _label_group(descriptor)
        if group is not None:
            positions.add(f"{group}:{position}")
    return positions


def _cip_reference_positions_by_case(
    record: dict[str, Any],
) -> tuple[set[str], set[str]]:
    upper = set()
    lower = set()
    for label in record["recommended_labels"]:
        match = _LABEL_PATTERN.fullmatch(str(label))
        if match is None:
            continue
        position, descriptor = match.groups()
        group = _label_group(descriptor)
        if group is None:
            continue
        target = upper if descriptor.isupper() else lower
        target.add(f"{group}:{position}")
    return upper, lower


def _cip_prediction_positions(
    elements: Iterable[PotentialStereoElement],
) -> set[str]:
    positions = set()
    for element in elements:
        support = element.support
        if element.element_type is StereoElementType.TETRAHEDRAL:
            positions.add(f"tetrahedral:{support.center + 1}")
        elif element.element_type in {
            StereoElementType.DOUBLE_BOND,
            StereoElementType.EXTENDED_CIS_TRANS,
        }:
            positions.update(f"planar:{position + 1}" for position in support.endpoints)
        elif element.element_type in _AXIS_TYPES:
            positions.update(f"axial:{position + 1}" for position in support.endpoints)
        elif element.element_type is StereoElementType.HELICAL:
            positions.update(
                f"axial:{position + 1}"
                for position in (support.path[0], support.path[-1])
            )
    return positions


def _unit_categories(record: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(
        value.strip()
        for value in str(record["stereo_units"]).split(",")
        if value.strip()
    )
    return values or ("none",)


def _molecule_with_local_orientation(smiles: str) -> Chem.Mol | None:
    """Parse input without RDKit deleting locally supplied orientation."""
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        return None
    operations = (
        Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
    )
    try:
        Chem.SanitizeMol(molecule, sanitizeOps=operations)
    except (ValueError, RuntimeError):
        return None
    Chem.SetBondStereoFromDirections(molecule)
    return molecule


def _cip_molecule_with_local_orientation(smiles: str) -> Chem.Mol | None:
    """Compatibility wrapper for the integrity tests and CIP benchmark."""
    return _molecule_with_local_orientation(smiles)


def _benchmark_cip(
    path: Path,
    *,
    check_renumbering: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    rows = load_cip(path)
    results = []
    constitution_keys = []
    id_to_smiles = {}
    started = time.perf_counter()
    for position, row in enumerate(rows, start=1):
        identifier = str(row["ID"])
        smiles = str(row["SMILES"])
        id_to_smiles[identifier] = smiles
        molecule = _cip_molecule_with_local_orientation(smiles)
        if molecule is None:
            results.append({"id": identifier, "parse_failure": True})
            continue
        neutral = neutralize_rdkit_configuration(molecule)
        constitution_keys.append(
            Chem.MolToSmiles(neutral, canonical=True, isomericSmiles=False)
        )
        reference = sorted(_cip_reference_positions(row))
        reference_upper, reference_lower = _cip_reference_positions_by_case(row)
        try:
            elements, invariant, issues = _detect(
                molecule,
                check_renumbering=check_renumbering,
                timeout_seconds=timeout_seconds,
            )
            assisted_elements, assisted_invariant, assisted_issues = _detect(
                molecule,
                check_renumbering=check_renumbering,
                timeout_seconds=timeout_seconds,
                retain_neighbor_configuration=True,
            )
        except _DetectionTimeout:
            results.append(
                {
                    "id": identifier,
                    "stereo_units": list(_unit_categories(row)),
                    "parse_failure": False,
                    "detection_error": "case_timeout",
                    "reference_positions": reference,
                    "reference_uppercase_positions": sorted(reference_upper),
                    "reference_lowercase_positions": sorted(reference_lower),
                    "recommended_labels": sorted(row["recommended_labels"]),
                }
            )
            continue
        confirmed = _confirmed(elements)
        broad_positions = _cip_prediction_positions(elements)
        confirmed_positions = _cip_prediction_positions(confirmed)
        assisted_positions = _cip_prediction_positions(assisted_elements)
        reference_set = set(reference)
        results.append(
            {
                "id": identifier,
                "stereo_units": list(_unit_categories(row)),
                "parse_failure": False,
                "detection_error": None,
                "reference_positions": reference,
                "reference_uppercase_positions": sorted(reference_upper),
                "reference_lowercase_positions": sorted(reference_lower),
                "recommended_labels": sorted(row["recommended_labels"]),
                "broad_predicted_positions": sorted(broad_positions),
                "confirmed_predicted_positions": sorted(confirmed_positions),
                "broad_missed_positions": sorted(reference_set - broad_positions),
                "confirmed_missed_positions": sorted(
                    reference_set - confirmed_positions
                ),
                "neighbor_configuration_assisted_predicted_positions": sorted(
                    assisted_positions
                ),
                "neighbor_configuration_assisted_missed_positions": sorted(
                    reference_set - assisted_positions
                ),
                "neighbor_configuration_assisted_additional_positions": sorted(
                    assisted_positions - reference_set
                ),
                "configuration_erased_tetrahedral_miss_details": (
                    _erased_tetrahedral_miss_details(
                        molecule,
                        neutral,
                        reference_set - broad_positions,
                        one_based=True,
                    )
                ),
                "detected_carriers": [
                    _serialize_element(element) for element in elements
                ],
                "neighbor_configuration_assisted_detected_carriers": [
                    _serialize_element(element) for element in assisted_elements
                ],
                "renumbering_invariant": invariant,
                "renumbering_issues": issues,
                "neighbor_configuration_assisted_renumbering_invariant": (
                    assisted_invariant
                ),
                "neighbor_configuration_assisted_renumbering_issues": (assisted_issues),
            }
        )
        if position % 100 == 0:
            print(f"CIP detection: {position}/{len(rows)}", file=sys.stderr)
    parsed = [record for record in results if not record["parse_failure"]]
    valid = [record for record in parsed if record["detection_error"] is None]
    all_reference = sum(len(record["reference_positions"]) for record in parsed)
    by_unit = {}
    categories = sorted(
        {category for record in valid for category in record["stereo_units"]}
    )
    for category in categories:
        subset = [record for record in valid if category in record["stereo_units"]]
        by_unit[category] = {
            "broad": _set_score(
                subset,
                reference_field="reference_positions",
                prediction_field="broad_predicted_positions",
            ),
            "confirmed": _set_score(
                subset,
                reference_field="reference_positions",
                prediction_field="confirmed_predicted_positions",
            ),
            "neighbor_configuration_assisted": _set_score(
                subset,
                reference_field="reference_positions",
                prediction_field=(
                    "neighbor_configuration_assisted_predicted_positions"
                ),
            ),
        }
    by_reference_class = {
        group: {
            mode: _reference_recall(
                valid,
                reference_field="reference_positions",
                prediction_field=prediction_field,
                prefix=f"{group}:",
            )
            for mode, prediction_field in (
                ("broad", "broad_predicted_positions"),
                ("confirmed", "confirmed_predicted_positions"),
                (
                    "neighbor_configuration_assisted",
                    "neighbor_configuration_assisted_predicted_positions",
                ),
            )
        }
        for group in ("tetrahedral", "planar", "axial")
    }
    by_reference_case = {
        case: {
            mode: _reference_recall(
                valid,
                reference_field=f"reference_{case}_positions",
                prediction_field=prediction_field,
            )
            for mode, prediction_field in (
                ("broad", "broad_predicted_positions"),
                ("confirmed", "confirmed_predicted_positions"),
                (
                    "neighbor_configuration_assisted",
                    "neighbor_configuration_assisted_predicted_positions",
                ),
            )
        }
        for case in ("uppercase", "lowercase")
    }
    return (
        {
            "task": (
                "strict configuration-neutral and oriented-neighbor-frame-"
                "assisted recovery of all local reference positions"
            ),
            "dataset_records": len(rows),
            "parsed_records": len(parsed),
            "evaluated_records": len(valid),
            "parse_failure_ids": [
                record["id"] for record in results if record["parse_failure"]
            ],
            "detection_errors": [
                {"id": record["id"], "error": record["detection_error"]}
                for record in parsed
                if record["detection_error"] is not None
            ],
            "case_timeout_seconds": timeout_seconds,
            "audited_sha256": CIP_SHA256,
            "structures_vendored": False,
            "dataset_reference_positions": all_reference,
            "broad": _set_score(
                valid,
                reference_field="reference_positions",
                prediction_field="broad_predicted_positions",
            ),
            "confirmed": _set_score(
                valid,
                reference_field="reference_positions",
                prediction_field="confirmed_predicted_positions",
            ),
            "neighbor_configuration_assisted": _set_score(
                valid,
                reference_field="reference_positions",
                prediction_field=(
                    "neighbor_configuration_assisted_predicted_positions"
                ),
            ),
            "detected_carriers_by_family": _prediction_counts(
                valid,
                "detected_carriers",
            ),
            "local_neighbor_orientation_retained_carriers_by_family": (
                _prediction_counts(
                    valid,
                    "neighbor_configuration_assisted_detected_carriers",
                )
            ),
            "by_reference_class": by_reference_class,
            "by_reference_case": by_reference_case,
            "two_setting_comparison": {
                "configuration_erased": _setting_summary(
                    valid,
                    reference_field="reference_positions",
                    prediction_field="broad_predicted_positions",
                ),
                "local_neighbor_orientation_retained": _setting_summary(
                    valid,
                    reference_field="reference_positions",
                    prediction_field=(
                        "neighbor_configuration_assisted_predicted_positions"
                    ),
                ),
            },
            "configuration_erased_unrecovered_classification": (
                _erased_miss_classification(
                    valid,
                    reference_field="reference_positions",
                    erased_prediction_field="broad_predicted_positions",
                    retained_prediction_field=(
                        "neighbor_configuration_assisted_predicted_positions"
                    ),
                )
            ),
            "configuration_erased_miss_structure": (_erased_miss_detail_summary(valid)),
            "by_stereo_unit": by_unit,
            "renumbering": _renumbering_summary(valid),
            "neighbor_configuration_assisted_renumbering": {
                "checked_records": sum(
                    record["neighbor_configuration_assisted_renumbering_invariant"]
                    is not None
                    for record in valid
                ),
                "invariant_records": sum(
                    record["neighbor_configuration_assisted_renumbering_invariant"]
                    is True
                    for record in valid
                ),
                "failures": [
                    {
                        "id": record["id"],
                        "issues": record[
                            ("neighbor_configuration_assisted_" "renumbering_issues")
                        ],
                    }
                    for record in valid
                    if record["neighbor_configuration_assisted_renumbering_invariant"]
                    is False
                ],
            },
            "records": results,
            "seconds": time.perf_counter() - started,
            "claim_boundary": (
                "CIP labels supply local reference positions. Configuration "
                "is erased for the strict neutral score. The separately "
                "reported neighbor-configuration-assisted score restricts "
                "graph automorphisms by oriented neighbor frames supplied at "
                "other local elements. It does not convert those frames to "
                "CIP or canonical local descriptors. The focal center's own "
                "frame and all CIP properties remain excluded. This does not "
                "score configuration assignment or global chirality."
            ),
        },
        constitution_keys,
        id_to_smiles,
    )


def benchmark_full_detection(
    cip_path: Path,
    *,
    check_renumbering: bool = True,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run all records from ACS, RotA, and the integrity-checked CIP suite."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = time.perf_counter()
    _DETECTION_CACHE.clear()
    acs, acs_keys = _benchmark_acs(
        check_renumbering=check_renumbering,
        timeout_seconds=timeout_seconds,
    )
    rota, rota_keys = _benchmark_rota(
        check_renumbering=check_renumbering,
        timeout_seconds=timeout_seconds,
    )
    cip, cip_keys, cip_smiles = _benchmark_cip(
        cip_path,
        check_renumbering=check_renumbering,
        timeout_seconds=timeout_seconds,
    )
    acs_rows = load_dataset()
    acs_smiles = {row["ID"]: row["Input SMILES"] for row in acs_rows}
    overlap = set(acs_smiles) & set(cip_smiles)
    exact_overlap = sum(
        acs_smiles[identifier] == cip_smiles[identifier] for identifier in overlap
    )
    all_keys = {*acs_keys, *rota_keys, *cip_keys}
    return {
        "schema": "synkit.full-stereo-carrier-detection/4",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "protocol": {
            "configuration_evidence": "removed_for_strict_neutral_scores",
            "all_datasets_local_neighbor_orientation_retained": (
                "oriented_neighbor_frames_at_other_local_elements"
            ),
            "two_setting_comparison": {
                "setting_1": ("retain_configuration_as_local_neighbor_orientation"),
                "setting_2": "erase_all_configuration",
            },
            "tetrahedral_stereogenicity_criterion": (
                "no_allowed_center_stabilizer_automorphism_with_odd_slot_action"
            ),
            "neighbor_orientation_representation": (
                "ordered_local_frames_without_cip_or_canonical_descriptors"
            ),
            "focal_local_frame_and_cip_properties": "excluded",
            "cip_parse": "sanitize_without_cleanup_chirality",
            "dataset_sampling": "none",
            "renumbering_checked": check_renumbering,
            "rota_extended_ring_axis_candidates": True,
            "rota_family_typed_projection": True,
            "case_timeout_seconds": timeout_seconds,
            "indexing": {
                "serialized_carrier_supports": "zero_based",
                "acs_supplied_loci": "zero_based",
                "rota_annotations": "source_zero_based",
                "rota_generated_record_ids": "one_based",
                "cip_reference_positions": "source_one_based",
            },
        },
        "coverage": {
            "dataset_rows": 258 + 650 + 300,
            "acs_rows": 258,
            "rota_rows": 650,
            "cip_rows": 300,
            "unique_configuration_neutral_constitutions": len(all_keys),
            "acs_cip_shared_ids": len(overlap),
            "acs_cip_identical_smiles": exact_overlap,
            "pooled_accuracy_reported": False,
        },
        "datasets": {
            "acs_stereomolgraph_molecular_chirality": acs,
            "chiralfinder_rota": rota,
            "cip_validation_suite": cip,
        },
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "All 1,208 dataset rows are evaluated without sampling, but the "
            "three task-specific scores are not pooled. ACS is an exact "
            "258-record subset of CIP by ID and SMILES. The designed 40-case "
            "matrix remains a separate API conformance dataset."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-renumbering", action="store_true")
    parser.add_argument("--case-timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    report = benchmark_full_detection(
        arguments.cip_path,
        check_renumbering=not arguments.skip_renumbering,
        timeout_seconds=arguments.case_timeout_seconds,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "coverage": report["coverage"],
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
