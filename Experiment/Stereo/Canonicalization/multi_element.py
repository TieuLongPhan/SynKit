#!/usr/bin/env python3
"""Audit composition of multiple configured stereo elements.

The synthetic tier enumerates complete binary assignments on designed
two- and three-element molecular graphs.  Dataset tiers have deliberately
different semantics:

* ACS supplies configured whole molecules and global chiral/achiral labels;
* RotA supplies positive axial loci, usually without handedness;
* CIP supplies local labels and carrier positions, not global chirality.

No structure from the externally registered CIP suite is copied into the
frozen report.
"""

from __future__ import annotations

from ast import literal_eval
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
import json
import multiprocessing
from pathlib import Path
from queue import Empty
import random
import signal
import sys
import time
from typing import Any

import networkx as nx
from rdkit import Chem
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    CIP_SHA256,
    ROTA,
    ROTA_SHA256,
    load_cip,
    load_rota,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    DATASET,
    EXPECTED_SHA256,
    load_dataset,
)
from synkit.Graph.Stereo import (  # noqa: E402
    canonicalize_configured_registry,
    canonicalize_rdkit_configured_stereograph,
    classify_rdkit_configured_stereograph_mirror,
    descriptors_from_rdkit,
    mirror_configured_descriptor,
)
from synkit.Graph.Stereo.canonical import (  # noqa: E402
    _rdkit_graph_and_registry,
)

STEREO_ROOT = ROOT / "Experiment" / "Stereo" / "Data"
ROTA_LOCUS_REPORT = STEREO_ROOT / "Perception" / "rota_locus_report.json"
CIP_ELEMENT_REPORT = STEREO_ROOT / "Perception" / "stereo_element_report.json"

_DEFAULT_ACS_CASE_IDS = (
    "VS066",  # two tetrahedral, chiral
    "VS219",  # two tetrahedral, achiral
    "VS058",  # two planar bonds
    "VS161",  # tetrahedral plus planar bond
    "VS230",  # three tetrahedral, achiral
    "VS034",  # three tetrahedral, chiral
    "VS084",  # four tetrahedral, chiral
)


@dataclass(frozen=True)
class _SyntheticCase:
    name: str
    smiles: str
    expected_elements: int
    expected_assignments: int
    expected_global_classes: int
    expected_mirror_fixed_classes: int
    expected_enantiomer_pairs: int
    expected_diastereomer_pairs: int


_SYNTHETIC_CASES = (
    _SyntheticCase(
        "asymmetric_two_tetrahedral",
        "F[C@H](Cl)[C@H](Br)I",
        2,
        4,
        4,
        0,
        2,
        4,
    ),
    _SyntheticCase(
        "symmetric_two_tetrahedral",
        "C[C@H](O)[C@H](O)C",
        2,
        4,
        3,
        1,
        1,
        2,
    ),
    _SyntheticCase(
        "tetrahedral_plus_planar_bond",
        "F[C@H](Cl)C/C=C/Br",
        2,
        4,
        4,
        0,
        2,
        4,
    ),
    _SyntheticCase(
        "two_planar_bonds",
        "F/C=C/C=C/Cl",
        2,
        4,
        4,
        4,
        0,
        6,
    ),
    _SyntheticCase(
        "three_asymmetric_tetrahedral",
        "F[C@H](Cl)[C@H](Br)[C@H](I)N",
        3,
        8,
        8,
        0,
        4,
        24,
    ),
    _SyntheticCase(
        "tetrahedral_plus_two_planar_bonds",
        "F[C@H](Cl)C/C=C/C=C/Br",
        3,
        8,
        8,
        0,
        4,
        24,
    ),
)


class _CaseTimeout(TimeoutError):
    pass


class _case_time_limit:
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
        raise _CaseTimeout


def _timed(
    operation: Callable[[], Any],
    *,
    timeout_seconds: float,
) -> tuple[Any | None, str | None, float]:
    started = time.perf_counter()
    try:
        with _case_time_limit(timeout_seconds):
            result = operation()
    except _CaseTimeout:
        return None, "case_timeout", time.perf_counter() - started
    except Exception as error:
        return (
            None,
            f"{type(error).__name__}:{error}",
            time.perf_counter() - started,
        )
    return result, None, time.perf_counter() - started


def _orders(size: int, *, count: int, seed: int) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        return ()
    base = tuple(range(size))
    candidates = [
        base[1:] + base[:1],
        base[2:] + base[:2],
        tuple(reversed(base)),
    ]
    generator = random.Random(seed)
    while len(candidates) < max(12, count * 3):
        values = list(base)
        generator.shuffle(values)
        candidates.append(tuple(values))
    unique = []
    for order in candidates:
        if order != base and order not in unique:
            unique.append(order)
        if len(unique) == count:
            break
    return tuple(unique)


def _graph_mapping(
    nodes: tuple[Any, ...],
    order: tuple[int, ...],
) -> dict[Any, Any]:
    return {
        source: nodes[target_position] for source, target_position in zip(nodes, order)
    }


def _family_counts(descriptors: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(item.descriptor_class for item in descriptors).items()))


def _acs_isolated_worker(
    molecule_binary: bytes,
    order: tuple[int, ...] | None,
    expected_code: str | None,
    mirror: bool,
    output: Any,
) -> None:
    """Run one ACS exact call in a terminable child process."""
    try:
        molecule = Chem.Mol(molecule_binary)
        if order is not None:
            molecule = Chem.RenumberAtoms(molecule, order)
        if mirror:
            result = classify_rdkit_configured_stereograph_mirror(molecule)
            payload = {
                "ok": True,
                "status": result.status.value,
            }
        else:
            result = canonicalize_rdkit_configured_stereograph(molecule)
            payload = {
                "ok": True,
                "same": (
                    expected_code is None or result.canonical_code == expected_code
                ),
                "digest": result.canonical_digest,
            }
    except Exception as error:  # pragma: no cover - isolated fail-closed path
        payload = {
            "ok": False,
            "error": f"{type(error).__name__}:{error}",
        }
    output.put(payload)


def _isolated_acs_call(
    molecule: Chem.Mol,
    *,
    order: tuple[int, ...] | None = None,
    expected_code: str | None = None,
    mirror: bool = False,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Return a small exact result while enforcing a hard wall-clock cap."""
    started = time.perf_counter()
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    output = context.Queue()
    process = context.Process(
        target=_acs_isolated_worker,
        args=(
            molecule.ToBinary(),
            order,
            expected_code,
            mirror,
            output,
        ),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        output.close()
        return None, "case_timeout", time.perf_counter() - started
    try:
        payload = output.get(timeout=1.0)
    except Empty:
        payload = None
    finally:
        output.close()
    if payload is None:
        return (
            None,
            f"worker_exit:{process.exitcode}",
            time.perf_counter() - started,
        )
    if not payload.get("ok"):
        return (
            None,
            str(payload.get("error", "worker_failure")),
            time.perf_counter() - started,
        )
    return payload, None, time.perf_counter() - started


def _assignment_registry(
    items: tuple[tuple[str, Any], ...],
    bits: tuple[int, ...],
) -> dict[str, Any]:
    return {
        key: descriptor if bit == 0 else descriptor.invert()
        for bit, (key, descriptor) in zip(bits, items)
    }


def _synthetic_case_audit(
    case: _SyntheticCase,
    *,
    representative_relabelings: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(case.smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected synthetic case {case.name}")
    graph, seed_registry = _rdkit_graph_and_registry(molecule)
    items = tuple(sorted(seed_registry.items()))
    raw = []
    failures = []
    timeouts = 0
    exact_calls = 0
    started = time.perf_counter()
    for bits in product((0, 1), repeat=len(items)):
        registry = _assignment_registry(items, bits)
        result, issue, elapsed = _timed(
            lambda graph=graph, registry=registry: (
                canonicalize_configured_registry(graph, registry)
            ),
            timeout_seconds=timeout_seconds,
        )
        exact_calls += 1
        timeouts += issue == "case_timeout"
        if result is None:
            failures.append(f"assignment:{bits}:{issue}")
        raw.append(
            {
                "bits": bits,
                "registry": registry,
                "result": result,
                "issue": issue,
                "seconds": elapsed,
            }
        )

    representatives: dict[str, dict[str, Any]] = {}
    for record in raw:
        result = record["result"]
        if result is not None:
            representatives.setdefault(result.canonical_code, record)

    nodes = tuple(sorted(graph.nodes, key=repr))
    classes = []
    mirror_pairs: set[tuple[str, str]] = set()
    mirror_codes = set()
    mirror_fixed = 0
    registry_order_checks = 0
    relabeling_checks = 0
    double_mirror_checks = 0
    for class_index, (code, record) in enumerate(representatives.items()):
        registry = record["registry"]
        baseline = record["result"]
        registry_order_invariant = None
        if class_index == 0:
            reversed_registry = dict(reversed(tuple(registry.items())))
            reordered, reorder_issue, _elapsed = _timed(
                lambda: canonicalize_configured_registry(
                    graph,
                    reversed_registry,
                ),
                timeout_seconds=timeout_seconds,
            )
            exact_calls += 1
            registry_order_checks += 1
            timeouts += reorder_issue == "case_timeout"
            registry_order_invariant = (
                reordered is not None and baseline.same_stereograph(reordered)
            )
            if not registry_order_invariant:
                failures.append(
                    f"class:{class_index}:registry_order:"
                    f"{reorder_issue or 'certificate_mismatch'}"
                )

        relabel_failures = []
        relabel_orders = _orders(
            len(nodes),
            count=representative_relabelings if class_index == 0 else 0,
            seed=4300 + class_index,
        )
        for order in relabel_orders:
            mapping = _graph_mapping(nodes, order)
            relabeled_registry = {
                key: descriptor.relabel(mapping) for key, descriptor in registry.items()
            }
            observed, issue, _elapsed = _timed(
                lambda mapping=mapping, relabeled_registry=relabeled_registry: (
                    canonicalize_configured_registry(
                        nx.relabel_nodes(graph, mapping, copy=True),
                        relabeled_registry,
                    )
                ),
                timeout_seconds=timeout_seconds,
            )
            exact_calls += 1
            relabeling_checks += 1
            timeouts += issue == "case_timeout"
            if observed is None or not baseline.same_stereograph(observed):
                relabel_failures.append(issue or "certificate_mismatch")
        if relabel_failures:
            failures.append(f"class:{class_index}:relabel")

        mirrored_registry = {
            key: mirror_configured_descriptor(descriptor)
            for key, descriptor in registry.items()
        }
        mirrored, mirror_issue, _elapsed = _timed(
            lambda: canonicalize_configured_registry(
                graph,
                mirrored_registry,
            ),
            timeout_seconds=timeout_seconds,
        )
        exact_calls += 1
        timeouts += mirror_issue == "case_timeout"
        if mirrored is None:
            failures.append(f"class:{class_index}:mirror:{mirror_issue}")
            mirror_digest = None
            fixed = None
        else:
            mirror_codes.add(mirrored.canonical_code)
            mirror_digest = mirrored.canonical_digest
            fixed = baseline.same_stereograph(mirrored)
            if fixed:
                mirror_fixed += 1
            else:
                mirror_pairs.add(tuple(sorted((code, mirrored.canonical_code))))

        double_mirror_invariant = None
        if class_index == 0:
            double_registry = {
                key: mirror_configured_descriptor(descriptor)
                for key, descriptor in mirrored_registry.items()
            }
            doubled, double_issue, _elapsed = _timed(
                lambda: canonicalize_configured_registry(
                    graph,
                    double_registry,
                ),
                timeout_seconds=timeout_seconds,
            )
            exact_calls += 1
            double_mirror_checks += 1
            timeouts += double_issue == "case_timeout"
            double_mirror_invariant = doubled is not None and baseline.same_stereograph(
                doubled
            )
            if not double_mirror_invariant:
                failures.append(
                    f"class:{class_index}:double_mirror:"
                    f"{double_issue or 'certificate_mismatch'}"
                )
        classes.append(
            {
                "class_index": class_index,
                "representative_bits": list(record["bits"]),
                "assignment_multiplicity": sum(
                    item["result"] is not None
                    and baseline.same_stereograph(item["result"])
                    for item in raw
                ),
                "canonical_digest": baseline.canonical_digest,
                "registry_order_invariant": registry_order_invariant,
                "relabelings_checked": len(relabel_orders),
                "relabeling_failures": relabel_failures,
                "mirror_digest": mirror_digest,
                "mirror_fixed": fixed,
                "double_mirror_invariant": double_mirror_invariant,
            }
        )

    observed_classes = len(representatives)
    mirror_closed = mirror_codes <= set(representatives)
    enantiomer_pairs = len(mirror_pairs)
    total_pairs = observed_classes * (observed_classes - 1) // 2
    diastereomer_pairs = total_pairs - enantiomer_pairs
    expectations = {
        "configured_elements": (
            len(items),
            case.expected_elements,
        ),
        "raw_assignments": (
            len(raw),
            case.expected_assignments,
        ),
        "global_classes": (
            observed_classes,
            case.expected_global_classes,
        ),
        "mirror_fixed_classes": (
            mirror_fixed,
            case.expected_mirror_fixed_classes,
        ),
        "enantiomer_pairs": (
            enantiomer_pairs,
            case.expected_enantiomer_pairs,
        ),
        "diastereomer_pairs": (
            diastereomer_pairs,
            case.expected_diastereomer_pairs,
        ),
    }
    for name, (observed, expected) in expectations.items():
        if observed != expected:
            failures.append(f"{name}:{observed}:expected:{expected}")
    if not mirror_closed:
        failures.append("mirror_not_closed_over_global_classes")
    return {
        "name": case.name,
        "atoms": molecule.GetNumAtoms(),
        "configured_elements": len(items),
        "families": _family_counts(seed_registry.values()),
        "raw_assignments": len(raw),
        "expected_global_classes": case.expected_global_classes,
        "global_classes": observed_classes,
        "mirror_closed_over_global_classes": mirror_closed,
        "expected_mirror_fixed_classes": (case.expected_mirror_fixed_classes),
        "mirror_fixed_classes": mirror_fixed,
        "expected_enantiomer_pairs": case.expected_enantiomer_pairs,
        "enantiomer_pairs": enantiomer_pairs,
        "expected_diastereomer_pairs": case.expected_diastereomer_pairs,
        "diastereomer_pairs": diastereomer_pairs,
        "registry_order_checks": registry_order_checks,
        "relabelings_checked": relabeling_checks,
        "double_mirror_checks": double_mirror_checks,
        "exact_canonicalizations": exact_calls,
        "timeouts": timeouts,
        "failures": failures,
        "passed": not failures,
        "classes": classes,
        "seconds": time.perf_counter() - started,
    }


def benchmark_synthetic_composition(
    *,
    case_names: tuple[str, ...] = (),
    representative_relabelings: int = 1,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    known = {case.name for case in _SYNTHETIC_CASES}
    unknown = sorted(set(case_names) - known)
    if unknown:
        raise ValueError(f"Unknown synthetic multi-element cases: {unknown}")
    selected = tuple(
        case for case in _SYNTHETIC_CASES if not case_names or case.name in case_names
    )
    records = [
        _synthetic_case_audit(
            case,
            representative_relabelings=representative_relabelings,
            timeout_seconds=timeout_seconds,
        )
        for case in selected
    ]
    return {
        "cases": records,
        "totals": {
            "cases": len(records),
            "raw_assignments": sum(item["raw_assignments"] for item in records),
            "global_classes": sum(item["global_classes"] for item in records),
            "mirror_fixed_classes": sum(
                item["mirror_fixed_classes"] for item in records
            ),
            "enantiomer_pairs": sum(item["enantiomer_pairs"] for item in records),
            "diastereomer_pairs": sum(item["diastereomer_pairs"] for item in records),
            "registry_order_checks": sum(
                item["registry_order_checks"] for item in records
            ),
            "relabelings_checked": sum(item["relabelings_checked"] for item in records),
            "double_mirror_checks": sum(
                item["double_mirror_checks"] for item in records
            ),
            "exact_canonicalizations": sum(
                item["exact_canonicalizations"] for item in records
            ),
            "timeouts": sum(item["timeouts"] for item in records),
            "failed_cases": sum(not item["passed"] for item in records),
        },
        "passed": all(item["passed"] for item in records),
    }


def _acs_inventory(path: Path) -> dict[str, Any]:
    records = []
    for row in load_dataset(path):
        molecule = Chem.MolFromSmiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected ACS case {row['ID']}")
        descriptors = tuple(
            descriptors_from_rdkit(
                molecule,
                require_atom_maps=False,
            ).values()
        )
        records.append(
            {
                "id": row["ID"],
                "atoms": molecule.GetNumAtoms(),
                "manual": row["manual"].lower(),
                "configured_elements": len(descriptors),
                "families": _family_counts(descriptors),
            }
        )
    multi = [record for record in records if record["configured_elements"] >= 2]
    distribution = Counter(record["configured_elements"] for record in records)
    return {
        "dataset": {
            "records": len(records),
            "audited_sha256": EXPECTED_SHA256,
        },
        "configured_element_distribution": {
            str(count): frequency for count, frequency in sorted(distribution.items())
        },
        "multi_element_records": len(multi),
        "max_configured_elements": max(
            record["configured_elements"] for record in records
        ),
        "multi_element_candidates": multi,
    }


def _acs_audit(
    path: Path,
    *,
    case_ids: tuple[str, ...],
    relabelings_per_case: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    rows = {row["ID"]: row for row in load_dataset(path)}
    missing = sorted(set(case_ids) - set(rows))
    if missing:
        raise ValueError(f"Unknown ACS multi-element cases: {missing}")
    records = []
    started = time.perf_counter()
    for case_index, identifier in enumerate(case_ids):
        molecule = Chem.MolFromSmiles(rows[identifier]["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected ACS case {identifier}")
        descriptors = tuple(
            descriptors_from_rdkit(
                molecule,
                require_atom_maps=False,
            ).values()
        )
        baseline, issue, baseline_seconds = _timed(
            lambda: canonicalize_rdkit_configured_stereograph(molecule),
            timeout_seconds=timeout_seconds,
        )
        failures = []
        timeouts = issue == "case_timeout"
        checked = 0
        relabel_seconds = 0.0
        for order in _orders(
            molecule.GetNumAtoms(),
            count=relabelings_per_case,
            seed=8400 + case_index,
        ):
            observed, relabel_issue, elapsed = _isolated_acs_call(
                molecule,
                order=order,
                expected_code=(
                    baseline.canonical_code if baseline is not None else None
                ),
                timeout_seconds=timeout_seconds,
            )
            checked += 1
            relabel_seconds += elapsed
            timeouts += relabel_issue == "case_timeout"
            if baseline is None or observed is None or observed["same"] is not True:
                failures.append(relabel_issue or issue or "certificate_mismatch")
        mirror, mirror_issue, mirror_seconds = _isolated_acs_call(
            molecule,
            mirror=True,
            timeout_seconds=timeout_seconds,
        )
        timeouts += mirror_issue == "case_timeout"
        mirror_status = mirror["status"] if mirror is not None else None
        manual = rows[identifier]["manual"].lower()
        records.append(
            {
                "id": identifier,
                "atoms": molecule.GetNumAtoms(),
                "configured_elements": len(descriptors),
                "families": _family_counts(descriptors),
                "manual_global_label": manual,
                "configured_certificate_digest": (
                    baseline.canonical_digest if baseline is not None else None
                ),
                "baseline_issue": issue,
                "relabelings_checked": checked,
                "relabeling_failures": failures,
                "mirror_status": mirror_status,
                "mirror_issue": mirror_issue,
                "mirror_matches_manual": mirror_status == manual,
                "timeouts": timeouts,
                "passed": (
                    len(descriptors) >= 2
                    and issue is None
                    and not failures
                    and mirror_issue is None
                    and mirror_status == manual
                ),
                "seconds": (baseline_seconds + relabel_seconds + mirror_seconds),
            }
        )
    return {
        "selection_rule": (
            "seven small public cases spanning two to four configured "
            "elements, same-family and mixed-family composition, and both "
            "global labels"
        ),
        "selected_ids": list(case_ids),
        "records": records,
        "totals": {
            "cases": len(records),
            "configured_elements": sum(item["configured_elements"] for item in records),
            "relabelings_checked": sum(item["relabelings_checked"] for item in records),
            "relabeling_failures": sum(
                len(item["relabeling_failures"]) for item in records
            ),
            "mirror_matches_manual": sum(
                item["mirror_matches_manual"] for item in records
            ),
            "timeouts": sum(item["timeouts"] for item in records),
            "failed_cases": sum(not item["passed"] for item in records),
        },
        "passed": all(item["passed"] for item in records),
        "seconds": time.perf_counter() - started,
    }


def _rota_extraction(
    path: Path,
    frozen_report_path: Path,
) -> dict[str, Any]:
    rows = load_rota(path)
    frozen = json.loads(frozen_report_path.read_text(encoding="utf-8"))
    if frozen.get("dataset", {}).get("audited_sha256") != ROTA_SHA256:
        raise ValueError("RotA locus report does not match the audited workbook.")
    frozen_records = {record["id"]: record for record in frozen["records"]}
    candidates = []
    distribution: Counter[int] = Counter()
    for index, row in enumerate(rows):
        identifier = f"RotA-{index:04d}"
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected RotA case {identifier}")
        pairs = {tuple(sorted(pair)) for pair in literal_eval(row["label"])}
        distribution[len(pairs)] += 1
        if len(pairs) < 2:
            continue
        prior = frozen_records[identifier]
        candidates.append(
            {
                "id": identifier,
                "atoms": molecule.GetNumAtoms(),
                "chiral_type": row["chiral_type"],
                "reference_loci": len(pairs),
                "all_reference_loci_recovered": prior["all_reference_pairs_recovered"],
                "exact_predicted_locus_set": prior["exact_pair_set"],
                "renumbering_invariant": prior["renumbering_invariant"],
            }
        )
    return {
        "dataset": {
            "records": len(rows),
            "audited_sha256": ROTA_SHA256,
            "positive_only": True,
        },
        "reference_locus_distribution": {
            str(count): frequency for count, frequency in sorted(distribution.items())
        },
        "multi_locus_records": len(candidates),
        "multi_reference_loci": sum(item["reference_loci"] for item in candidates),
        "multi_records_all_loci_recovered": sum(
            item["all_reference_loci_recovered"] for item in candidates
        ),
        "multi_records_exact_predicted_set": sum(
            item["exact_predicted_locus_set"] for item in candidates
        ),
        "multi_records_renumbering_invariant": sum(
            item["renumbering_invariant"] for item in candidates
        ),
        "multi_locus_candidates": candidates,
        "canonicalization_role": (
            "multi-support extraction and relabeling stress only; RotA does "
            "not generally supply a fixed axial configuration"
        ),
    }


def _cip_structure_inventory(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "reason": (
                "The audited compounds.smi checkout is not present. "
                "Structures are external-only and are not redistributed."
            ),
        }
    records = []
    for record in load_cip(path):
        molecule = Chem.MolFromSmiles(str(record["SMILES"]))
        if molecule is None:
            raise ValueError(f"RDKit rejected CIP case {record['ID']}")
        descriptors = tuple(
            descriptors_from_rdkit(
                molecule,
                require_atom_maps=False,
            ).values()
        )
        records.append(
            {
                "id": str(record["ID"]),
                "atoms": molecule.GetNumAtoms(),
                "configured_elements": len(descriptors),
                "families": _family_counts(descriptors),
            }
        )
    distribution = Counter(record["configured_elements"] for record in records)
    return {
        "available": True,
        "records": len(records),
        "audited_sha256": CIP_SHA256,
        "configured_element_distribution": {
            str(count): frequency for count, frequency in sorted(distribution.items())
        },
        "multi_element_records": sum(
            record["configured_elements"] >= 2 for record in records
        ),
        "multi_element_candidates": [
            record for record in records if record["configured_elements"] >= 2
        ],
    }


def _cip_extraction(
    frozen_report_path: Path,
    *,
    cip_path: Path | None,
) -> dict[str, Any]:
    frozen = json.loads(frozen_report_path.read_text(encoding="utf-8"))
    dataset = frozen.get("dataset", {})
    if dataset.get("records") != 300 or dataset.get("audited_sha256") != CIP_SHA256:
        raise ValueError("CIP element report has unexpected provenance.")
    candidates = []
    attached_distribution: Counter[int] = Counter()
    reference_distribution: Counter[int] = Counter()
    for record in frozen["records"]:
        reference_count = len(record["reference_rs_positions"])
        attached_count = len(record["attached_configurations"])
        reference_distribution[reference_count] += 1
        attached_distribution[attached_count] += 1
        if reference_count < 2:
            continue
        candidates.append(
            {
                "id": record["id"],
                "stereo_units": record["stereo_units"],
                "reference_rs_positions": record["reference_rs_positions"],
                "attached_configurations": record["attached_configurations"],
                "renumbering_invariant": record["renumbering_invariant"],
            }
        )
    attached_multi = [
        record for record in candidates if len(record["attached_configurations"]) >= 2
    ]
    return {
        "dataset": {
            "records": 300,
            "audited_sha256": CIP_SHA256,
            "structures_vendored": False,
        },
        "reference_rs_position_distribution": {
            str(count): frequency
            for count, frequency in sorted(reference_distribution.items())
        },
        "attached_configuration_distribution": {
            str(count): frequency
            for count, frequency in sorted(attached_distribution.items())
        },
        "records_with_multiple_reference_rs_positions": len(candidates),
        "records_with_multiple_attached_configurations": len(attached_multi),
        "max_attached_configurations": max(
            len(record["attached_configurations"]) for record in candidates
        ),
        "multi_attached_renumbering_invariant": sum(
            record["renumbering_invariant"] for record in attached_multi
        ),
        "multi_center_candidates": candidates,
        "structure_inventory": _cip_structure_inventory(cip_path),
        "canonicalization_role": (
            "local multi-carrier and label-projection evidence; CIP labels "
            "are not independent whole-molecule chirality truth"
        ),
    }


def benchmark_multi_element_canonicalization(
    *,
    acs_path: Path = DATASET,
    rota_path: Path = ROTA,
    rota_report_path: Path = ROTA_LOCUS_REPORT,
    cip_element_report_path: Path = CIP_ELEMENT_REPORT,
    cip_path: Path | None = None,
    synthetic_case_names: tuple[str, ...] = (),
    synthetic_relabelings: int = 1,
    acs_case_ids: tuple[str, ...] = _DEFAULT_ACS_CASE_IDS,
    acs_relabelings: int = 2,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    if synthetic_relabelings < 0 or acs_relabelings < 0:
        raise ValueError("Relabeling counts must be non-negative.")
    started = time.perf_counter()
    synthetic = benchmark_synthetic_composition(
        case_names=synthetic_case_names,
        representative_relabelings=synthetic_relabelings,
        timeout_seconds=timeout_seconds,
    )
    acs_inventory = _acs_inventory(acs_path)
    acs_audit = _acs_audit(
        acs_path,
        case_ids=acs_case_ids,
        relabelings_per_case=acs_relabelings,
        timeout_seconds=timeout_seconds,
    )
    return {
        "schema": "synkit.multi-element-canonicalization-benchmark/1",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "task": (
            "exact composition of multiple configured stereo elements and "
            "task-scoped extraction of public multi-element records"
        ),
        "protocol": {
            "synthetic_cases": [
                case.name
                for case in _SYNTHETIC_CASES
                if (not synthetic_case_names or case.name in synthetic_case_names)
            ],
            "synthetic_relabelings_per_case_representative": (synthetic_relabelings),
            "acs_selected_ids": list(acs_case_ids),
            "acs_relabelings_per_case": acs_relabelings,
            "case_timeout_seconds": timeout_seconds,
            "cip_structures_supplied": cip_path is not None,
        },
        "synthetic_composition": synthetic,
        "acs_configured_whole_molecules": {
            "inventory": acs_inventory,
            "selected_exact_audit": acs_audit,
        },
        "rota_multi_locus_extraction": _rota_extraction(
            rota_path,
            rota_report_path,
        ),
        "cip_multi_center_extraction": _cip_extraction(
            cip_element_report_path,
            cip_path=cip_path,
        ),
        "passed": synthetic["passed"] and acs_audit["passed"],
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Synthetic assignments prove composition behavior on six "
            "designed two- and three-element graphs. The selected ACS tier "
            "tests configured public whole molecules and may use its manual "
            "global labels. RotA contributes only positive axial support "
            "annotations. CIP contributes local carrier/R/S evidence and an "
            "ID-only manifest unless the audited external checkout is "
            "provided; neither RotA nor CIP is treated as global chirality "
            "truth."
        ),
    }
