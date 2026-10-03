#!/usr/bin/env python3
"""Canonicalize factorized joint assignments for mixed-stereo ABC cases.

Formal mode canonicalizes one representative from every joint formal class.
Raw mode canonicalizes the complete Cartesian product of local representations
and verifies that they collapse consistently into those formal classes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import product
import json
from math import prod
from pathlib import Path
import sys
import time
from typing import Any

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Canonicalization.configuration_free_local import (  # noqa: E402
    _seed_from_potential,
    _stereo_neutral,
)
from Experiment.Stereo.Canonicalization.global_local import (  # noqa: E402
    _case_time_limit,
)
from Experiment.Stereo.Canonicalization.local_permutations import (  # noqa: E402
    all_local_arrangements,
)
from Experiment.Stereo.Canonicalization.independent_model import (  # noqa: E402
    from_descriptor, raw_representation_key, raw_states, state_key,
)
from Experiment.Stereo.Canonicalization.independent_orbits import (  # noqa: E402
    partition_agreement, reference_partition,
)
from Experiment.Stereo.provenance import capture_source, finish_source  # noqa: E402
from Experiment.Stereo.Canonicalization.exact_code_store import ExactCodeStore  # noqa: E402
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
    detect_potential_stereo_elements,
)
from synkit.Graph.Stereo import (  # noqa: E402
    canonicalize_configured_registry,
    descriptor_id,
)
from synkit.Graph.Stereo.canonical import (  # noqa: E402
    _rdkit_graph_and_registry,
)
from synkit.Graph.Stereo.enumeration import (  # noqa: E402
    _fixed_seed,
    _rdkit_identifier_map,
)

from Experiment.Stereo.Canonicalization.configuration_free_cases import (  # noqa: E402
    CASES_BY_BUDGET, CaseSpec, _BUDGETS,
)
from Experiment.Stereo.Canonicalization.configuration_free_cases import _CASES as _CASES  # noqa: E402,F401

_ENUMERATION_MODES = ("formal", "raw")
_TABLE_FIELDS = (
    "budget",
    "enumeration_mode",
    "source",
    "record_id",
    "carrier_composition",
    "stereo_carriers",
    "configuration_dependent_supports",
    "full_raw_representation_product",
    "joint_formal_assignments_expected",
    "enumerated_assignments_expected",
    "enumerated_assignments_completed",
    "formal_assignment_tuples_covered",
    "formal_assignment_tuples_invariant",
    "observed_global_canonical_classes",
    "formal_assignments_per_global_class",
    "raw_assignments_per_global_class",
    "representation_invariance_passed",
    "independent_partition_passed",
    "timeouts",
    "passed",
    "seconds",
    "issues",
)


def _carrier_inventory(
    molecule: Chem.Mol,
) -> tuple[Any, tuple[dict[str, Any], ...], dict[str, int]]:
    """Build configuration-free carriers while preserving known supports."""
    _input_graph, supplied = _rdkit_graph_and_registry(molecule)
    neutral = _stereo_neutral(molecule)
    graph, erased = _rdkit_graph_and_registry(neutral)
    identifiers = _rdkit_identifier_map(neutral)
    potential_elements = detect_potential_stereo_elements(neutral)
    entries: dict[str, dict[str, Any]] = {}
    for descriptor in supplied.values():
        seed = _fixed_seed(descriptor)
        key = descriptor_id(seed)
        entries[key] = {
            "carrier_id": key,
            "seed": seed,
            "support_sources": {"supplied_input"},
            "carrier_status": "supplied_support",
        }
    for element in potential_elements:
        seed = _seed_from_potential(element, neutral, identifiers)
        key = descriptor_id(seed)
        if key not in entries:
            entries[key] = {
                "carrier_id": key,
                "seed": seed,
                "support_sources": {"neutral_perception"},
                "carrier_status": element.carrier_status.value,
            }
        else:
            entries[key]["support_sources"].add("neutral_perception")
            entries[key]["carrier_status"] = element.carrier_status.value

    carriers = []
    for key in sorted(entries):
        entry = entries[key]
        arrangements = all_local_arrangements(entry["seed"])
        grouped: dict[str, list[Any]] = defaultdict(list)
        for arrangement in arrangements:
            grouped[repr(state_key(from_descriptor(arrangement)))].append(arrangement)
        actual_raw = {raw_representation_key(from_descriptor(value)) for value in arrangements}
        expected_raw = {raw_representation_key(value) for value in raw_states(from_descriptor(entry["seed"]))}
        class_keys = sorted(grouped)
        sources = tuple(sorted(entry["support_sources"]))
        carriers.append(
            {
                "carrier_id": key,
                "family": entry["seed"].descriptor_class,
                "carrier_status": entry["carrier_status"],
                "support_sources": sources,
                "configuration_dependent_support": sources == ("supplied_input",),
                "raw_local_permutations": len(arrangements),
                "raw_inventory_exhaustive": actual_raw == expected_raw and len(actual_raw) == len(arrangements),
                "formal_configuration_classes": len(grouped),
                "formal_class_multiplicities": [
                    len(grouped[class_key]) for class_key in class_keys
                ],
                "class_representatives": tuple(
                    grouped[class_key][0] for class_key in class_keys
                ),
                "raw_representations": tuple(
                    descriptor
                    for class_key in class_keys
                    for descriptor in grouped[class_key]
                ),
                "raw_formal_indices": tuple(
                    class_index
                    for class_index, class_key in enumerate(class_keys)
                    for _descriptor in grouped[class_key]
                ),
            }
        )
    return (
        graph,
        tuple(carriers),
        {
            "supplied_supports": len(supplied),
            "neutral_potential_supports": len(potential_elements),
            "configurations_after_erasure": len(erased),
        },
    )


def _canonicalize_assignment(
    graph: Any,
    registry: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[str | None, str | None, float]:
    started = time.perf_counter()
    try:
        with _case_time_limit(timeout_seconds):
            result = canonicalize_configured_registry(
                graph,
                registry,
                enumerate_automorphism_group=False,
            )
    except TimeoutError:
        return None, "timeout", time.perf_counter() - started
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return (
            None,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
        )
    if not result.complete or not result.exact:
        return None, "non_exact_or_incomplete_result", time.perf_counter() - started
    return result.canonical_code, None, time.perf_counter() - started


def _composition_text(composition: tuple[tuple[str, int], ...]) -> str:
    return ";".join(f"{family}:{count}" for family, count in composition)


def _duration_text(seconds: float) -> str:
    """Render progress durations compactly without hiding long raw runs."""
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{final_seconds:02d}"
    return f"{minutes:d}:{final_seconds:02d}"


class _RawProgress:
    """Emit bounded, worker-safe progress lines for one raw benchmark case."""

    def __init__(
        self,
        spec: CaseSpec,
        *,
        total: int,
        formal_total: int,
        interval_seconds: float | None,
    ) -> None:
        self._label = f"raw {spec.budget} {spec.source}:{spec.record_id}"
        self._total = total
        self._formal_total = formal_total
        self._interval_seconds = interval_seconds
        self._started = time.perf_counter()
        self._last_report = self._started

    @property
    def enabled(self) -> bool:
        return self._interval_seconds is not None and self._interval_seconds > 0

    def start(self, *, carriers: int) -> None:
        if not self.enabled:
            return
        print(
            f"[{self._label}] starting {self._total:,} assignments "
            f"({self._formal_total:,} formal tuples; "
            f"{carriers} carriers)",
            flush=True,
        )

    def update(
        self,
        *,
        attempted: int,
        successful: int,
        failures: int,
        formal_covered: int,
    ) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        finished = attempted >= self._total
        if not finished and now - self._last_report < float(self._interval_seconds):
            return
        elapsed = max(now - self._started, 1e-12)
        rate = attempted / elapsed
        remaining = max(self._total - attempted, 0)
        eta = remaining / rate if rate > 0 else 0.0
        percent = 100.0 * attempted / self._total
        print(
            f"[{self._label}] "
            f"{attempted:,}/{self._total:,} ({percent:6.2f}%) | "
            f"ok={successful:,} failed={failures:,} | "
            f"formal={formal_covered:,}/{self._formal_total:,} | "
            f"rate={rate:,.2f}/s elapsed={_duration_text(elapsed)} "
            f"ETA={_duration_text(eta)}",
            flush=True,
        )
        self._last_report = now


def _case_validation(spec, composition, raw_product, formal_count, max_assignments, carriers):
    """Check frozen input composition and independent enumeration coverage."""
    issues = []
    if not all(carrier["raw_inventory_exhaustive"] for carrier in carriers):
        issues.append("raw_inventory_not_independently_exhaustive")
    if composition != spec.composition:
        issues.append(
            f"composition_mismatch:{_composition_text(composition)}"
        )
    if raw_product != spec.raw_product:
        issues.append(
            f"raw_product_mismatch:{raw_product}!={spec.raw_product}"
        )
    if formal_count != spec.formal_assignments:
        issues.append(
            f"formal_count_mismatch:{formal_count}!={spec.formal_assignments}"
        )
    if len(composition) < 2:
        issues.append("not_mixed_family")
    if formal_count > max_assignments:
        issues.append(f"assignment_limit:{formal_count}>{max_assignments}")
    return issues


def _case_result(
    item: tuple[CaseSpec, str, str, float, int, float | None],
) -> dict[str, Any]:
    """Bound full-certificate storage for formal and raw Cartesian products."""
    with ExactCodeStore() as store:
        return _case_result_exact(item, store)


def _case_result_exact(
    item: tuple[CaseSpec, str, str, float, int, float | None], code_store: ExactCodeStore,
) -> dict[str, Any]:
    (
        spec,
        smiles,
        enumeration_mode,
        timeout_seconds,
        max_assignments,
        progress_interval_seconds,
    ) = item
    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            "budget": spec.budget,
            "enumeration_mode": enumeration_mode,
            "source": spec.source,
            "record_id": spec.record_id,
            "status": "parse_failure",
            "passed": False,
            "issues": ["parse_failure"],
            "carriers": [],
            "seconds": time.perf_counter() - started,
        }

    try:
        graph, carriers, inventory = _carrier_inventory(molecule)
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return {
            "budget": spec.budget,
            "enumeration_mode": enumeration_mode,
            "source": spec.source,
            "record_id": spec.record_id,
            "status": "carrier_inventory_failed",
            "passed": False,
            "issues": [f"{type(error).__name__}: {error}"],
            "carriers": [],
            "seconds": time.perf_counter() - started,
        }

    composition = tuple(sorted(Counter(c["family"] for c in carriers).items()))
    raw_product = prod(c["raw_local_permutations"] for c in carriers)
    formal_count = prod(c["formal_configuration_classes"] for c in carriers)
    validation_issues = _case_validation(spec, composition, raw_product, formal_count, max_assignments, carriers)
    public_carriers = [
        {
            key: value
            for key, value in carrier.items()
            if key != "class_representatives"
            and key != "raw_representations"
            and key != "raw_formal_indices"
        }
        for carrier in carriers
    ]
    if validation_issues:
        return {
            "budget": spec.budget,
            "enumeration_mode": enumeration_mode,
            "source": spec.source,
            "record_id": spec.record_id,
            "status": "validation_failed",
            "passed": False,
            "issues": validation_issues,
            "carrier_composition": _composition_text(composition),
            "stereo_carriers": len(carriers),
            "configuration_dependent_supports": sum(
                carrier["configuration_dependent_support"] for carrier in carriers
            ),
            "full_raw_representation_product": raw_product,
            "joint_formal_assignments_expected": formal_count,
            "enumerated_assignments_expected": (
                raw_product if enumeration_mode == "raw" else formal_count
            ),
            "enumerated_assignments_completed": 0,
            "formal_assignment_tuples_covered": 0,
            "formal_assignment_tuples_invariant": 0,
            "observed_global_canonical_classes": 0,
            "formal_assignments_per_global_class": [],
            "raw_assignments_per_global_class": [],
            "representation_invariance_passed": False,
            "timeouts": 0,
            "carriers": public_carriers,
            "inventory": inventory,
            "global_classes": [],
            "seconds": time.perf_counter() - started,
        }

    if enumeration_mode == "raw":
        enumeration_choices = tuple(
            carrier["raw_representations"] for carrier in carriers
        )
        enumeration_to_formal = tuple(
            carrier["raw_formal_indices"] for carrier in carriers
        )
        enumerated_expected = raw_product
    else:
        enumeration_choices = tuple(
            carrier["class_representatives"] for carrier in carriers
        )
        enumeration_to_formal = tuple(
            tuple(range(len(choices))) for choices in enumeration_choices
        )
        enumerated_expected = formal_count

    digest_raw_counts: Counter[int] = Counter()
    formal_digest_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)
    formal_enumeration_counts: Counter[tuple[int, ...]] = Counter()
    assignment_issues = []
    completed = 0
    canonical_seconds = 0.0
    progress = _RawProgress(
        spec,
        total=enumerated_expected,
        formal_total=formal_count,
        interval_seconds=(
            progress_interval_seconds if enumeration_mode == "raw" else None
        ),
    )
    progress.start(carriers=len(carriers))
    for assignment_index, enumeration_indices in enumerate(
        product(*(range(len(choices)) for choices in enumeration_choices))
    ):
        registry = {
            carrier["carrier_id"]: enumeration_choices[position][choice_index]
            for position, (carrier, choice_index) in enumerate(
                zip(carriers, enumeration_indices)
            )
        }
        formal_indices = tuple(
            enumeration_to_formal[position][choice_index]
            for position, choice_index in enumerate(enumeration_indices)
        )
        digest, issue, duration = _canonicalize_assignment(
            graph,
            registry,
            timeout_seconds=timeout_seconds,
        )
        canonical_seconds += duration
        if issue is not None or digest is None:
            assignment_issues.append(
                {
                    "assignment_index": assignment_index,
                    "enumeration_indices": list(enumeration_indices),
                    "formal_class_indices": list(formal_indices),
                    "issue": issue or "missing_result",
                }
            )
        else:
            digest = code_store.intern(digest)
            completed += 1
            digest_raw_counts[digest] += 1
            formal_digest_sets[formal_indices].add(digest)
            formal_enumeration_counts[formal_indices] += 1
        progress.update(
            attempted=assignment_index + 1,
            successful=completed,
            failures=len(assignment_issues),
            formal_covered=len(formal_digest_sets),
        )

    expected_formal_tuples = tuple(
        product(
            *(range(carrier["formal_configuration_classes"]) for carrier in carriers)
        )
    )
    formal_expected_counts = {
        formal_indices: (
            prod(
                carrier["formal_class_multiplicities"][class_index]
                for carrier, class_index in zip(carriers, formal_indices)
            )
            if enumeration_mode == "raw"
            else 1
        )
        for formal_indices in expected_formal_tuples
    }
    invariant_formal_tuples = {
        formal_indices
        for formal_indices in expected_formal_tuples
        if len(formal_digest_sets[formal_indices]) == 1
        and formal_enumeration_counts[formal_indices]
        == formal_expected_counts[formal_indices]
    }
    digest_formal_tuples: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for formal_indices in expected_formal_tuples:
        digests = formal_digest_sets[formal_indices]
        if len(digests) == 1:
            digest_formal_tuples[next(iter(digests))].append(formal_indices)

    global_classes = []
    for index, digest in enumerate(sorted(digest_raw_counts)):
        formal_tuples = sorted(digest_formal_tuples.get(digest, ()))
        represented_raw = sum(
            prod(
                carrier["formal_class_multiplicities"][class_index]
                for carrier, class_index in zip(carriers, formal_indices)
            )
            for formal_indices in formal_tuples
        )
        global_classes.append(
            {
                "global_class_index": index,
                "canonical_digest": code_store.index_digest(digest),
                "formal_assignment_multiplicity": len(formal_tuples),
                "formal_assignments": [
                    list(formal_indices) for formal_indices in formal_tuples
                ],
                "enumerated_assignment_multiplicity": digest_raw_counts[digest],
                "represented_raw_assignment_multiplicity": represented_raw,
            }
        )
    issues = [entry["issue"] for entry in assignment_issues]
    complete = completed == enumerated_expected and not issues
    formal_multiplicities = sorted(
        result["formal_assignment_multiplicity"] for result in global_classes
    )
    raw_multiplicities = sorted(
        result["represented_raw_assignment_multiplicity"] for result in global_classes
    )
    representation_invariance = len(invariant_formal_tuples) == formal_count
    reference = reference_partition(
        graph, tuple(carrier["class_representatives"] for carrier in carriers),
        max_assignments=max_assignments, timeout_seconds=timeout_seconds,
    )
    full_codes = [next(iter(formal_digest_sets[indices])) if len(formal_digest_sets[indices]) == 1 else None
                  for indices in expected_formal_tuples]
    independent_passed = partition_agreement(reference, expected_formal_tuples, full_codes)
    if not independent_passed:
        failure = "independent_partition_disagreement" if reference["status"] == "complete" else "oracle_incomplete"
        issues.append(failure)
    passed = (
        complete
        and bool(global_classes)
        and representation_invariance
        and independent_passed
        and sum(formal_multiplicities) == formal_count
        and sum(raw_multiplicities) == raw_product
    )
    return {
        "budget": spec.budget,
        "enumeration_mode": enumeration_mode,
        "source": spec.source,
        "record_id": spec.record_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "issues": issues,
        "carrier_composition": _composition_text(composition),
        "stereo_carriers": len(carriers),
        "configuration_dependent_supports": sum(
            carrier["configuration_dependent_support"] for carrier in carriers
        ),
        "full_raw_representation_product": raw_product,
        "joint_formal_assignments_expected": formal_count,
        "enumerated_assignments_expected": enumerated_expected,
        "enumerated_assignments_completed": completed,
        "formal_assignment_tuples_covered": len(formal_digest_sets),
        "formal_assignment_tuples_invariant": len(invariant_formal_tuples),
        "observed_global_canonical_classes": len(global_classes),
        "formal_assignments_per_global_class": formal_multiplicities,
        "raw_assignments_per_global_class": raw_multiplicities,
        "representation_invariance_passed": representation_invariance,
        "independent_partition_passed": independent_passed,
        "independent_reference": reference,
        "timeouts": sum(issue == "timeout" for issue in issues),
        "canonicalization_seconds": canonical_seconds,
        "carriers": public_carriers,
        "inventory": inventory,
        "assignment_issues": assignment_issues,
        "global_classes": global_classes,
        "seconds": time.perf_counter() - started,
    }


def _selected_rows(
    specs: tuple[CaseSpec, ...],
    *,
    acs_path: Path,
    cip_path: Path | None,
    rota_path: Path,
) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    needed = {case.source for case in specs}
    rows: dict[tuple[str, str], str] = {}
    integrity: dict[str, Any] = {}
    if "acs" in needed:
        loaded = load_dataset(acs_path)
        rows.update(
            {("acs", str(row["ID"])): str(row["Input SMILES"]) for row in loaded}
        )
        integrity["acs"] = {
            "path": str(acs_path),
            "audited_sha256": ACS_SHA256,
        }
    if "cip" in needed:
        if cip_path is None:
            raise ValueError("This budget requires --cip-path.")
        loaded = load_cip(cip_path)
        rows.update({("cip", str(row["ID"])): str(row["SMILES"]) for row in loaded})
        integrity["cip"] = {
            "path": str(cip_path),
            "audited_sha256": CIP_SHA256,
        }
    if "rota" in needed:
        loaded = load_rota(rota_path)
        rows.update(
            {
                ("rota", f"RotA-{index:04d}"): str(row["SMILES"])
                for index, row in enumerate(loaded)
            }
        )
        integrity["rota"] = {
            "path": str(rota_path),
            "audited_sha256": ROTA_SHA256,
        }
    return rows, integrity


def _run_cases(
    items: list[tuple[CaseSpec, str, str, float, int, float | None]],
    *,
    jobs: int,
) -> list[dict[str, Any]]:
    if jobs == 1:
        return [_case_result(item) for item in items]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_case_result, items))


def benchmark_budget(
    budget: str,
    *,
    enumeration_mode: str = "formal",
    acs_path: Path = ACS_DATASET,
    cip_path: Path | None = None,
    rota_path: Path = ROTA,
    case_ids: tuple[str, ...] = (),
    limit: int | None = None,
    jobs: int = 1,
    timeout_seconds: float = 10.0,
    max_assignments: int = 4096,
    progress_interval_seconds: float | None = None,
) -> dict[str, Any]:
    source_provenance = capture_source()
    budget = budget.upper()
    if budget not in CASES_BY_BUDGET:
        raise ValueError(f"Unknown budget: {budget}.")
    if enumeration_mode not in _ENUMERATION_MODES:
        raise ValueError(f"Unknown enumeration mode: {enumeration_mode}.")
    specs = CASES_BY_BUDGET[budget]
    known_ids = {case.record_id for case in specs}
    missing = sorted(set(case_ids) - known_ids)
    if missing:
        raise ValueError(f"Unknown {budget} case identifiers: {missing}")
    selected = tuple(
        case for case in specs if not case_ids or case.record_id in case_ids
    )
    if limit is not None:
        selected = selected[:limit]
    rows, integrity = _selected_rows(
        selected,
        acs_path=acs_path,
        cip_path=cip_path,
        rota_path=rota_path,
    )
    absent = [
        f"{case.source}:{case.record_id}"
        for case in selected
        if (case.source, case.record_id) not in rows
    ]
    if absent:
        raise ValueError(f"Selected dataset rows are absent: {absent}")

    started = time.perf_counter()
    records = _run_cases(
        [
            (
                case,
                rows[(case.source, case.record_id)],
                enumeration_mode,
                timeout_seconds,
                max_assignments,
                progress_interval_seconds,
            )
            for case in selected
        ],
        jobs=jobs,
    )
    formal_expected = sum(
        record.get("joint_formal_assignments_expected", 0) for record in records
    )
    enumerated_expected = sum(
        record.get("enumerated_assignments_expected", 0) for record in records
    )
    enumerated_completed = sum(
        record.get("enumerated_assignments_completed", 0) for record in records
    )
    return {
        "schema": "synkit.configuration-free-global-abc/2",
        "provenance": finish_source(source_provenance),
        "budget": budget,
        "enumeration_mode": enumeration_mode,
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "source_integrity": integrity,
        "protocol": {
            "mixed_family_cases_only": True,
            "fixed_molecular_constitution": True,
            "supplied_configurations_used": False,
            "known_input_supports_retained": True,
            "neutral_potential_supports_merged": True,
            "local_raw_orbits_factorized": True,
            "all_carriers_configured_simultaneously": True,
            "joint_formal_assignment_product": "exhaustive",
            "local_class_truth": "independent_geometry_rotation_orbits",
            "global_partition_truth": "exhaustive_inventory_preserving_constitutional_automorphisms",
            "identity_comparison": "full_exact_codes; digests_are_indices_only",
            "oracle_incomplete_is_pass": False,
            "enumeration_mode": enumeration_mode,
            "raw_representation_cartesian_product_enumerated": (
                enumeration_mode == "raw"
            ),
            "timeout_seconds_per_assignment": timeout_seconds,
            "max_assignments_per_case": max_assignments,
            "worker_processes": jobs,
            "progress_interval_seconds": progress_interval_seconds,
        },
        "summary": {
            "cases_selected": len(records),
            "cases_passed": sum(record["passed"] for record in records),
            "cases_failed": sum(not record["passed"] for record in records),
            "record_statuses": dict(
                sorted(Counter(record["status"] for record in records).items())
            ),
            "stereo_carriers": sum(
                record.get("stereo_carriers", 0) for record in records
            ),
            "configuration_dependent_supports": sum(
                record.get("configuration_dependent_supports", 0) for record in records
            ),
            "largest_full_raw_representation_product": max(
                (
                    record.get("full_raw_representation_product", 0)
                    for record in records
                ),
                default=0,
            ),
            "joint_formal_assignments_expected": formal_expected,
            "enumerated_assignments_expected": enumerated_expected,
            "enumerated_assignments_completed": enumerated_completed,
            "formal_assignment_tuples_invariant": sum(
                record.get("formal_assignment_tuples_invariant", 0)
                for record in records
            ),
            "observed_global_canonical_classes": sum(
                record.get("observed_global_canonical_classes", 0) for record in records
            ),
            "timeouts": sum(record.get("timeouts", 0) for record in records),
            "complete": enumerated_completed == enumerated_expected,
        },
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Formal mode canonicalizes one representative of every joint "
            "formal configuration. Raw mode canonicalizes the complete raw "
            "representation Cartesian product and verifies its collapse into "
            "the same formal tuples."
        ),
    }


def _write_table(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in report["records"]:
            row = {}
            for field in _TABLE_FIELDS:
                value = record.get(field, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, list):
                    value = ";".join(map(str, value))
                row[field] = value
            writer.writerow(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget", choices=_BUDGETS)
    parser.add_argument(
        "--enumeration-mode",
        choices=_ENUMERATION_MODES,
        default="formal",
    )
    parser.add_argument("--acs-path", type=Path, default=ACS_DATASET)
    parser.add_argument("--cip-path", type=Path)
    parser.add_argument("--rota-path", type=Path, default=ROTA)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-assignments", type=int, default=4096)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help=("raw-mode terminal progress interval; use 0 to disable " "(default: 10)"),
    )
    parser.add_argument(
        "--allow-expensive-raw",
        action="store_true",
        help=(
            "explicitly authorize raw B/C Cartesian-product stress runs; "
            "formal B/C are the maintained scientific gates"
        ),
    )
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
    if arguments.max_assignments < 1:
        parser.error("--max-assignments must be positive")
    if arguments.progress_interval < 0:
        parser.error("--progress-interval cannot be negative")
    if (
        arguments.enumeration_mode == "raw"
        and arguments.budget in {"B", "C"}
        and not arguments.allow_expensive_raw
    ):
        parser.error(
            "raw B/C are optional Cartesian-product stress runs, not "
            "maintained benchmark gates; pass --allow-expensive-raw to "
            "authorize the cost explicitly"
        )
    try:
        report = benchmark_budget(
            arguments.budget,
            enumeration_mode=arguments.enumeration_mode,
            acs_path=arguments.acs_path,
            cip_path=arguments.cip_path,
            rota_path=arguments.rota_path,
            case_ids=tuple(arguments.case_id or ()),
            limit=arguments.limit,
            jobs=arguments.jobs,
            timeout_seconds=arguments.timeout,
            max_assignments=arguments.max_assignments,
            progress_interval_seconds=(
                arguments.progress_interval
                if arguments.enumeration_mode == "raw"
                else None
            ),
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
                "budget": arguments.budget,
                "enumeration_mode": arguments.enumeration_mode,
                "output": str(arguments.output),
                "table": str(arguments.table),
                "summary": report["summary"],
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not report["summary"]["cases_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
