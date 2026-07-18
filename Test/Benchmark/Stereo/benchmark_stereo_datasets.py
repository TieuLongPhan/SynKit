#!/usr/bin/env python3
"""Run task-aware molecule-stereo benchmarks from the central registry."""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Test.Chem.Molecule.benchmark_molecular_chirality import (  # noqa: E402
    load_dataset,
)
from synkit.Chem.Molecule.chirality import (  # noqa: E402
    MolecularChiralityOutcome,
    assess_molecular_chirality,
    classify_molecular_chirality,
    clear_molecular_chirality_cache,
)

STEREO_ROOT = ROOT / "Data" / "Benchmark" / "Stereo"
ROTA = STEREO_ROOT / "ChiralFinder-RotA" / "RotA.xlsx"
ROTA_SHA256 = "141ae5281c034f8d08b900454fc387556b5c8702b3e2347d7141ddbd69c4daff"
CIP_METADATA = STEREO_ROOT / "CIPValidationSuite" / "metadata.json"
CIP_SHA256 = "df178635c00b6c41fad820d2609fc4ff18403c63dec4e5c3e1756a6db5858059"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class _BenchmarkTimeout(TimeoutError):
    """Raised when one diagnostic case exceeds its declared time budget."""


class _case_time_limit:
    """Bound one benchmark case on POSIX while restoring signal state."""

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
        raise _BenchmarkTimeout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_error_type(error: Exception) -> str:
    return (
        "case_timeout" if isinstance(error, _BenchmarkTimeout) else type(error).__name__
    )


def _classification_summary(
    rows: list[dict[str, str]],
    predictions: list[str],
) -> dict[str, Any]:
    disagreements = [
        row["ID"]
        for row, prediction in zip(rows, predictions)
        if prediction != row["manual"]
    ]
    return {
        "correct": len(rows) - len(disagreements),
        "accuracy": (len(rows) - len(disagreements)) / len(rows),
        "predictions": dict(sorted(Counter(predictions).items())),
        "disagreement_ids": disagreements,
    }


def _benchmark_acs(max_isomers: int) -> dict[str, Any]:
    rows = load_dataset()
    molecules = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected ACS case {row['ID']}")
        molecules.append(molecule)

    started = time.perf_counter()
    synkit_predictions = [
        classify_molecular_chirality(molecule).classification.value
        for molecule in molecules
    ]
    normal_seconds = time.perf_counter() - started
    normal = {
        "protocol": "supplied_stereo_binary",
        "manual_labels": dict(sorted(Counter(row["manual"] for row in rows).items())),
        "synkit": {
            **_classification_summary(rows, synkit_predictions),
            "seconds": normal_seconds,
        },
        "published_stereomolgraph": _classification_summary(
            rows,
            [row["StereoMolGraph"] for row in rows],
        ),
        "published_rdkit_smiles": _classification_summary(
            rows,
            [row["RDKit SMILES"] for row in rows],
        ),
        "published_inchi": _classification_summary(
            rows,
            [row["InChI"] for row in rows],
        ),
        "published_chython": _classification_summary(
            rows,
            [row["chython"] for row in rows],
        ),
    }

    grouped: dict[str, tuple[Chem.Mol, list[dict[str, str]]]] = {}
    for row, molecule in zip(rows, molecules):
        stripped = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(stripped)
        key = Chem.MolToSmiles(stripped, canonical=True, isomericSmiles=False)
        if key not in grouped:
            grouped[key] = stripped, []
        grouped[key][1].append(row)

    clear_molecular_chirality_cache()
    started = time.perf_counter()
    assessments = {
        key: assess_molecular_chirality(molecule, max_isomers=max_isomers)
        for key, (molecule, _members) in grouped.items()
    }
    stripped_seconds = time.perf_counter() - started
    row_outcomes: Counter[str] = Counter()
    definitive = 0
    manual_covered = 0
    configuration_dependent_ids = []
    incomplete_cases = []
    for key, (_molecule, members) in grouped.items():
        assessment = assessments[key]
        row_outcomes[assessment.outcome.value] += len(members)
        if assessment.is_definitive:
            definitive += len(members)
        observed = {
            classification.value
            for classification in assessment.observed_classifications
        }
        manual_covered += sum(row["manual"] in observed for row in members)
        identifiers = [row["ID"] for row in members]
        if assessment.outcome is MolecularChiralityOutcome.CONFIGURATION_DEPENDENT:
            configuration_dependent_ids.extend(identifiers)
        if assessment.outcome is MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE:
            incomplete_cases.append(
                {
                    "ids": identifiers,
                    "constitutional_smiles": key,
                    "theoretical_isomer_upper_bound": (
                        assessment.theoretical_isomer_upper_bound
                    ),
                    "evaluated_isomer_count": (assessment.evaluated_isomer_count),
                    "observed_classifications": sorted(observed),
                    "unsupported_stereo_loci": (assessment.unsupported_stereo_loci),
                }
            )

    stripped = {
        "protocol": "stereo_stripped_four_state",
        "max_isomers": max_isomers,
        "unique_constitutions": len(grouped),
        "row_outcomes": dict(sorted(row_outcomes.items())),
        "definitive_rows": definitive,
        "manual_label_in_observed_population": manual_covered,
        "configuration_dependent_ids": sorted(configuration_dependent_ids),
        "incomplete_cases": incomplete_cases,
        "seconds": stripped_seconds,
        "interpretation": (
            "Coverage asks whether the erased manual label remains among the "
            "enumerated possibilities; it is not recovery accuracy."
        ),
    }
    return {
        "records": len(rows),
        "task": "whole_molecule_chiral_achiral",
        "normal": normal,
        "stereo_removed": stripped,
    }


def load_rota(path: Path = ROTA) -> list[dict[str, str]]:
    """Load the MIT-licensed RotA worksheet without an Excel dependency."""
    digest = _sha256(path)
    if digest != ROTA_SHA256:
        raise ValueError(f"Unexpected RotA SHA-256: {digest}")
    with ZipFile(path) as workbook:
        root = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.findall(f".//{XLSX_NS}row"):
        values = []
        for cell in row.findall(f"{XLSX_NS}c"):
            inline = cell.find(f"{XLSX_NS}is/{XLSX_NS}t")
            scalar = cell.find(f"{XLSX_NS}v")
            values.append(
                inline.text
                if inline is not None
                else (scalar.text if scalar is not None else "")
            )
        rows.append(values)
    header, *records = rows
    parsed = [dict(zip(header, values)) for values in records]
    if len(parsed) != 650:
        raise ValueError(f"Expected 650 RotA records, found {len(parsed)}")
    return parsed


def load_cip(path: Path) -> list[dict[str, Any]]:
    """Load an external CIP suite checkout after exact-integrity validation."""
    digest = _sha256(path)
    if digest != CIP_SHA256:
        raise ValueError(f"Unexpected CIP Validation Suite SHA-256: {digest}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split("\t")
        if len(fields) < 4:
            raise ValueError(f"Malformed CIP row {line_number}")
        fields.extend([""] * (6 - len(fields)))
        rows.append(
            {
                "SMILES": fields[0],
                "ID": fields[1],
                "recommended_labels": tuple(fields[2].split()),
                "reference": fields[3],
                "stereo_units": fields[4],
                "rules": fields[5],
            }
        )
    if len(rows) != 300:
        raise ValueError(f"Expected 300 CIP records, found {len(rows)}")
    if len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("CIP record identifiers are not unique")
    return rows


def _two_protocol_diagnostic(
    records: list[dict[str, Any]],
    *,
    max_isomers: int,
    case_timeout_seconds: float,
    progress_label: str,
) -> dict[str, Any]:
    """Run both global protocols without treating local labels as truth."""
    molecules: list[tuple[str, Chem.Mol]] = []
    parse_failures = []
    for record in records:
        identifier = str(record["ID"])
        molecule = Chem.MolFromSmiles(str(record["SMILES"]))
        if molecule is None:
            parse_failures.append(identifier)
        else:
            molecules.append((identifier, molecule))

    original_predictions: Counter[str] = Counter()
    original_statuses: Counter[str] = Counter()
    original_errors = []
    started = time.perf_counter()
    for position, (identifier, molecule) in enumerate(molecules, start=1):
        try:
            with _case_time_limit(case_timeout_seconds):
                result = classify_molecular_chirality(molecule)
        except Exception as error:  # pragma: no cover - fail-closed path
            original_errors.append(
                {"id": identifier, "error_type": _benchmark_error_type(error)}
            )
            continue
        original_predictions[result.classification.value] += 1
        original_statuses[result.input_stereo_status] += 1
        if position % 100 == 0:
            print(
                f"{progress_label} original: {position}/{len(molecules)}",
                file=sys.stderr,
                flush=True,
            )
    original_seconds = time.perf_counter() - started

    grouped: dict[str, tuple[Chem.Mol, list[str]]] = {}
    for identifier, molecule in molecules:
        stripped = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(stripped)
        key = Chem.MolToSmiles(stripped, canonical=True, isomericSmiles=False)
        if key not in grouped:
            grouped[key] = stripped, []
        grouped[key][1].append(identifier)

    clear_molecular_chirality_cache()
    row_outcomes: Counter[str] = Counter()
    incomplete_cases = []
    stripped_errors = []
    started = time.perf_counter()
    for position, (molecule, identifiers) in enumerate(grouped.values(), start=1):
        try:
            with _case_time_limit(case_timeout_seconds):
                assessment = assess_molecular_chirality(
                    molecule,
                    max_isomers=max_isomers,
                )
        except Exception as error:  # pragma: no cover - fail-closed path
            stripped_errors.extend(
                {
                    "id": identifier,
                    "error_type": _benchmark_error_type(error),
                }
                for identifier in identifiers
            )
            continue
        row_outcomes[assessment.outcome.value] += len(identifiers)
        if assessment.outcome is MolecularChiralityOutcome.UNSUPPORTED_OR_INCOMPLETE:
            incomplete_cases.append(
                {
                    "ids": identifiers,
                    "theoretical_isomer_upper_bound": (
                        assessment.theoretical_isomer_upper_bound
                    ),
                    "evaluated_isomer_count": (assessment.evaluated_isomer_count),
                    "observed_classifications": sorted(
                        value.value for value in assessment.observed_classifications
                    ),
                    "unsupported_stereo_loci": (assessment.unsupported_stereo_loci),
                }
            )
        if position % 100 == 0:
            print(
                f"{progress_label} stripped: {position}/{len(grouped)}",
                file=sys.stderr,
                flush=True,
            )
    stripped_seconds = time.perf_counter() - started

    diagnostic_note = (
        "Diagnostic distribution only: this dataset's reference annotations "
        "are not whole-molecule chiral/achiral labels, so accuracy is "
        "undefined."
    )
    return {
        "rdkit_parse_failures": parse_failures,
        "normal": {
            "executed": True,
            "protocol": "supplied_stereo_binary",
            "case_timeout_seconds": case_timeout_seconds,
            "evaluated_records": len(molecules) - len(original_errors),
            "predictions": dict(sorted(original_predictions.items())),
            "input_stereo_status": dict(sorted(original_statuses.items())),
            "errors": original_errors,
            "seconds": original_seconds,
            "reference_accuracy": None,
            "interpretation": diagnostic_note,
        },
        "stereo_removed": {
            "executed": True,
            "protocol": "stereo_stripped_four_state",
            "max_isomers": max_isomers,
            "case_timeout_seconds": case_timeout_seconds,
            "unique_constitutions": len(grouped),
            "evaluated_records": len(molecules) - len(stripped_errors),
            "row_outcomes": dict(sorted(row_outcomes.items())),
            "incomplete_cases": incomplete_cases,
            "errors": stripped_errors,
            "seconds": stripped_seconds,
            "reference_accuracy": None,
            "interpretation": diagnostic_note,
        },
    }


def _audit_rota(
    max_isomers: int = 256,
    case_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    rows = load_rota()
    specified_stereo_rows = 0
    removal_changed_rows = 0
    axis_counts = []
    for index, row in enumerate(rows, start=1):
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            continue
        before = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        if any(
            info.specified == Chem.StereoSpecified.Specified
            for info in Chem.FindPotentialStereo(molecule)
        ):
            specified_stereo_rows += 1
        Chem.RemoveStereochemistry(molecule)
        after = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        removal_changed_rows += before != after
        axis_counts.append(len(literal_eval(row["label"])))
    records = [
        {"ID": f"RotA-{index:04d}", "SMILES": row["SMILES"]}
        for index, row in enumerate(rows, start=1)
    ]
    diagnostic = _two_protocol_diagnostic(
        records,
        max_isomers=max_isomers,
        case_timeout_seconds=case_timeout_seconds,
        progress_label="RotA",
    )
    return {
        "records": len(rows),
        "task": "axial_stereo_locus_detection",
        "chiral_type_counts": dict(
            sorted(Counter(row["chiral_type"] for row in rows).items())
        ),
        "axis_count_distribution": dict(sorted(Counter(axis_counts).items())),
        "rows_with_rdkit_specified_stereo": specified_stereo_rows,
        "rows_changed_by_stereo_removal": removal_changed_rows,
        "reference_accuracy_applicable": False,
        "task_boundary": (
            "RotA labels axial atom-pair/chain loci in a positive-only "
            "corpus; "
            "its labels are not global chiral/achiral truth."
        ),
        **diagnostic,
    }


def _benchmark_cip(
    path: Path,
    max_isomers: int,
    case_timeout_seconds: float,
) -> dict[str, Any]:
    rows = load_cip(path)
    stereo_unit_counts = Counter(
        unit.strip()
        for row in rows
        for unit in str(row["stereo_units"]).split(",")
        if unit.strip()
    )
    diagnostic = _two_protocol_diagnostic(
        rows,
        max_isomers=max_isomers,
        case_timeout_seconds=case_timeout_seconds,
        progress_label="CIP",
    )
    return {
        "records": len(rows),
        "task": "local_cip_descriptor_assignment",
        "vendored": False,
        "benchmark_run": True,
        "source_integrity": {"sha256": CIP_SHA256},
        "stereo_unit_counts": dict(sorted(stereo_unit_counts.items())),
        "reference_accuracy_applicable": False,
        "task_boundary": (
            "CIP annotations are local R/S/r/s/M/P/m/p/E/Z assignments, not "
            "whole-molecule chiral/achiral truth."
        ),
        **diagnostic,
    }


def benchmark(
    max_isomers: int = 256,
    *,
    cip_path: Path | None = None,
    case_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Return the task-aware registry benchmark report."""
    cip = json.loads(CIP_METADATA.read_text(encoding="utf-8"))
    return {
        "schema": "synkit.stereo-dataset-benchmark/1",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "datasets": {
            "acs_stereomolgraph_molecular_chirality": _benchmark_acs(max_isomers),
            "chiralfinder_rota": _audit_rota(
                max_isomers,
                case_timeout_seconds,
            ),
            "cip_validation_suite": (
                _benchmark_cip(
                    cip_path,
                    max_isomers,
                    case_timeout_seconds,
                )
                if cip_path is not None
                else {
                    "records": cip["records"],
                    "task": "local_cip_descriptor_assignment",
                    "vendored": False,
                    "benchmark_run": False,
                    "reason": cip["reason"],
                }
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-isomers", type=int, default=256)
    parser.add_argument("--case-timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--cip-path",
        type=Path,
        help="external compounds.smi checkout (validated by SHA-256)",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.max_isomers < 1:
        parser.error("max-isomers must be positive")
    if arguments.case_timeout_seconds <= 0:
        parser.error("case-timeout-seconds must be positive")
    RDLogger.DisableLog("rdApp.*")
    report = benchmark(
        arguments.max_isomers,
        cip_path=arguments.cip_path,
        case_timeout_seconds=arguments.case_timeout_seconds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
