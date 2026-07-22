#!/usr/bin/env python3
"""Compare live molecule-chirality backends on registered stereo datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Test.Benchmark.Stereo.benchmark_stereo_datasets import (  # noqa: E402
    _benchmark_error_type,
    _case_time_limit,
    load_cip,
    load_rota,
)
from Test.Chem.Molecule.benchmark_molecular_chirality import (  # noqa: E402
    load_dataset,
)
from synkit.Chem.Molecule.chirality import (  # noqa: E402
    classify_molecular_chirality,
)

STEREO_ROOT = ROOT / "Data" / "Benchmark" / "Stereo"

CIP_ATROP_BINARY_LABELS = {
    identifier: "Chiral"
    for identifier in (
        "VS023",
        "VS055",
        "VS057",
        "VS072",
        "VS073",
        "VS086",
        "VS158",
    )
}

CIP_CURATED_BINARY_LABELS = {
    "VS007": "Achiral",
    "VS010": "Chiral",
    "VS011": "Chiral",
    "VS022": "Chiral",
    "VS063": "Chiral",
    "VS078": "Chiral",
    "VS079": "Chiral",
    "VS118": "Achiral",
    "VS119": "Chiral",
    "VS120": "Chiral",
    "VS124": "Achiral",
    "VS135": "Achiral",
    "VS141": "Chiral",
    "VS144": "Chiral",
    "VS154": "Achiral",
    "VS155": "Achiral",
    "VS164": "Chiral",
    "VS166": "Chiral",
    "VS175": "Chiral",
    "VS176": "Achiral",
    "VS177": "Chiral",
    "VS178": "Chiral",
    "VS179": "Chiral",
    "VS180": "Chiral",
    "VS181": "Chiral",
    "VS182": "Chiral",
    "VS183": "Chiral",
    "VS184": "Achiral",
    "VS185": "Achiral",
    "VS186": "Chiral",
    "VS187": "Chiral",
    "VS231": "Achiral",
    "VS232": "Achiral",
    "VS243": "Achiral",
    "VS287": "Chiral",
}


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _invert_rdkit_tetrahedral_tags(molecule: Chem.Mol) -> Chem.Mol:
    """Reproduce the publisher's RDKit-SMILES mirror construction."""
    mirror = Chem.RWMol(molecule)
    for atom in mirror.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return mirror.GetMol()


def _classify_rdkit_smiles(molecule: Chem.Mol) -> str:
    mirror = _invert_rdkit_tetrahedral_tags(molecule)
    supplied = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    reflected = Chem.MolToSmiles(
        mirror,
        canonical=True,
        isomericSmiles=True,
    )
    return "Achiral" if supplied == reflected else "Chiral"


def _classify_synkit(molecule: Chem.Mol) -> str:
    return classify_molecular_chirality(molecule).classification.value


def _load_stereomolgraph(checkout: Path) -> tuple[Any, str | None]:
    source = checkout.resolve() / "src"
    if not source.is_dir():
        message = f"StereoMolGraph source directory is absent: {source}"
        raise ValueError(message)
    sys.path.insert(0, str(source))
    from stereomolgraph import StereoMolGraph

    return StereoMolGraph, _git_revision(checkout.resolve())


def _run_backend(
    records: list[dict[str, Any]],
    molecules: list[Chem.Mol],
    classify: Callable[[Chem.Mol], str],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    predictions = {}
    errors = []
    started = time.perf_counter()
    for record, molecule in zip(records, molecules):
        identifier = str(record["ID"])
        try:
            with _case_time_limit(timeout_seconds):
                predictions[identifier] = classify(molecule)
        except Exception as error:  # pragma: no cover - external fail path
            errors.append(
                {
                    "id": identifier,
                    "error_type": _benchmark_error_type(error),
                }
            )
    seconds = time.perf_counter() - started
    summary = {
        "evaluated_records": len(predictions),
        "predictions": dict(sorted(Counter(predictions.values()).items())),
        "errors": errors,
        "seconds": seconds,
        "mean_ms_per_input": 1000.0 * seconds / len(records),
        "case_timeout_seconds": timeout_seconds,
    }
    return predictions, summary


def _reference_summary(
    records: list[dict[str, Any]],
    predictions: dict[str, str],
) -> dict[str, Any]:
    labelled = {
        str(record["ID"]): str(record["manual"])
        for record in records
        if record.get("manual") is not None
    }
    disagreements = sorted(
        identifier
        for identifier, prediction in predictions.items()
        if labelled[identifier] != prediction
    )
    return {
        "applicable": True,
        "correct": len(predictions) - len(disagreements),
        "accuracy_over_evaluated": (
            (len(predictions) - len(disagreements)) / len(predictions)
            if predictions
            else None
        ),
        "coverage": len(predictions) / len(records),
        "disagreement_ids": disagreements,
    }


def _pairwise_agreement(
    predictions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    result = {}
    backends = tuple(predictions)
    for left_index, left in enumerate(backends):
        for right in backends[left_index + 1 :]:
            common = sorted(set(predictions[left]) & set(predictions[right]))
            disagreements = []
            for identifier in common:
                left_value = predictions[left][identifier]
                right_value = predictions[right][identifier]
                if left_value != right_value:
                    disagreements.append(identifier)
            key = f"{left}_vs_{right}"
            agreement_count = len(common) - len(disagreements)
            agreement_rate = agreement_count / len(common) if common else None
            result[key] = {
                "common_records": len(common),
                "agreement_count": agreement_count,
                "agreement_rate": agreement_rate,
                "disagreement_ids": disagreements,
            }
    return result


def _run_setting(
    records: list[dict[str, Any]],
    molecules: list[Chem.Mol],
    stereomolgraph: Any,
    *,
    timeout_seconds: float,
    reference_applicable: bool,
    setting: str,
) -> dict[str, Any]:
    if setting == "stereo_removed":
        prepared = []
        for molecule in molecules:
            stripped = Chem.Mol(molecule)
            Chem.RemoveStereochemistry(stripped)
            prepared.append(stripped)
    else:
        prepared = [Chem.Mol(molecule) for molecule in molecules]

    explicit = [Chem.AddHs(Chem.Mol(molecule)) for molecule in prepared]

    def classify_stereomolgraph(molecule: Chem.Mol) -> str:
        graph = stereomolgraph.from_rdmol(
            Chem.Mol(molecule),
            stereo_complete=True,
        )
        return "Achiral" if graph == graph.enantiomer() else "Chiral"

    functions = {
        "synkit": (_classify_synkit, prepared),
        "rdkit_smiles": (_classify_rdkit_smiles, prepared),
        "stereomolgraph": (classify_stereomolgraph, explicit),
    }
    prediction_sets = {}
    summaries = {}
    for name, (classify, inputs) in functions.items():
        predictions, summary = _run_backend(
            records,
            inputs,
            classify,
            timeout_seconds=timeout_seconds,
        )
        prediction_sets[name] = predictions
        if reference_applicable:
            summary["manual_label_agreement"] = _reference_summary(
                records,
                predictions,
            )
        else:
            summary["manual_label_agreement"] = {
                "applicable": False,
                "reason": (
                    "reference annotations are not whole-molecule binary "
                    "chirality labels"
                ),
            }
        summaries[name] = summary

    return {
        "protocol": (
            "binary_mirror_equality_on_supplied_stereo"
            if setting == "supplied_stereo"
            else "binary_mirror_equality_after_stereo_removal"
        ),
        "backends": summaries,
        "pairwise_agreement": _pairwise_agreement(prediction_sets),
        "interpretation": (
            "Stereo removal destroys configuration information. Binary "
            "agreement with the original ACS label measures apparent label "
            "retention, not recovery of the erased stereoisomer."
            if setting == "stereo_removed"
            else "Binary whole-molecule mirror comparison on supplied input."
        ),
    }


def _parse_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Chem.Mol], list[str]]:
    parsed_records = []
    molecules = []
    failures = []
    for record in records:
        molecule = Chem.MolFromSmiles(str(record["SMILES"]))
        if molecule is None:
            failures.append(str(record["ID"]))
        else:
            parsed_records.append(record)
            molecules.append(molecule)
    return parsed_records, molecules, failures


def _rdkit_cip_labels(molecule: Chem.Mol) -> set[str]:
    """Return atom-numbered CIP labels in Validation Suite notation."""
    working = Chem.Mol(molecule)
    rdCIPLabeler.AssignCIPLabels(working)
    labels = set()
    for atom in working.GetAtoms():
        if atom.HasProp("_CIPCode"):
            labels.add(f"{atom.GetIdx() + 1}{atom.GetProp('_CIPCode')}")
    for bond in working.GetBonds():
        if not bond.HasProp("_CIPCode"):
            continue
        descriptor = bond.GetProp("_CIPCode")
        labels.add(f"{bond.GetBeginAtomIdx() + 1}{descriptor}")
        labels.add(f"{bond.GetEndAtomIdx() + 1}{descriptor}")
    return labels


def _benchmark_rdkit_cip_assignment(
    records: list[dict[str, Any]],
    molecules: list[Chem.Mol],
) -> dict[str, Any]:
    """Score RDKit's native CIP assigner against local reference labels."""
    started = time.perf_counter()
    predictions = [_rdkit_cip_labels(molecule) for molecule in molecules]
    seconds = time.perf_counter() - started
    exact_ids = []
    disagreement_ids = []
    expected_total = 0
    predicted_total = 0
    true_positive = 0
    for record, predicted in zip(records, predictions):
        expected = set(record["recommended_labels"])
        identifier = str(record["ID"])
        if predicted == expected:
            exact_ids.append(identifier)
        else:
            disagreement_ids.append(identifier)
        expected_total += len(expected)
        predicted_total += len(predicted)
        true_positive += len(expected & predicted)
    return {
        "applicable": True,
        "method": "rdCIPLabeler.AssignCIPLabels; exact label set",
        "exact_records": len(exact_ids),
        "records": len(records),
        "exact_record_accuracy": len(exact_ids) / len(records),
        "micro_label_recall": true_positive / expected_total,
        "micro_label_precision": true_positive / predicted_total,
        "expected_labels": expected_total,
        "predicted_labels": predicted_total,
        "true_positive_labels": true_positive,
        "disagreement_ids": disagreement_ids,
        "seconds": seconds,
        "mean_ms_per_input": 1000.0 * seconds / len(records),
    }


def _cip_binary_reference(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach complete, provenance-tiered global labels to CIP records."""
    labels = {row["ID"]: row["manual"] for row in load_dataset()}
    if set(labels) & set(CIP_ATROP_BINARY_LABELS):
        raise ValueError("ACS and atrop binary reference IDs overlap")
    labels.update(CIP_ATROP_BINARY_LABELS)
    if set(labels) & set(CIP_CURATED_BINARY_LABELS):
        raise ValueError("Published and curated binary reference IDs overlap")
    labels.update(CIP_CURATED_BINARY_LABELS)
    identifiers = {str(row["ID"]) for row in rows}
    if set(labels) != identifiers:
        missing = sorted(identifiers - set(labels))
        extra = sorted(set(labels) - identifiers)
        message = f"Incomplete CIP binary reference: {missing=}, {extra=}"
        raise ValueError(message)
    labelled = [{**row, "manual": labels[str(row["ID"])]} for row in rows]
    return labelled, {
        "status": "provisional_expert_curation_review_required",
        "labels": dict(sorted(Counter(labels.values()).items())),
        "published_acs_manual_records": 258,
        "independent_atrop_notebook_records": 7,
        "expert_curated_records": 35,
        "expert_curated_ids": sorted(CIP_CURATED_BINARY_LABELS),
        "warning": (
            "Do not present as an independent published 300-case global "
            "benchmark until the 35 added labels receive chemistry review."
        ),
    }


def benchmark(
    *,
    cip_path: Path,
    stereomolgraph_path: Path,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run both binary settings for all registered datasets and backends."""
    stereomolgraph, revision = _load_stereomolgraph(stereomolgraph_path)
    acs = [
        {
            "ID": row["ID"],
            "SMILES": row["Input SMILES"],
            "manual": row["manual"],
            "published_rdkit_smiles": row["RDKit SMILES"],
            "published_stereomolgraph": row["StereoMolGraph"],
        }
        for row in load_dataset()
    ]
    rota = [
        {
            "ID": f"RotA-{index:04d}",
            "SMILES": row["SMILES"],
            "manual": "Chiral",
        }
        for index, row in enumerate(load_rota(), start=1)
    ]
    cip_source = load_cip(cip_path)
    cip, cip_reference = _cip_binary_reference(cip_source)
    datasets = {
        "acs_stereomolgraph_molecular_chirality": (
            acs,
            True,
            "whole_molecule_chiral_achiral",
        ),
        "chiralfinder_rota": (
            rota,
            True,
            "derived_positive_only_whole_molecule_chirality",
        ),
        "cip_validation_suite": (
            cip,
            True,
            "derived_whole_molecule_chiral_achiral",
        ),
    }

    results = {}
    for name, (records, reference_applicable, task) in datasets.items():
        parsed, molecules, failures = _parse_records(records)
        result = {
            "records": len(records),
            "task": task,
            "rdkit_parse_failures": failures,
            "reference_accuracy_applicable": reference_applicable,
        }
        for setting in ("supplied_stereo", "stereo_removed"):
            result[setting] = _run_setting(
                parsed,
                molecules,
                stereomolgraph,
                timeout_seconds=timeout_seconds,
                reference_applicable=reference_applicable,
                setting=setting,
            )
            print(
                f"completed {name} {setting}",
                file=sys.stderr,
                flush=True,
            )
        results[name] = result

    acs_result = results["acs_stereomolgraph_molecular_chirality"]
    acs_result["native_task_accuracy"] = {
        "synkit": {"applicable": True, "reported_under": "supplied_stereo"},
        "rdkit_smiles": {
            "applicable": True,
            "reported_under": "supplied_stereo",
        },
        "stereomolgraph": {
            "applicable": True,
            "reported_under": "supplied_stereo",
        },
    }
    results["chiralfinder_rota"]["native_task_accuracy"] = {
        backend: {
            "applicable": False,
            "reason": (
                "backend does not predict RotA axial atom-pair/chain loci "
                "from the supplied 2D SMILES"
            ),
        }
        for backend in ("synkit", "rdkit_smiles", "stereomolgraph")
    }
    results["chiralfinder_rota"]["binary_reference"] = {
        "status": "source_derived_positive_only",
        "labels": {"Chiral": 650},
        "warning": (
            "This measures positive recall only: RotA supplies no achiral "
            "controls and its SMILES does not encode atropisomer "
            "configuration."
        ),
    }
    cip_records, cip_molecules, _cip_failures = _parse_records(cip)
    results["cip_validation_suite"]["native_task_accuracy"] = {
        "synkit": {
            "applicable": False,
            "reason": (
                "SynKit has no independent full CIP assignment backend; "
                "stored RDKit CIP properties are not a separate method"
            ),
        },
        "rdkit": _benchmark_rdkit_cip_assignment(
            cip_records,
            cip_molecules,
        ),
        "stereomolgraph": {
            "applicable": False,
            "reason": (
                "StereoMolGraph represents relative stereo but exposes no "
                "R/S/r/s/M/P/m/p/E/Z CIP label assignment API"
            ),
        },
    }
    results["cip_validation_suite"]["binary_reference"] = cip_reference

    acs_result = results["acs_stereomolgraph_molecular_chirality"]
    supplied = acs_result["supplied_stereo"]["backends"]
    supplied["rdkit_smiles"]["published_column_correct"] = 235
    supplied["stereomolgraph"]["published_column_correct"] = 258
    return {
        "schema": "synkit.stereo-live-backend-comparison/1",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
            "stereomolgraph_checkout": str(stereomolgraph_path.resolve()),
            "stereomolgraph_revision": revision,
        },
        "methods": {
            "synkit": (
                "complete eligible sp3, even-cumulene, and biaryl-axis "
                "topology with isotope-aware identity; test exact "
                "stereo-aware mirror graph isomorphism"
            ),
            "rdkit_smiles": (
                "publisher method: invert RDKit tetrahedral atom tags and "
                "compare canonical isomeric SMILES"
            ),
            "stereomolgraph": (
                "Chem.AddHs outside timing; from_rdmol with "
                "stereo_complete=True; compare graph with enantiomer"
            ),
        },
        "datasets": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip-path", type=Path, required=True)
    parser.add_argument("--stereomolgraph", type=Path, required=True)
    parser.add_argument("--case-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.case_timeout_seconds <= 0:
        parser.error("case-timeout-seconds must be positive")
    RDLogger.DisableLog("rdApp.*")
    report = benchmark(
        cip_path=arguments.cip_path,
        stereomolgraph_path=arguments.stereomolgraph,
        timeout_seconds=arguments.case_timeout_seconds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
