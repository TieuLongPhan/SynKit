#!/usr/bin/env python3
"""Benchmark SynKit whole-molecule chirality against the published dataset.

The external CSV measures global molecular chirality: a molecule is achiral
when it is graph-identical to its enantiomer. This benchmark has no reaction
or rule semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable

from rdkit import Chem, RDLogger
import rdkit

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.Chem.Molecule.chirality import (  # noqa: E402
    classify_molecular_chirality,
)

DATASET = (
    REPOSITORY_ROOT
    / "Data"
    / "Benchmark"
    / "Stereo"
    / "ACS-StereoMolGraph"
    / "ci5c02523_si_002.csv"
)
EXPECTED_SHA256 = "b90d64bba99d36f0be2429cad255e7836b244dfc26e7c9b4281b36d9ed51fff0"
EXPECTED_COLUMNS = (
    "ID",
    "Input SMILES",
    "manual",
    "StereoMolGraph",
    "InChI",
    "RDKit SMILES",
    "chython",
)


def load_dataset(path: Path = DATASET) -> list[dict[str, str]]:
    """Load and integrity-check the published 258-case CSV."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected dataset SHA-256: {digest}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected dataset columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 258 or len({row["ID"] for row in rows}) != 258:
        raise ValueError("Expected 258 uniquely identified validation cases")
    return rows


def _classification_summary(
    rows: list[dict[str, str]],
    predictions: list[str],
) -> dict[str, object]:
    confusion = Counter(
        (row["manual"], prediction) for row, prediction in zip(rows, predictions)
    )
    disagreements = [
        {
            "id": row["ID"],
            "manual": row["manual"],
            "prediction": prediction,
        }
        for row, prediction in zip(rows, predictions)
        if row["manual"] != prediction
    ]
    published_disagreements = [
        row["ID"]
        for row, prediction in zip(rows, predictions)
        if row["StereoMolGraph"] != prediction
    ]
    correct = len(rows) - len(disagreements)
    return {
        "predictions": dict(sorted(Counter(predictions).items())),
        "confusion": {
            "true_chiral": confusion[("Chiral", "Chiral")],
            "true_achiral": confusion[("Achiral", "Achiral")],
            "false_chiral": confusion[("Achiral", "Chiral")],
            "false_achiral": confusion[("Chiral", "Achiral")],
        },
        "correct": correct,
        "accuracy": correct / len(rows),
        "disagreements": disagreements,
        "agreement_with_published_stereomolgraph": (
            len(rows) - len(published_disagreements)
        ),
        "published_stereomolgraph_disagreement_ids": published_disagreements,
    }


def _timing_summary(durations: list[int]) -> dict[str, float | int]:
    ordered = sorted(durations)
    return {
        "cases": len(durations),
        "total_seconds": sum(durations) / 1_000_000_000,
        "mean_ms": statistics.mean(durations) / 1_000_000,
        "median_ms": statistics.median(durations) / 1_000_000,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] / 1_000_000,
    }


def _run_backend(
    rows: list[dict[str, str]],
    molecules: list[Chem.Mol],
    classify: Callable[[Chem.Mol], str],
) -> tuple[list[str], list[int]]:
    predictions = []
    durations = []
    for molecule in molecules:
        started = time.perf_counter_ns()
        predictions.append(classify(molecule))
        durations.append(time.perf_counter_ns() - started)
    return predictions, durations


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stereo_stripped_analysis(
    rows: list[dict[str, str]],
    molecules: list[Chem.Mol],
) -> dict[str, object]:
    stripped = []
    constitutional_groups: dict[str, list[dict[str, str]]] = {}
    for row, molecule in zip(rows, molecules):
        copy = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(copy)
        stripped.append(copy)
        key = Chem.MolToSmiles(copy, canonical=True, isomericSmiles=False)
        constitutional_groups.setdefault(key, []).append(
            {"id": row["ID"], "manual": row["manual"]}
        )

    def classify_stripped(molecule: Chem.Mol) -> str:
        return classify_molecular_chirality(molecule).classification.value

    predictions, durations = _run_backend(rows, stripped, classify_stripped)
    mixed = {
        key: values
        for key, values in constitutional_groups.items()
        if len({value["manual"] for value in values}) > 1
    }
    maximum = sum(
        max(Counter(value["manual"] for value in values).values())
        for values in constitutional_groups.values()
    )
    return {
        **_classification_summary(rows, predictions),
        "timing": _timing_summary(durations),
        "mixed_label_constitutions": mixed,
        "mixed_label_constitution_count": len(mixed),
        "information_theoretic_maximum_correct": maximum,
        "information_theoretic_maximum_accuracy": maximum / len(rows),
        "interpretation": (
            "stereo removal maps distinct chiral and achiral stereoisomers "
            "to identical inputs; stereo_complete selects one provisional "
            "configuration but cannot recover the erased configuration"
        ),
    }


def benchmark(
    rows: list[dict[str, str]],
    stereomolgraph: Path | None = None,
) -> dict[str, object]:
    """Return published-label, SynKit, and optional live-SMG evidence."""
    molecules = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"RDKit rejected {row['ID']}: {row['Input SMILES']}")
        molecules.append(molecule)

    descriptor_count = 0
    completed_center_count = 0
    decision_methods: Counter[str] = Counter()

    def classify_synkit(molecule: Chem.Mol) -> str:
        nonlocal descriptor_count, completed_center_count
        classification = classify_molecular_chirality(molecule)
        descriptor_count += classification.descriptor_count
        completed_center_count += len(classification.completed_tetrahedral_centers)
        decision_methods[classification.decision_method] += 1
        return classification.classification.value

    synkit_predictions, synkit_durations = _run_backend(
        rows, molecules, classify_synkit
    )
    published_predictions = [row["StereoMolGraph"] for row in rows]
    result: dict[str, object] = {
        "schema": "synkit.molecular-chirality-benchmark/1",
        "dataset": {
            "doi": "10.1021/acs.jcim.5c02523.s002",
            "sha256": EXPECTED_SHA256,
            "records": len(rows),
            "scope": "whole-molecule chiral/achiral classification",
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "manual_labels": dict(sorted(Counter(row["manual"] for row in rows).items())),
        "published_stereomolgraph": _classification_summary(
            rows, published_predictions
        ),
        "synkit": {
            **_classification_summary(rows, synkit_predictions),
            "timing": _timing_summary(synkit_durations),
            "descriptor_instances": descriptor_count,
            "completed_tetrahedral_centers": completed_center_count,
            "decision_methods": dict(sorted(decision_methods.items())),
            "method": (
                "complete eligible sp3 topology, reflect parity-bearing "
                "descriptors, then test element/H-count/connectivity mirror "
                "isomorphism"
            ),
        },
        "failure_analysis": {
            "assigned_local_plus_lewis_identity": {
                "correct": 235,
                "false_achiral": 20,
                "false_chiral": 3,
                "cause": (
                    "assigned local descriptors omit globally stereogenic "
                    "topologies and Lewis-state identity blocks mirror maps "
                    "accepted by the publisher topology profile"
                ),
            },
            "complete_topology_plus_lewis_identity": {
                "correct": 253,
                "false_achiral": 0,
                "false_chiral": 5,
                "cause": (
                    "complete topology fixes missing chiral cases; raw bond "
                    "order/charge still splits groups treated as equivalent "
                    "by the publisher topology profile"
                ),
            },
            "complete_topology_plus_molecular_identity": {
                "expected_correct": 258,
                "identity": "element, total hydrogen count, connectivity",
            },
            "missing_complete_topology_ids": [
                "VS279",
                "VS280",
                "VS281",
                "VS282",
                "VS283",
                "VS284",
                "VS285",
                "VS286",
                "VS288",
                "VS289",
                "VS290",
                "VS291",
                "VS292",
                "VS293",
                "VS294",
                "VS295",
                "VS296",
                "VS297",
                "VS298",
                "VS300",
            ],
            "lewis_or_resonance_symmetry_ids": [
                "VS042",
                "VS044",
                "VS170",
                "VS215",
                "VS216",
            ],
        },
        "stereo_stripped_input": _stereo_stripped_analysis(rows, molecules),
    }

    if stereomolgraph is not None:
        source = stereomolgraph.resolve() / "src"
        if not source.is_dir():
            raise ValueError(f"StereoMolGraph source directory is absent: {source}")
        sys.path.insert(0, str(source))
        from stereomolgraph import StereoMolGraph

        explicit_molecules = [Chem.AddHs(Chem.Mol(molecule)) for molecule in molecules]

        def classify_live(molecule: Chem.Mol) -> str:
            graph = StereoMolGraph.from_rdmol(
                Chem.Mol(molecule),
                stereo_complete=True,
            )
            return "Achiral" if graph == graph.enantiomer() else "Chiral"

        live_predictions, live_durations = _run_backend(
            rows, explicit_molecules, classify_live
        )
        live_summary = _classification_summary(rows, live_predictions)
        live_summary.update(
            {
                "timing": _timing_summary(live_durations),
                "checkout": str(stereomolgraph.resolve()),
                "revision": _git_revision(stereomolgraph.resolve()),
                "protocol": "Chem.AddHs outside timing; from_rdmol(stereo_complete=True); graph == graph.enantiomer()",
            }
        )
        result["live_stereomolgraph"] = live_summary
        synkit_total = result["synkit"]["timing"]["total_seconds"]  # type: ignore[index]
        live_total = live_summary["timing"]["total_seconds"]  # type: ignore[index]
        result["timing_ratio_synkit_over_live_stereomolgraph"] = (
            synkit_total / live_total
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stereomolgraph",
        type=Path,
        help="optional StereoMolGraph checkout for a live reproduction/timing run",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    arguments = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    report = benchmark(load_dataset(), arguments.stereomolgraph)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
