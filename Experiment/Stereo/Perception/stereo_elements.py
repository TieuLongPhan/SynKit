#!/usr/bin/env python3
"""Audit stereo-element perception on the native 300-record CIP suite.

The suite is used here only as a collection of configured molecular inputs and
local reference positions.  This runner does not score CIP label prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    CIP_SHA256,
    load_cip,
)
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    TetrahedralConstitutionStatus,
    TetrahedralFrameStatus,
    perceive_tetrahedral_stereo,
)
from synkit.Graph.Stereo import TetrahedralStereo  # noqa: E402

_LABEL_PATTERN = re.compile(r"^(\d+)([A-Za-z]+)$")
_TETRAHEDRAL_LABELS = frozenset({"R", "S", "r", "s"})
_TETRAHEDRAL_TAGS = frozenset(
    {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
)


def _unit_categories(record: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(
        value.strip()
        for value in str(record["stereo_units"]).split(",")
        if value.strip()
    )
    return values or ("none",)


def _reference_rs_positions(record: dict[str, Any]) -> tuple[int, ...]:
    """Return R/S-like reference positions without assigning element type."""
    positions = []
    for label in record["recommended_labels"]:
        match = _LABEL_PATTERN.fullmatch(str(label))
        if match is None:
            continue
        position, descriptor = match.groups()
        if descriptor in _TETRAHEDRAL_LABELS:
            positions.append(int(position))
    return tuple(sorted(set(positions)))


def _transport_frame(
    frame: tuple[int | str, ...] | None,
    mapping: dict[int, int],
) -> tuple[int | str, ...] | None:
    if frame is None:
        return None
    return TetrahedralStereo(frame, 1).relabel(mapping).atoms


def _audit_carrier_pair(
    center: int,
    evidence: Any,
    transported: Any,
    mapping: dict[int, int],
) -> tuple[str, ...]:
    issues = []
    comparisons = (
        ("constitution_status", evidence.status, transported.status),
        ("frame_status", evidence.frame_status, transported.frame_status),
        (
            "dependency_depth",
            evidence.dependency_depth,
            transported.dependency_depth,
        ),
        (
            "neighbor_keys",
            tuple(item.neighborhood_key for item in evidence.ligand_classes),
            tuple(item.neighborhood_key for item in transported.ligand_classes),
        ),
        (
            "canonical_frame",
            _transport_frame(evidence.canonical_frame, mapping),
            transported.canonical_frame,
        ),
    )
    for label, original, image in comparisons:
        if original != image:
            issues.append(f"{label}:{center + 1}")
    return tuple(issues)


def _renumbering_audit(molecule: Chem.Mol) -> tuple[bool, tuple[str, ...]]:
    size = molecule.GetNumAtoms()
    order = list(reversed(range(size)))
    mapping = {old: new for new, old in enumerate(order)}
    renumbered = Chem.RenumberAtoms(molecule, order)
    original_perception = perceive_tetrahedral_stereo(molecule)
    transported_perception = perceive_tetrahedral_stereo(renumbered)
    original_evidence = {
        item.support.center: item for item in original_perception.carrier_evidence
    }
    transported_evidence = {
        item.support.center: item for item in transported_perception.carrier_evidence
    }
    issues = []
    for center, evidence in original_evidence.items():
        transported = transported_evidence.get(mapping[center])
        if transported is None:
            issues.append(f"missing_carrier:{center + 1}")
            continue
        issues.extend(
            _audit_carrier_pair(
                center,
                evidence,
                transported,
                mapping,
            )
        )

    original_elements = {
        item.support.center: item for item in original_perception.elements
    }
    transported_elements = {
        item.support.center: item for item in transported_perception.elements
    }
    for center, element in original_elements.items():
        transported = transported_elements.get(mapping[center])
        if transported is None:
            issues.append(f"missing_element:{center + 1}")
            continue
        if element.configuration_state is not transported.configuration_state:
            issues.append(f"configuration_state:{center + 1}")
        if element.configuration is None:
            if transported.configuration is not None:
                issues.append(f"configuration_presence:{center + 1}")
        elif transported.configuration != element.configuration.relabel(mapping):
            issues.append(f"configuration:{center + 1}")
    return not issues, tuple(issues)


def _record_result(
    record: dict[str, Any],
    molecule: Chem.Mol,
    *,
    check_renumbering: bool = True,
) -> dict[str, Any]:
    perception = perceive_tetrahedral_stereo(molecule)
    evidence = perception.carrier_evidence
    elements = perception.elements
    broad = {item.support.center + 1 for item in evidence}
    confirmed = {
        item.support.center + 1 for item in evidence if item.confirms_stereogenic_center
    }
    primary = {
        item.support.center + 1
        for item in evidence
        if item.status is TetrahedralConstitutionStatus.CONSTITUTIONALLY_DISTINCT
    }
    dependent = {
        item.support.center + 1
        for item in evidence
        if item.status is TetrahedralConstitutionStatus.STEREO_DEPENDENT_DISTINCT
    }
    canonical = {
        item.support.center + 1 for item in evidence if item.has_canonical_frame
    }
    symmetry_related = {
        item.support.center + 1
        for item in evidence
        if item.frame_status is TetrahedralFrameStatus.SYMMETRY_RELATED
    }
    collisions = {
        item.support.center + 1
        for item in evidence
        if item.frame_status is TetrahedralFrameStatus.NEIGHBORHOOD_KEY_COLLISION
    }
    supplied = {
        atom.GetIdx() + 1
        for atom in molecule.GetAtoms()
        if atom.GetChiralTag() in _TETRAHEDRAL_TAGS
    }
    attached = {
        item.support.center + 1 for item in elements if item.configuration is not None
    }
    unsupported_geometry = supplied - broad
    constitutionally_unresolved = supplied & symmetry_related
    unresolved_collision = supplied & collisions
    reference = set(_reference_rs_positions(record))
    if check_renumbering:
        invariant, renumbering_issues = _renumbering_audit(molecule)
    else:
        invariant, renumbering_issues = None, ()
    return {
        "id": str(record["ID"]),
        "stereo_units": list(_unit_categories(record)),
        "reference_rs_positions": sorted(reference),
        "broad_carriers": sorted(broad),
        "confirmed_centers": sorted(confirmed),
        "primary_centers": sorted(primary),
        "stereo_dependent_centers": sorted(dependent),
        "canonical_frames": sorted(canonical),
        "symmetry_related_carriers": sorted(symmetry_related),
        "neighborhood_key_collisions": sorted(collisions),
        "supplied_tetrahedral_centers": sorted(supplied),
        "attached_configurations": sorted(attached),
        "supplied_unsupported_geometry": sorted(unsupported_geometry),
        "supplied_constitutionally_unresolved": sorted(constitutionally_unresolved),
        "supplied_unresolved_key_collision": sorted(unresolved_collision),
        "reference_rs_confirmed": sorted(reference & confirmed),
        "reference_rs_canonicalized": sorted(reference & canonical),
        "supplied_configuration_attached": sorted(supplied & attached),
        "renumbering_invariant": invariant,
        "renumbering_issues": list(renumbering_issues),
        "dependency_iterations": perception.dependency_iterations,
    }


def _sum_lengths(results: Iterable[dict[str, Any]], field: str) -> int:
    return sum(len(item[field]) for item in results)


def benchmark_stereo_elements(
    path: Path,
    *,
    check_renumbering: bool = True,
) -> dict[str, Any]:
    """Audit carrier, frame, and configuration coverage without CIP scoring."""
    records = load_cip(path)
    results = []
    parse_failures = []
    started = time.perf_counter()
    for record in records:
        molecule = Chem.MolFromSmiles(str(record["SMILES"]))
        if molecule is None:
            parse_failures.append(str(record["ID"]))
            continue
        results.append(
            _record_result(
                record,
                molecule,
                check_renumbering=check_renumbering,
            )
        )
    reference = _sum_lengths(results, "reference_rs_positions")
    supplied = _sum_lengths(results, "supplied_tetrahedral_centers")
    seconds = time.perf_counter() - started
    frame_statuses = Counter()
    for item in results:
        frame_statuses["canonical_primary"] += len(item["primary_centers"])
        frame_statuses["canonical_stereo_dependent"] += len(
            item["stereo_dependent_centers"]
        )
        frame_statuses["symmetry_related"] += len(item["symmetry_related_carriers"])
        frame_statuses["neighborhood_key_collision"] += len(
            item["neighborhood_key_collisions"]
        )
    return {
        "schema": "synkit.stereo-element-audit/1",
        "dataset": {
            "records": len(records),
            "audited_sha256": CIP_SHA256,
            "structures_vendored": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "task": (
            "staged tetrahedral carrier, primary/dependent canonical-frame, "
            "and supplied-configuration audit"
        ),
        "parse_failures": parse_failures,
        "totals": {
            "broad_carriers": _sum_lengths(results, "broad_carriers"),
            "confirmed_centers": _sum_lengths(results, "confirmed_centers"),
            "primary_centers": _sum_lengths(results, "primary_centers"),
            "stereo_dependent_centers": _sum_lengths(
                results, "stereo_dependent_centers"
            ),
            "canonical_frames": _sum_lengths(results, "canonical_frames"),
            "reference_rs_positions": reference,
            "reference_rs_confirmed": _sum_lengths(results, "reference_rs_confirmed"),
            "reference_rs_canonicalized": _sum_lengths(
                results, "reference_rs_canonicalized"
            ),
            "supplied_tetrahedral_centers": supplied,
            "supplied_configuration_attached": _sum_lengths(
                results, "supplied_configuration_attached"
            ),
            "supplied_unsupported_geometry": _sum_lengths(
                results, "supplied_unsupported_geometry"
            ),
            "supplied_constitutionally_unresolved": _sum_lengths(
                results, "supplied_constitutionally_unresolved"
            ),
            "supplied_unresolved_key_collision": _sum_lengths(
                results, "supplied_unresolved_key_collision"
            ),
            "renumbering_checked_records": sum(
                item["renumbering_invariant"] is not None for item in results
            ),
            "renumbering_invariant_records": sum(
                item["renumbering_invariant"] is True for item in results
            ),
        },
        "frame_statuses": dict(sorted(frame_statuses.items())),
        "reference_rs_fixed_point_coverage": (
            _sum_lengths(results, "reference_rs_confirmed") / reference
            if reference
            else None
        ),
        "reference_rs_canonicalization_coverage": (
            _sum_lengths(results, "reference_rs_canonicalized") / reference
            if reference
            else None
        ),
        "supplied_configuration_attachment_rate": (
            _sum_lengths(results, "supplied_configuration_attached") / supplied
            if supplied
            else None
        ),
        "renumbering_failures": [
            {
                "id": item["id"],
                "issues": item["renumbering_issues"],
            }
            for item in results
            if item["renumbering_invariant"] is False
        ],
        "records": results,
        "seconds": seconds,
        "mean_ms_per_input": 1000.0 * seconds / len(records),
        "claim_boundary": (
            "The 300 inputs are reused for stereo-element and invariance "
            "coverage only. This report does not score CIP label prediction, "
            "global chirality, or physical configurational stability."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-renumbering", action="store_true")
    arguments = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    report = benchmark_stereo_elements(
        arguments.cip_path,
        check_renumbering=not arguments.skip_renumbering,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
