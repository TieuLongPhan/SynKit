#!/usr/bin/env python3
"""Run an exact conformance matrix for whole-molecule stereo relations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any

from rdkit import Chem
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Graph.Stereo import (  # noqa: E402
    classify_rdkit_stereoisomer_relation,
)

_CASES = (
    {
        "id": "identical_atom_relabel",
        "left": "F[C@](Cl)(Br)I",
        "right": "F[C@](Cl)(Br)I",
        "renumber_right": True,
        "expected": "identical",
    },
    {
        "id": "tetrahedral_mirror",
        "left": "F[C@](Cl)(Br)I",
        "right": "F[C@@](Cl)(Br)I",
        "expected": "enantiomers",
    },
    {
        "id": "all_centers_inverted",
        "left": "F[C@H](Cl)[C@H](Br)I",
        "right": "F[C@@H](Cl)[C@@H](Br)I",
        "expected": "enantiomers",
    },
    {
        "id": "one_of_two_centers_inverted",
        "left": "F[C@H](Cl)[C@H](Br)I",
        "right": "F[C@@H](Cl)[C@H](Br)I",
        "expected": "diastereomers",
    },
    {
        "id": "e_z_pair",
        "left": "F/C=C/F",
        "right": "F/C=C\\F",
        "expected": "diastereomers",
    },
    {
        "id": "e_z_change_centers_unchanged",
        "left": "F/C=C/[C@H](Cl)[C@@H](Br)I",
        "right": "F/C=C\\[C@H](Cl)[C@@H](Br)I",
        "expected": "diastereomers",
    },
    {
        "id": "meso_mirror_identity",
        "left": "C[C@H](O)[C@H](O)C",
        "right": "C[C@@H](O)[C@@H](O)C",
        "expected": "identical",
    },
    {
        "id": "different_constitution",
        "left": "CC",
        "right": "CCC",
        "expected": "constitutionally_different",
    },
    {
        "id": "unspecified_locus",
        "left": "FC(Cl)(Br)I",
        "right": "F[C@](Cl)(Br)I",
        "expected": "incomplete",
    },
)


def benchmark_stereoisomer_relations() -> dict[str, Any]:
    records = []
    started_all = time.perf_counter()
    for case in _CASES:
        left = Chem.MolFromSmiles(case["left"])
        right = Chem.MolFromSmiles(case["right"])
        if left is None or right is None:
            raise ValueError(f"RDKit rejected conformance case {case['id']}")
        if case.get("renumber_right"):
            right = Chem.RenumberAtoms(
                right,
                tuple(reversed(range(right.GetNumAtoms()))),
            )
        started = time.perf_counter()
        result = classify_rdkit_stereoisomer_relation(left, right)
        observed = result.relation.value
        records.append(
            {
                **case,
                "observed": observed,
                "correct": observed == case["expected"],
                "left_constitution_digest": (result.left_constitution.canonical_digest),
                "right_constitution_digest": (
                    result.right_constitution.canonical_digest
                ),
                "left_stereograph_digest": (
                    result.left.canonical_digest if result.left else None
                ),
                "right_stereograph_digest": (
                    result.right.canonical_digest if result.right else None
                ),
                "mirror_left_digest": (
                    result.mirror_left.canonical_digest if result.mirror_left else None
                ),
                "incomplete_loci": list(result.incomplete_loci),
                "unsupported_loci": list(result.unsupported_loci),
                "duration_ms": 1000.0 * (time.perf_counter() - started),
            }
        )
    return {
        "schema": "synkit.stereoisomer-relation-conformance/1",
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "task": (
            "constitution/identity/mirror/residual-difference relation "
            "classification"
        ),
        "totals": {
            "cases": len(records),
            "correct": sum(record["correct"] for record in records),
            "expected_relations": dict(
                sorted(Counter(record["expected"] for record in records).items())
            ),
            "observed_relations": dict(
                sorted(Counter(record["observed"] for record in records).items())
            ),
        },
        "records": records,
        "seconds": time.perf_counter() - started_all,
        "claim_boundary": (
            "These designed conformance cases exercise exact relation logic; "
            "they are not a population-frequency or chemical-stability dataset."
        ),
    }
