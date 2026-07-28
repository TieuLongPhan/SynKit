#!/usr/bin/env python3
"""Record bounded-cost evidence for coupled-framework stereographs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Any

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Chem.Molecule.chirality import _indexed_copy  # noqa: E402
from synkit.Chem.Molecule.global_stereo import (  # noqa: E402
    analyze_global_stereo_support,
)
from synkit.Graph.Stereo import (  # noqa: E402
    FrameworkStereo,
    classify_configured_stereograph_mirror,
    classify_rdkit_stereograph_mirror,
    descriptor_id,
    expand_configured_stereograph,
)
from synkit.IO.mol_to_graph import MolToGraph  # noqa: E402

SCHEMA = "synkit.global-stereo-performance/1"
CASES = {
    "minimal_positive": "C1C2(OCC1)OCCC2",
    "achiral_near_miss": "C1C2(CCC1)CCCC2",
    "vs300": (
        "O[C@H](C[C@@]12C=3CCC=C2CCC=C1CCC3)"
        "C[C@]45C=6CCC=C5CCC=C4CCC6"
    ),
}


def measure_case(smiles: str, *, repeats: int = 3) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot parse performance case: {smiles!r}.")
    certificate = analyze_global_stereo_support(molecule)
    if certificate.descriptor is None:
        raise ValueError("Performance cases require a coupled-frame descriptor.")
    descriptor = FrameworkStereo(
        certificate.descriptor.support_atoms,
        certificate.descriptor.frames,
        1,
        "performance:orientation_probe",
    )
    working = _indexed_copy(molecule)
    graph = MolToGraph(attr_profile="minimal").transform(
        working,
        use_index_as_atom_map=True,
    )
    registry = {descriptor_id(descriptor): descriptor}
    auxiliary = expand_configured_stereograph(
        graph,
        registry.values(),
        atom_color=("element", "isotope", "hcount"),
        bond_color=lambda _attributes: "bond",
    )
    durations = []
    result = None
    tracemalloc.start()
    for _repeat in range(repeats):
        started = time.perf_counter()
        result = classify_configured_stereograph_mirror(
            graph,
            registry,
            atom_color=("element", "isotope", "hcount"),
            bond_color=lambda _attributes: "bond",
        )
        durations.append((time.perf_counter() - started) * 1000)
    _current, global_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    baseline_durations = []
    baseline = None
    tracemalloc.start()
    for _repeat in range(repeats):
        started = time.perf_counter()
        baseline = classify_rdkit_stereograph_mirror(
            molecule,
            require_complete=False,
            identity_profile="chemical",
        )
        baseline_durations.append((time.perf_counter() - started) * 1000)
    _current, baseline_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result is not None
    assert baseline is not None
    return {
        "atom_count": len(graph),
        "bond_count": graph.number_of_edges(),
        "frame_count": len(descriptor.frames),
        "auxiliary_nodes": auxiliary.number_of_nodes(),
        "auxiliary_edges": auxiliary.number_of_edges(),
        "auxiliary_node_growth": round(
            auxiliary.number_of_nodes() / len(graph),
            3,
        ),
        "status": result.status.value,
        "wall_ms": {
            "median": round(statistics.median(durations), 3),
            "minimum": round(min(durations), 3),
            "maximum": round(max(durations), 3),
            "repeats": repeats,
        },
        "peak_traced_bytes": global_peak_bytes,
        "source_declared_baseline": {
            "status": baseline.status.value,
            "wall_ms": {
                "median": round(statistics.median(baseline_durations), 3),
                "minimum": round(min(baseline_durations), 3),
                "maximum": round(max(baseline_durations), 3),
                "repeats": repeats,
            },
            "peak_traced_bytes": baseline_peak_bytes,
        },
    }


def run() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "cases": {
            name: measure_case(smiles)
            for name, smiles in CASES.items()
        },
        "claim_boundary": (
            "Microbenchmark timings are environment-specific. Auxiliary graph "
            "sizes and classifications are deterministic regression evidence. "
            "Peak traced bytes cover Python allocations only, not native-library "
            "memory."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
