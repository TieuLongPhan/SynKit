"""Enumerate the finite witness-basis boundary and replay equivalent inputs.

This is a local algebra audit with distinct formal references, not a chemical
corpus or a selectivity experiment. Counts are exact finite enumerations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import synkit  # noqa: E402
from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoChange,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
)
from synkit.Graph.Stereo.orbits import SHAPE_DEFINITIONS  # noqa: E402

TYPES = (TetrahedralStereo, SquarePlanarStereo, TrigonalBipyramidalStereo,
         OctahedralStereo, PlanarBondStereo, AtropBondStereo)


def compose(left, right):
    """Compose slot images as functions, independently of the then API."""
    return tuple(left[index] for index in right)


def inverse(image):
    result = [0] * len(image)
    for output, source in enumerate(image):
        result[source] = output
    return tuple(result)


def audit():
    rows = []
    for descriptor_type in TYPES:
        shape = descriptor_type.descriptor_class
        definition = SHAPE_DEFINITIONS[shape]
        group = tuple(p.image for p in definition.preserving_group.elements)
        preserving = set(group)
        admitted = tuple(p.image for p in definition.unspecified_group.elements)
        base = tuple(range(definition.frame_arity))
        parity = 0 if shape in {"square_planar", "planar_bond"} else 1
        source = descriptor_type(base, parity)
        histogram = Counter()
        replayed = 0
        normalizers = 0
        for pi in admitted:
            target = descriptor_type(compose(base, pi), parity)
            change = StereoChange.from_endpoints(source, target)
            fixed_products = set()
            normalizes = {compose(compose(inverse(pi), gamma), pi)
                          for gamma in group} == preserving
            normalizers += normalizes
            for gamma in group:
                # The operator T_pi acts on the same input orbit with gamma
                # representatives. Its product orbit is computed independently.
                raw = compose(gamma, pi)
                fixed_products.add(min(compose(raw, h) for h in group))
                pi_gamma = compose(compose(inverse(gamma), pi), gamma)
                rebased = compose(gamma, pi_gamma)
                assert rebased == compose(pi, gamma)
                equivalent_source = descriptor_type(compose(base, gamma), parity)
                assert equivalent_source.same_configuration(source)
                product = change.apply_to(equivalent_source)
                assert product.same_configuration(target), (shape, pi, gamma)
                replayed += 1
            histogram[len(fixed_products)] += 1
            assert (len(fixed_products) == 1) == normalizes, (shape, pi)
        rows.append({
            "shape": shape,
            "preserving_group_order": len(group),
            "admissible_group_order": len(admitted),
            "normalizer_order": normalizers,
            "fixed_witness_product_class_histogram": {
                str(k): v for k, v in sorted(histogram.items())},
            "replayed_equivalent_input_pairs": replayed,
            "all_production_replays_match_declared_target_orbit": True,
        })
    assert sum(row["admissible_group_order"] for row in rows) == 904
    assert sum(row["replayed_equivalent_input_pairs"] for row in rows) == 18544
    sources = [Path(__file__).resolve(), ROOT / "synkit/Graph/Stereo/changes.py",
               ROOT / "synkit/Graph/Stereo/orbits.py",
               ROOT / "synkit/Graph/Stereo/_descriptor_core.py"]
    return {
        "schema": "synkit.stereo-rxn-witness-basis-audit/1",
        "passed": True,
        "package_version": synkit.__version__,
        "records": rows,
        "admitted_witnesses": 904,
        "equivalent_input_replays": 18544,
        "normalizer_criterion_checked_for_every_admitted_witness": True,
        "basis_rebasing_checked_for_every_pair": True,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "scope": "Six native shapes, distinct formal references, all admitted witnesses "
                 "and preserving representatives. "
                 "No coordinates, chemical selectivity or extended-family reaction execution are inferred.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/stereo_rxn/results/witness_basis_audit.json")
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shape", "product_classes", "admitted_witness_count", "admissible_group_order"])
        for row in report["records"]:
            for count, witnesses in row["fixed_witness_product_class_histogram"].items():
                writer.writerow([row["shape"], count, witnesses, row["admissible_group_order"]])
    print(json.dumps({"passed": report["passed"], "output": str(args.output),
                      "admitted_witnesses": report["admitted_witnesses"],
                      "equivalent_input_replays": report["equivalent_input_replays"]}))


if __name__ == "__main__":
    main()
