"""Exhaustive small action checks and independently constructed shape groups."""

from __future__ import annotations

import argparse
from itertools import permutations, product
import hashlib
import json
from math import factorial
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compose(left, right):
    """Compose slot maps as ordinary functions: left after right."""
    return tuple(left[right[i]] for i in range(len(left)))


def inverse(image):
    return tuple(image.index(i) for i in range(len(image)))


def sign(image):
    return (-1) ** sum(image[i] > image[j] for i in range(len(image)) for j in range(i + 1, len(image)))


def tbp_rotations():
    """Proper rotations: preserve the axial pair and equatorial triple.

    A cyclic equatorial action keeps the axis; an odd equatorial action
    reverses it. This defines D3 geometrically without production shape tables.
    """
    return tuple(
        image for image in permutations(range(5))
        if set(image[:2]) == {0, 1}
        and sign(tuple(value - 2 for value in image[2:])) == (1 if image[0] == 0 else -1)
    )


def determinant(columns):
    a, b, c = columns
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
    )


def octahedral_rotations():
    """Proper isometries of six exact integer vertices of an octahedron."""
    vertices = ((0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))

    def dot(left, right):
        return sum(a * b for a, b in zip(left, right))

    return tuple(
        image for image in permutations(range(6))
        if determinant(tuple(vertices[image[i]] for i in (2, 3, 0))) == 1
        and all(dot(vertices[i], vertices[j]) == dot(vertices[image[i]], vertices[image[j]])
                for i in range(6) for j in range(6))
    )


def normalizer(group):
    members = set(group)
    return tuple(
        image for image in permutations(range(len(group[0])))
        if {compose(compose(image, member), inverse(image)) for member in members} == members
    )


def action_audit():
    from synkit.Graph.Stereo.orbits import Permutation

    symmetric = tuple(permutations(range(3)))
    action_failures = 0
    reversed_action_failures = 0
    for left, right in product(symmetric, repeat=2):
        p, q = Permutation(left), Permutation(right)
        observed = p.then(q).apply(("a", "b", "c"))
        action_failures += observed != q.apply(p.apply(("a", "b", "c")))
        reversed_action_failures += observed != p.apply(q.apply(("a", "b", "c")))

    def alignment(frame, target, mapping):
        return tuple(target.index(mapping[value]) for value in frame)

    cocycle_failures = 0
    reversed_cocycle_failures = 0
    for u, v, w, f, g in product(symmetric, repeat=5):
        rho_f, rho_g = alignment(u, v, f), alignment(v, w, g)
        direct = alignment(u, w, compose(g, f))
        cocycle_failures += direct != Permutation(rho_g).then(Permutation(rho_f)).image
        reversed_cocycle_failures += direct != Permutation(rho_f).then(Permutation(rho_g)).image
    return {
        "action_pairs": len(symmetric) ** 2, "action_failures": action_failures,
        "wrong_action_detected_cases": reversed_action_failures,
        "cocycle_cases": len(symmetric) ** 5, "cocycle_failures": cocycle_failures,
        "wrong_cocycle_detected_cases": reversed_cocycle_failures,
    }


def shape_audit():
    from synkit.Graph.Stereo.orbits import SHAPE_DEFINITIONS

    results = {}
    for shape, group in [("trigonal_bipyramidal", tbp_rotations()), ("octahedral", octahedral_rotations())]:
        registered = {
            tuple(value - 1 for value in item.image[1:])
            for item in SHAPE_DEFINITIONS[shape].preserving_group.elements
        }
        norm = normalizer(group)
        results[shape] = {
            "geometric_group_order": len(group), "matches_registered_action": set(group) == registered,
            "frame_index": factorial(len(group[0])) // len(group),
            "normalizer_order": len(norm), "normalizer_quotient_order": len(norm) // len(group),
            "nontrivial_involutions_modulo_group": [
                image for image in norm if image not in group and compose(image, image) in group
            ],
        }
    return results


def expansion_size_audit():
    import networkx as nx
    from synkit.Graph.Stereo import (
        FrameworkFrame, FrameworkStereo, HelicalStereo, PlanarChiralityStereo,
        expand_configured_stereograph,
    )

    records = []

    def check(name, graph, descriptor, expected):
        expanded = expand_configured_stereograph(graph, (descriptor,))
        added = len(expanded) - len(graph) - graph.number_of_edges()
        records.append({"family": name, "support_length": len(descriptor.dependencies),
                        "observed_added_vertices": added, "expected_added_vertices": expected,
                        "passed": added == expected})

    for length in (4, 8, 12):
        check("open_helical", nx.path_graph(length), HelicalStereo(tuple(range(length)), 1), 3 + 2 * length)
        check("cyclic_helical", nx.cycle_graph(length),
              HelicalStereo(tuple(range(length)), 1, cyclic=True), 1 + 2 * length + 2 * length ** 2)
        graph = nx.cycle_graph(length)
        graph.add_edge(0, length)
        check("planar_chirality", graph, PlanarChiralityStereo(tuple(range(length)), length, 1),
              1 + length + length * (length + 1))
    for virtual in (False, True):
        graph = nx.star_graph(3 if virtual else 4)
        frame = FrameworkFrame(0, (1, 2, 3, "@H:0" if virtual else 4), 1)
        graph.nodes[0]["hcount"] = int(virtual)
        descriptor = FrameworkStereo(frozenset(graph), (frame,), 1)
        check("framework_virtual" if virtual else "framework", graph, descriptor,
              1 + len(graph) + 61 + 12 * int(virtual))
    return records


def run_audit():
    actions, shapes, sizes = action_audit(), shape_audit(), expansion_size_audit()
    passed = (
        actions["action_failures"] == actions["cocycle_failures"] == 0
        and actions["wrong_action_detected_cases"] > 0 and actions["wrong_cocycle_detected_cases"] > 0
        and all(value["matches_registered_action"] and value["normalizer_quotient_order"] == 2
                for value in shapes.values())
        and all(value["passed"] for value in sizes)
    )
    source_paths = (
        "Experiment/Stereo/Canonicalization/theory_audit.py",
        "synkit/Graph/Stereo/orbits.py", "synkit/Graph/Stereo/configured.py",
        "synkit/Graph/Stereo/global_configured.py", "paper/stereo_graph/main.tex",
    )
    provenance = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "python": platform.python_version(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths},
    }
    return {"schema": "synkit.stereo-theory-audit/1", "provenance": provenance,
            "actions": actions, "shapes": shapes,
            "expansion_sizes": sizes, "passed": passed,
            "boundary": ("Independent geometric group construction and finite algebra/size checks; "
                         "not a proof of every expansion family.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "actions": report["actions"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
