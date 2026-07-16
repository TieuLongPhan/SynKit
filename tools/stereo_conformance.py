#!/usr/bin/env python3
"""Optional shared-subset conformance check against StereoMolGraph.

This runner compares equality matrices instead of importing upstream objects
into SynKit. It is intentionally a development tool, not a runtime dependency.
StereoMolGraph is MIT licensed; copyright (c) 2025 Maxim Papusha.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    stereo_isomorphic,
)

PINNED_COMMIT = "2189f610f23eaaf992e2e01a12ea4d0532496601"


def _equality_matrix(values: Sequence[Any]) -> list[list[bool]]:
    return [[left == right for right in values] for left in values]


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_conformance(repository: Path) -> dict[str, Any]:  # noqa: C901
    source = repository / "src"
    if not source.is_dir():
        raise ValueError(f"StereoMolGraph source directory is absent: {source}")
    sys.path.insert(0, str(source))
    try:
        from stereomolgraph import StereoMolGraph  # type: ignore[import-not-found]
        from stereomolgraph.stereodescriptors import (  # type: ignore[import-not-found]
            AtropBond,
            Octahedral,
            PlanarBond,
            SquarePlanar,
            Tetrahedral,
            TrigonalBipyramidal,
        )
    finally:
        sys.path.pop(0)

    fixtures = {
        "tetrahedral_specified": (
            [
                TetrahedralStereo((6, 0, 1, 2, 3), 1),
                TetrahedralStereo((6, 1, 2, 0, 3), 1),
                TetrahedralStereo((6, 0, 2, 1, 3), -1),
                TetrahedralStereo((6, 0, 2, 1, 3), 1),
            ],
            [
                Tetrahedral((6, 0, 1, 2, 3), 1),
                Tetrahedral((6, 1, 2, 0, 3), 1),
                Tetrahedral((6, 0, 2, 1, 3), -1),
                Tetrahedral((6, 0, 2, 1, 3), 1),
            ],
        ),
        "tetrahedral_unknown": (
            [
                TetrahedralStereo((6, 0, 1, 2, 3), None),
                TetrahedralStereo((6, 1, 2, 3, 0), None),
            ],
            [
                Tetrahedral((6, 0, 1, 2, 3), None),
                Tetrahedral((6, 1, 2, 3, 0), None),
            ],
        ),
        "planar_bond_specified": (
            [
                PlanarBondStereo((5, 4, 3, 2, 1, 0), 0),
                PlanarBondStereo((4, 5, 3, 2, 0, 1), 0),
                PlanarBondStereo((1, 0, 2, 3, 5, 4), 0),
                PlanarBondStereo((4, 5, 3, 2, 1, 0), 0),
            ],
            [
                PlanarBond((5, 4, 3, 2, 1, 0), 0),
                PlanarBond((4, 5, 3, 2, 0, 1), 0),
                PlanarBond((1, 0, 2, 3, 5, 4), 0),
                PlanarBond((4, 5, 3, 2, 1, 0), 0),
            ],
        ),
        "planar_bond_unknown": (
            [
                PlanarBondStereo((1, 0, 2, 3, 5, 4), None),
                PlanarBondStereo((5, 4, 3, 2, 1, 0), None),
            ],
            [
                PlanarBond((1, 0, 2, 3, 5, 4), None),
                PlanarBond((5, 4, 3, 2, 1, 0), None),
            ],
        ),
        "square_planar_specified": (
            [
                SquarePlanarStereo((6, 0, 1, 2, 3), 0),
                SquarePlanarStereo((6, 1, 2, 3, 0), 0),
                SquarePlanarStereo((6, 1, 0, 2, 3), 0),
            ],
            [
                SquarePlanar((6, 0, 1, 2, 3), 0),
                SquarePlanar((6, 1, 2, 3, 0), 0),
                SquarePlanar((6, 1, 0, 2, 3), 0),
            ],
        ),
        "square_planar_unknown": (
            [
                SquarePlanarStereo((6, 0, 1, 2, 3), None),
                SquarePlanarStereo((6, 2, 3, 0, 1), None),
            ],
            [
                SquarePlanar((6, 0, 1, 2, 3), None),
                SquarePlanar((6, 2, 3, 0, 1), None),
            ],
        ),
        "trigonal_bipyramidal_specified": (
            [
                TrigonalBipyramidalStereo((6, 0, 1, 2, 3, 4), 1),
                TrigonalBipyramidalStereo((6, 1, 0, 2, 3, 4), -1),
                TrigonalBipyramidalStereo((6, 1, 0, 3, 4, 2), -1),
                TrigonalBipyramidalStereo((6, 1, 0, 3, 4, 2), 1),
            ],
            [
                TrigonalBipyramidal((6, 0, 1, 2, 3, 4), 1),
                TrigonalBipyramidal((6, 1, 0, 2, 3, 4), -1),
                TrigonalBipyramidal((6, 1, 0, 3, 4, 2), -1),
                TrigonalBipyramidal((6, 1, 0, 3, 4, 2), 1),
            ],
        ),
        "trigonal_bipyramidal_unknown": (
            [
                TrigonalBipyramidalStereo((6, 0, 1, 2, 3, 4), None),
                TrigonalBipyramidalStereo((6, 1, 0, 3, 4, 2), None),
            ],
            [
                TrigonalBipyramidal((6, 0, 1, 2, 3, 4), None),
                TrigonalBipyramidal((6, 1, 0, 3, 4, 2), None),
            ],
        ),
        "octahedral_specified": (
            [
                OctahedralStereo((6, 0, 1, 2, 3, 4, 5), 1),
                OctahedralStereo((6, 0, 1, 5, 2, 3, 4), 1),
                OctahedralStereo((6, 1, 0, 2, 3, 4, 5), -1),
                OctahedralStereo((6, 1, 0, 2, 3, 4, 5), 1),
            ],
            [
                Octahedral((6, 0, 1, 2, 3, 4, 5), 1),
                Octahedral((6, 0, 1, 5, 2, 3, 4), 1),
                Octahedral((6, 1, 0, 2, 3, 4, 5), -1),
                Octahedral((6, 1, 0, 2, 3, 4, 5), 1),
            ],
        ),
        "octahedral_unknown": (
            [
                OctahedralStereo((6, 0, 1, 2, 3, 4, 5), None),
                OctahedralStereo((6, 5, 4, 3, 2, 1, 0), None),
            ],
            [
                Octahedral((6, 0, 1, 2, 3, 4, 5), None),
                Octahedral((6, 5, 4, 3, 2, 1, 0), None),
            ],
        ),
        "atrop_bond_specified": (
            [
                AtropBondStereo((0, 1, 2, 3, 4, 5), 1),
                AtropBondStereo((1, 0, 2, 3, 5, 4), 1),
                AtropBondStereo((1, 0, 2, 3, 4, 5), -1),
                AtropBondStereo((1, 0, 2, 3, 4, 5), 1),
            ],
            [
                AtropBond((0, 1, 2, 3, 4, 5), 1),
                AtropBond((1, 0, 2, 3, 5, 4), 1),
                AtropBond((1, 0, 2, 3, 4, 5), -1),
                AtropBond((1, 0, 2, 3, 4, 5), 1),
            ],
        ),
        "atrop_bond_unknown": (
            [
                AtropBondStereo((0, 1, 2, 3, 4, 5), None),
                AtropBondStereo((4, 5, 3, 2, 0, 1), None),
            ],
            [
                AtropBond((0, 1, 2, 3, 4, 5), None),
                AtropBond((4, 5, 3, 2, 0, 1), None),
            ],
        ),
    }

    checks = {}
    for name, (synkit_values, upstream_values) in fixtures.items():
        synkit_matrix = _equality_matrix(synkit_values)
        upstream_matrix = _equality_matrix(upstream_values)
        checks[name] = {
            "status": "PASS" if synkit_matrix == upstream_matrix else "FAIL",
            "synkit": synkit_matrix,
            "stereomolgraph": upstream_matrix,
        }

    atom_centered_synkit = (
        TetrahedralStereo,
        SquarePlanarStereo,
        TrigonalBipyramidalStereo,
        OctahedralStereo,
    )

    def synkit_descriptor_graph(descriptor: Any) -> nx.Graph:
        graph = nx.Graph()
        elements = ("C", "F", "Cl", "Br", "I", "N", "O")
        atoms = tuple(sorted(descriptor.dependencies))
        graph.add_nodes_from(
            (atom, {"atom_map": atom, "element": element})
            for atom, element in zip(atoms, elements)
        )
        if isinstance(descriptor, atom_centered_synkit):
            graph.add_edges_from(
                (descriptor.atoms[0], reference) for reference in descriptor.atoms[1:]
            )
        else:
            left, right = descriptor.atoms[2:4]
            graph.add_edge(left, right)
            graph.add_edges_from(
                (left, reference) for reference in descriptor.atoms[:2]
            )
            graph.add_edges_from(
                (right, reference) for reference in descriptor.atoms[4:]
            )
        graph.graph["stereo_descriptors"] = {descriptor_id(descriptor): descriptor}
        return graph

    atom_centered_upstream = (
        Tetrahedral,
        SquarePlanar,
        TrigonalBipyramidal,
        Octahedral,
    )

    def upstream_descriptor_graph(descriptor: Any) -> Any:
        graph = StereoMolGraph()
        elements = ("C", "F", "Cl", "Br", "I", "N", "O")
        atoms = tuple(sorted(value for value in descriptor.atoms if value is not None))
        for atom, element in zip(atoms, elements):
            graph.add_atom(atom, element)
        if isinstance(descriptor, atom_centered_upstream):
            for reference in descriptor.atoms[1:]:
                graph.add_bond(descriptor.atoms[0], reference)
            graph.set_atom_stereo(descriptor)
        else:
            left, right = descriptor.atoms[2:4]
            graph.add_bond(left, right)
            for reference in descriptor.atoms[:2]:
                graph.add_bond(left, reference)
            for reference in descriptor.atoms[4:]:
                graph.add_bond(right, reference)
            graph.set_bond_stereo(descriptor)
        return graph

    graph_fixtures = {
        "tetrahedral": (
            TetrahedralStereo,
            Tetrahedral,
            (0, 1, 2, 3, 4),
            1,
        ),
        "square_planar": (
            SquarePlanarStereo,
            SquarePlanar,
            (0, 1, 2, 3, 4),
            0,
        ),
        "trigonal_bipyramidal": (
            TrigonalBipyramidalStereo,
            TrigonalBipyramidal,
            (0, 1, 2, 3, 4, 5),
            1,
        ),
        "octahedral": (
            OctahedralStereo,
            Octahedral,
            (0, 1, 2, 3, 4, 5, 6),
            1,
        ),
        "planar_bond": (
            PlanarBondStereo,
            PlanarBond,
            (0, 1, 2, 3, 4, 5),
            0,
        ),
        "atrop_bond": (
            AtropBondStereo,
            AtropBond,
            (0, 1, 2, 3, 4, 5),
            1,
        ),
    }
    for name, (synkit_type, upstream_type, atoms, parity) in graph_fixtures.items():
        synkit_value = synkit_type(atoms, parity)
        upstream_value = upstream_type(atoms, parity)
        mapping = {atom: atom + 10 for atom in atoms}
        synkit_equivalent = synkit_value.relabel(mapping)
        upstream_equivalent = upstream_type(
            tuple(mapping[atom] for atom in atoms),
            parity,
        )
        if name == "square_planar":
            distinct_atoms = (atoms[0], atoms[2], atoms[1], *atoms[3:])
            synkit_distinct = synkit_type(distinct_atoms, parity)
            upstream_distinct = upstream_type(distinct_atoms, parity)
        elif name == "planar_bond":
            distinct_atoms = (atoms[1], atoms[0], *atoms[2:])
            synkit_distinct = synkit_type(distinct_atoms, parity)
            upstream_distinct = upstream_type(distinct_atoms, parity)
        else:
            synkit_distinct = synkit_value.invert()
            upstream_distinct = upstream_type(atoms, -parity)

        synkit_graph_results = [
            stereo_isomorphic(
                synkit_descriptor_graph(synkit_value),
                synkit_descriptor_graph(candidate),
            )
            for candidate in (synkit_equivalent, synkit_distinct)
        ]
        upstream_graph_results = [
            upstream_descriptor_graph(upstream_value).is_isomorphic(
                upstream_descriptor_graph(candidate)
            )
            for candidate in (upstream_equivalent, upstream_distinct)
        ]
        checks[f"{name}_graph_isomorphism"] = {
            "status": (
                "PASS" if synkit_graph_results == upstream_graph_results else "FAIL"
            ),
            "synkit": synkit_graph_results,
            "stereomolgraph": upstream_graph_results,
        }

    commit = _git_commit(repository)
    return {
        "oracle": "StereoMolGraph",
        "expected_commit": PINNED_COMMIT,
        "observed_commit": commit,
        "commit_status": "PASS" if commit == PINNED_COMMIT else "FAIL",
        "scope": [
            "tetrahedral",
            "square_planar",
            "trigonal_bipyramidal",
            "octahedral",
            "planar_bond",
            "atrop_bond",
            "unknown_parity",
        ],
        "deferred": [
            "rigid_bond_33",
            "rigid_bond_23",
            "rigid_bond_13",
            "rigid_bond_12",
            "coordinate_inference",
        ],
        "classified_differences": [
            {
                "feature": "unknown-versus-specified descriptor equality",
                "stereomolgraph": (
                    "parity=None participates as a wildcard in descriptor equality"
                ),
                "synkit": (
                    "descriptor equality preserves unknown as a distinct stored "
                    "value; wildcard behavior is selected explicitly by "
                    "stereo_query_mode or a per-rule query policy"
                ),
                "classification": "intentional rule-query semantic boundary",
            }
        ],
        "checks": checks,
        "status": (
            "PASS"
            if commit == PINNED_COMMIT
            and all(value["status"] == "PASS" for value in checks.values())
            else "FAIL"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repository",
        type=Path,
        help="Local StereoMolGraph checkout pinned to the expected commit.",
    )
    args = parser.parse_args()
    report = run_conformance(args.repository.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
