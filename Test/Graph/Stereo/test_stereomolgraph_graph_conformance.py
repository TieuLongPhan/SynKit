"""Always-on replay of the pinned six-class graph-isomorphism report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import pytest

from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    stereo_isomorphic,
)

REPORT_PATH = (
    Path(__file__).parents[3]
    / "Data"
    / "Conformance"
    / "stereomolgraph_graph_conformance.json"
)
REPORT = json.loads(REPORT_PATH.read_text())

DESCRIPTORS = {
    "tetrahedral": TetrahedralStereo((0, 1, 2, 3, 4), 1),
    "square_planar": SquarePlanarStereo((0, 1, 2, 3, 4), 0),
    "trigonal_bipyramidal": TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1),
    "octahedral": OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1),
    "planar_bond": PlanarBondStereo((0, 1, 2, 3, 4, 5), 0),
    "atrop_bond": AtropBondStereo((0, 1, 2, 3, 4, 5), 1),
}
ATOM_CENTERED = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
)


def _graph(descriptor):
    graph = nx.Graph()
    elements = ("C", "F", "Cl", "Br", "I", "N", "O")
    atoms = tuple(sorted(descriptor.dependencies))
    graph.add_nodes_from(
        (
            atom,
            {
                "atom_map": atom,
                "element": element,
                "charge": 0,
                "lone_pairs": 0,
                "radical": 0,
                "aromatic": False,
                "hcount": 0,
            },
        )
        for atom, element in zip(atoms, elements)
    )
    if isinstance(descriptor, ATOM_CENTERED):
        graph.add_edges_from(
            (descriptor.atoms[0], reference, {"order": 1})
            for reference in descriptor.atoms[1:]
        )
    else:
        left, right = descriptor.atoms[2:4]
        graph.add_edge(left, right, order=1)
        graph.add_edges_from(
            (left, reference, {"order": 1}) for reference in descriptor.atoms[:2]
        )
        graph.add_edges_from(
            (right, reference, {"order": 1}) for reference in descriptor.atoms[4:]
        )
    graph.graph["stereo_descriptors"] = {descriptor_id(descriptor): descriptor}
    return graph


def _distinct(descriptor):
    if isinstance(descriptor, SquarePlanarStereo):
        atoms = (
            descriptor.atoms[0],
            descriptor.atoms[2],
            descriptor.atoms[1],
            *descriptor.atoms[3:],
        )
        return SquarePlanarStereo(atoms, 0)
    return descriptor.invert()


@pytest.mark.parametrize("name", tuple(DESCRIPTORS))
def test_pinned_graph_isomorphism_result_replays_without_upstream_runtime(name):
    descriptor = DESCRIPTORS[name]
    mapping = {atom: atom + 10 for atom in descriptor.dependencies}
    observed = [
        stereo_isomorphic(_graph(descriptor), _graph(candidate))
        for candidate in (descriptor.relabel(mapping), _distinct(descriptor))
    ]

    assert observed == [True, False]
    assert observed == REPORT["checks"][name]["synkit"]
    assert observed == REPORT["checks"][name]["stereomolgraph"]
    assert REPORT["checks"][name]["status"] == "PASS"


def test_graph_conformance_report_is_pinned_and_scope_bounded():
    assert REPORT["expected_commit"] == ("2189f610f23eaaf992e2e01a12ea4d0532496601")
    assert REPORT["observed_commit"] == REPORT["expected_commit"]
    assert REPORT["commit_status"] == REPORT["descriptor_matrix_status"] == "PASS"
    assert set(REPORT["checks"]) == set(DESCRIPTORS)
    assert REPORT["classified_differences"] == [
        {
            "feature": "unknown-versus-specified descriptor equality",
            "stereomolgraph": (
                "parity=None participates as a wildcard in descriptor equality"
            ),
            "synkit": (
                "unknown is a distinct stored state; wildcard behavior is "
                "selected explicitly by the rule query policy"
            ),
            "classification": "intentional rule-query semantic boundary",
        }
    ]
    assert REPORT["status"] == "PASS"


def test_optional_pinned_upstream_replay_matches_the_permanent_report():
    configured = os.environ.get("STEREOMOLGRAPH_REPOSITORY")
    repository = Path(configured) if configured else Path("/tmp/StereoMolGraph")
    if not (repository / "src" / "stereomolgraph").is_dir():
        pytest.skip("pinned StereoMolGraph checkout is unavailable")

    from tools.stereo_conformance import run_conformance

    observed = run_conformance(repository)

    assert observed["observed_commit"] == REPORT["observed_commit"]
    assert observed["status"] == "PASS"
    for name, expected in REPORT["checks"].items():
        check = observed["checks"][f"{name}_graph_isomorphism"]
        assert check["synkit"] == expected["synkit"]
        assert check["stereomolgraph"] == expected["stereomolgraph"]
        assert check["status"] == expected["status"]
