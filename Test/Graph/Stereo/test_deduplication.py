"""Sprint 11 deduplication-consumer contract tests."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
)
from synkit.Mechanism import (
    MechanismRecord,
    MechanisticStep,
    StereoDescriptor,
    StereoEffect,
    mechanism_equivalent,
)
from synkit.Rule.Compose._identity import cluster_rule_objects
from synkit.Rule.syn_rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from synkit.Synthesis.reactor_utils import _get_unique_aam

ATOM_DESCRIPTORS = (
    TetrahedralStereo((1, 2, 3, 4, 5), 1),
    SquarePlanarStereo((1, 2, 3, 4, 5), 0),
    TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
    OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
)
SIX_CLASS_DESCRIPTORS = (
    *ATOM_DESCRIPTORS,
    PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
    AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
)


def _descriptor_graph(descriptor, *, its=False):
    graph = nx.Graph()
    for index, atom_map in enumerate(sorted(descriptor.dependencies)):
        graph.add_node(
            atom_map,
            atom_map=(atom_map, atom_map) if its else atom_map,
            element=f"E{index}",
            aromatic=False,
            hcount=0,
            charge=0,
            radical=0,
            lone_pairs=0,
            valence_electrons=0,
            present=(True, True) if its else True,
        )
    if isinstance(
        descriptor,
        (
            TetrahedralStereo,
            SquarePlanarStereo,
            TrigonalBipyramidalStereo,
            OctahedralStereo,
        ),
    ):
        edges = ((descriptor.atoms[0], atom) for atom in descriptor.atoms[1:])
    else:
        left, right = descriptor.atoms[2:4]
        edges = (
            (left, right),
            *((left, atom) for atom in descriptor.atoms[:2]),
            *((right, atom) for atom in descriptor.atoms[4:]),
        )
    for left, right in edges:
        if isinstance(left, int) and isinstance(right, int):
            graph.add_edge(
                left,
                right,
                order=(1, 1) if its else 1,
                standard_order=(1, 1) if its else 1,
                kekule_order=(1, 1) if its else 1,
                sigma_order=(1, 1) if its else 1,
                pi_order=(0, 0) if its else 0,
            )
    registry = {descriptor_id(descriptor): descriptor}
    graph.graph["stereo_descriptors"] = (
        {"reactant": {}, "product": registry} if its else registry
    )
    return graph


def _relabel_graph(graph, descriptor, *, offset=100, its=False):
    mapping = {atom_map: atom_map + offset for atom_map in descriptor.dependencies}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    for old, new in mapping.items():
        relabeled.nodes[new]["atom_map"] = (new, new) if its else new
    mapped_descriptor = descriptor.relabel(mapping)
    registry = {descriptor_id(mapped_descriptor): mapped_descriptor}
    relabeled.graph["stereo_descriptors"] = (
        {"reactant": {}, "product": registry} if its else registry
    )
    return relabeled


def _different_orientation(descriptor):
    if isinstance(descriptor, SquarePlanarStereo):
        atoms = (
            descriptor.atoms[0],
            descriptor.atoms[2],
            descriptor.atoms[1],
            *descriptor.atoms[3:],
        )
        return SquarePlanarStereo(atoms, 0)
    return descriptor.invert()


@pytest.mark.parametrize("descriptor", SIX_CLASS_DESCRIPTORS)
def test_graph_cluster_collapses_relabels_but_preserves_stereo(descriptor):
    original = _descriptor_graph(descriptor)
    relabeled = _relabel_graph(original, descriptor)
    changed = _descriptor_graph(_different_orientation(descriptor))
    clusterer = GraphCluster()

    clusters, _assignments = clusterer.iterative_cluster(
        [original, relabeled, changed],
        nodeMatch=clusterer.nodeMatch,
        edgeMatch=clusterer.edgeMatch,
    )

    assert clusters == [{0, 1}, {2}]


def test_graph_cluster_keeps_unknown_specified_and_absent_separate():
    specified_descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    unknown_descriptor = TetrahedralStereo(specified_descriptor.atoms, None)
    specified = _descriptor_graph(specified_descriptor)
    unknown = _descriptor_graph(unknown_descriptor)
    absent = _descriptor_graph(specified_descriptor)
    absent.graph["stereo_descriptors"] = {}
    clusterer = GraphCluster()

    clusters, _assignments = clusterer.iterative_cluster(
        [specified, unknown, absent],
        nodeMatch=clusterer.nodeMatch,
        edgeMatch=clusterer.edgeMatch,
    )

    assert clusters == [{0}, {1}, {2}]


def test_serialized_aam_dedup_collapses_relabels_but_preserves_enantiomers():
    original = "[CH3:1][C@H:2]([F:3])[OH:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"
    relabeled = "[CH3:11][C@H:12]([F:13])[OH:14]>>" "[CH3:11][C@H:12]([F:13])[OH:14]"
    enantiomer = "[CH3:1][C@@H:2]([F:3])[OH:4]>>" "[CH3:1][C@@H:2]([F:3])[OH:4]"

    assert _get_unique_aam([original, relabeled, enantiomer]) == [
        original,
        enantiomer,
    ]


@pytest.mark.parametrize("descriptor", SIX_CLASS_DESCRIPTORS)
def test_reactor_its_dedup_collapses_only_stereo_equivalent_relabels(descriptor):
    original = _descriptor_graph(descriptor, its=True)
    relabeled = _relabel_graph(original, descriptor, its=True)
    changed = _descriptor_graph(_different_orientation(descriptor), its=True)

    unique = SynReactor._deduplicate_structural_its([original, relabeled, changed])

    assert unique == [original, changed]


def _mapped_rule(offset=0, *, weights=(0.7, 0.3)):
    values = [value + offset for value in range(1, 5)]
    carbon, center, fluorine, oxygen = values
    reaction = (
        f"[CH3:{carbon}][CH+:{center}][F:{fluorine}].[OH-:{oxygen}]>>"
        f"[CH3:{carbon}][C@H:{center}]([F:{fluorine}])[OH:{oxygen}]"
    )
    return SynRule.from_smart(
        reaction,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={
            f"atom:{center}": {
                "kind": "ENANTIOMERIC_MIXTURE",
                "weights": list(weights),
            }
        },
    )


def test_rule_identity_and_composition_clustering_are_map_invariant():
    original = _mapped_rule()
    relabeled = _mapped_rule(10)
    different_weights = _mapped_rule(weights=(0.6, 0.4))

    assert original == relabeled
    assert hash(original) == hash(relabeled)
    assert original != different_weights
    assert len(cluster_rule_objects([original, relabeled, different_weights])) == 2


def _mechanism_record(descriptor, *, equivalent=False):
    fragments = [
        "[C:1]",
        "[F:2]",
        "[Cl:3]",
        "[Br:4]",
        "[I:5]",
        "[N:6]",
        "[O:7]",
    ]
    reaction = ".".join(fragments)
    if equivalent:
        permutation = (
            (0, 3, 1, 2, *range(4, len(descriptor.atoms)))
            if isinstance(descriptor, TetrahedralStereo)
            else descriptor._PERMUTATIONS[1]
        )
        descriptor = type(descriptor)(
            tuple(descriptor.atoms[index] for index in permutation),
            descriptor.parity,
            descriptor.provenance,
        )
    envelope = StereoDescriptor(
        descriptor.descriptor_class,
        descriptor.atoms,
        descriptor.parity,
    )
    effect = StereoEffect(
        ("atom", descriptor.atoms[0]),
        "PRESERVE",
        before=envelope,
    )
    return MechanismRecord(
        f"{reaction}>>{reaction}",
        (MechanisticStep("stereo", (), (effect,)),),
    )


@pytest.mark.parametrize("descriptor", ATOM_DESCRIPTORS)
def test_mechanism_equivalence_uses_relative_stereo_and_stereo_only_events(
    descriptor,
):
    original = _mechanism_record(descriptor)
    equivalent = _mechanism_record(descriptor, equivalent=True)
    changed = _mechanism_record(_different_orientation(descriptor))

    assert mechanism_equivalent(original, equivalent, level="events")
    assert mechanism_equivalent(original, equivalent, level="trajectory")
    assert not mechanism_equivalent(original, changed, level="events")
    assert not mechanism_equivalent(original, changed, level="trajectory")
