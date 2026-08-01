import networkx as nx
import pytest

from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_destruction import ITSDestruction
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.rc_extractor import RCExtractor
from synkit.Graph.ITS.stereo import StereoITSValidationError
from synkit.Graph.Stereo import (
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    FrameworkFrame,
    FrameworkStereo,
    HelicalStereo,
    PlanarChiralityStereo,
    StereoRefusalCode,
    TetrahedralStereo,
    descriptor_id,
)


def _tetrahedral_graph(descriptor=None):
    graph = nx.Graph()
    graph.add_node(1, element="C", hcount=0, lone_pairs=0)
    for atom in (2, 3, 4, 5):
        graph.add_node(atom, element="C", hcount=0, lone_pairs=0)
        graph.add_edge(
            1,
            atom,
            order=1.0,
            sigma_order=1.0,
            pi_order=0.0,
        )
    if descriptor is not None:
        graph.graph["stereo_descriptors"] = {"atom:1": descriptor}
    return graph


def test_stereo_only_reaction_center_survives_strict_its_and_reversion():
    before = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    after = before.invert()
    reactant = _tetrahedral_graph(before)
    product = _tetrahedral_graph(after)

    its = ITSConstruction.construct(
        reactant,
        product,
        stereo_validation="strict",
    )
    minimal = RCExtractor().extract(its, include_context_edges=False)
    left, right = ITSDestruction(its).decompose()
    reverted = ITSReverter(its)

    assert all(data["standard_order"] == 0 for *_edge, data in its.edges(data=True))
    assert its.graph["stereo_changes"]["atom:1"].change == "INVERTED"
    assert set(minimal.nodes) == {1, 2, 3, 4, 5}
    assert not minimal.edges
    assert minimal.graph["stereo_changes"]["atom:1"].change == "INVERTED"
    assert left.graph["stereo_descriptors"] == {"atom:1": before}
    assert right.graph["stereo_descriptors"] == {"atom:1": after}
    assert reverted.to_reactant_graph().graph["stereo_descriptors"] == {
        "atom:1": before
    }
    assert reverted.to_product_graph().graph["stereo_descriptors"] == {"atom:1": after}


def test_transition_only_descriptor_is_first_class_and_reversible():
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    reactant = _tetrahedral_graph()
    product = _tetrahedral_graph()
    transition = _tetrahedral_graph(descriptor)

    its = ITSConstruction.construct(
        reactant,
        product,
        transition_graph=transition,
        stereo_validation="strict",
    )
    projected = ITSReverter(its).to_transition_state_graph()

    assert its.graph["stereo_changes"]["atom:1"].change == "FLEETING"
    assert its.graph["stereo_descriptors"]["transition"] == {"atom:1": descriptor}
    assert projected.graph["stereo_descriptors"] == {"atom:1": descriptor}
    assert projected.graph["stereo_projection"] == "transition"


def test_strict_construction_rejects_dangling_product_reference():
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    reactant = _tetrahedral_graph(descriptor)
    product = _tetrahedral_graph(descriptor)
    product.remove_node(5)

    with pytest.raises(StereoITSValidationError) as error:
        ITSConstruction.construct(
            reactant,
            product,
            stereo_validation="strict",
        )

    assert error.value.refusals[0].code is StereoRefusalCode.INVALID_REFERENCE
    assert "ligand atom map 5 is absent" in error.value.refusals[0].detail


def test_strict_construction_rejects_stale_virtual_h_resource():
    descriptor = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    reactant = _tetrahedral_graph()
    reactant.remove_node(5)
    reactant.graph["stereo_descriptors"] = {"atom:1": descriptor}

    with pytest.raises(StereoITSValidationError) as error:
        ITSConstruction.construct(
            reactant,
            reactant,
            stereo_validation="strict",
        )

    assert "requires 1 hcount resource" in error.value.refusals[0].detail


def test_preserve_mode_keeps_graph_only_descriptor_for_compatibility():
    descriptor = TetrahedralStereo((1, 2, 3, 4, 99), 1)
    graph = _tetrahedral_graph()
    graph.graph["stereo_descriptors"] = {"atom:1": descriptor}

    its = ITSConstruction.construct(graph, graph)

    assert its.graph["stereo_validation"] == "preserve"
    assert its.graph["stereo_descriptors"]["reactant"] == {"atom:1": descriptor}


def test_strict_extended_path_support_requires_every_path_edge():
    descriptor = HelicalStereo((1, 2, 3, 4), 1)
    graph = nx.path_graph((1, 2, 3, 4))
    for node in graph:
        graph.nodes[node].update(
            element="C",
            hcount=0,
            lone_pairs=0,
        )
    for left, right in graph.edges:
        graph.edges[left, right].update(
            order=1.0,
            sigma_order=1.0,
            pi_order=0.0,
        )
    graph.graph["stereo_descriptors"] = {"path:1-2-3-4": descriptor}

    valid = ITSConstruction.construct(
        graph,
        graph,
        stereo_validation="strict",
    )
    assert valid.graph["stereo_changes"]["path:1-2-3-4"].change == "RETAINED"

    broken = graph.copy()
    broken.remove_edge(2, 3)
    with pytest.raises(StereoITSValidationError) as error:
        ITSConstruction.construct(
            broken,
            broken,
            stereo_validation="strict",
        )
    assert "helical path edges are absent: 2-3" in error.value.refusals[0].detail


def test_strict_support_covers_extended_and_framework_families():
    cumulene = CumuleneAxisStereo(
        (1, 2, 3),
        ((4, 5), (6, 7)),
        1,
    )
    extended = ExtendedCisTransStereo(
        (1, 2, 3, 8),
        ((4, 5), (6, 7)),
        0,
    )
    planar = PlanarChiralityStereo((1, 2, 3, 8), 9, 1)
    framework = FrameworkStereo(
        frozenset({1, 2, 3, 4, 5}),
        (FrameworkFrame(1, (2, 3, 4, 5), 1),),
        1,
    )
    for descriptor in (cumulene, extended, planar, framework):
        graph = nx.Graph()
        for atom in descriptor.dependencies:
            graph.add_node(
                atom,
                element="C",
                hcount=0,
                lone_pairs=0,
            )
        if isinstance(descriptor, CumuleneAxisStereo):
            graph.add_edges_from(((1, 2), (2, 3), (1, 4), (1, 5), (3, 6), (3, 7)))
        elif isinstance(descriptor, ExtendedCisTransStereo):
            graph.add_edges_from(
                ((1, 2), (2, 3), (3, 8), (1, 4), (1, 5), (8, 6), (8, 7))
            )
        elif isinstance(descriptor, FrameworkStereo):
            graph.add_edges_from((1, atom) for atom in (2, 3, 4, 5))
        for left, right in graph.edges:
            graph.edges[left, right].update(
                order=1.0,
                sigma_order=1.0,
                pi_order=0.0,
            )
        key = descriptor_id(descriptor)
        graph.graph["stereo_descriptors"] = {key: descriptor}

        its = ITSConstruction.construct(
            graph,
            graph,
            stereo_validation="strict",
        )

        assert its.graph["stereo_changes"][key].change == "RETAINED"


def test_legacy_itsgraph_threads_transition_and_strict_options():
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    graph = _tetrahedral_graph()
    transition = _tetrahedral_graph(descriptor)

    its = ITSConstruction.ITSGraph(
        graph,
        graph,
        transition_graph=transition,
        stereo_validation="strict",
    )

    assert its.graph["stereo_changes"]["atom:1"].change == "FLEETING"
    assert its.graph["stereo_validation"] == "strict"
