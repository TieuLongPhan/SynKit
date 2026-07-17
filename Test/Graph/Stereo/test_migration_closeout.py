import networkx as nx

from synkit.Graph.Fusion import FusionInterface, construct_pushout
from synkit.Graph.Morphism import StereoEffect
from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoChange,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    candidate_mapping_stereo_matches,
    classify_stereo_change,
    descriptor_id,
    descriptor_query_matches,
    summarize_stereo_comparisons,
)


DESCRIPTORS = (
    TetrahedralStereo((1, 2, 3, 4, 5), 1),
    SquarePlanarStereo((1, 2, 3, 4, 5), 0),
    TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
    OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
    PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
    AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
)


def _descriptor_graph(descriptor):
    graph = nx.Graph()
    for reference in descriptor.dependencies:
        graph.add_node(reference, element="C", atom_map=reference)
    if descriptor.descriptor_class in {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }:
        for ligand in descriptor.atoms[1:]:
            graph.add_edge(descriptor.atoms[0], ligand, order=1.0)
    else:
        left, right = descriptor.atoms[2:4]
        graph.add_edge(left, right, order=1.0)
        for ligand in descriptor.atoms[:2]:
            graph.add_edge(left, ligand, order=1.0)
        for ligand in descriptor.atoms[4:]:
            graph.add_edge(right, ligand, order=1.0)
    graph.graph["stereo_descriptors"] = {
        descriptor_id(descriptor): descriptor
    }
    return graph


def test_complete_six_geometry_migration_audit_has_no_unregistered_divergence():
    diagnostics = []
    for descriptor in DESCRIPTORS:
        opposite = descriptor.invert()
        descriptor.same_configuration(
            descriptor,
            semantics="compare",
            diagnostics=diagnostics,
        )
        descriptor_query_matches(
            descriptor,
            descriptor,
            semantics="compare",
            diagnostics=diagnostics,
        )

        pattern = _descriptor_graph(descriptor)
        relabeling = {node: node + 100 for node in pattern}
        host = nx.relabel_nodes(pattern, relabeling, copy=True)
        for old, new in relabeling.items():
            host.nodes[new]["atom_map"] = new
        host.graph["stereo_descriptors"] = {
            descriptor_id(descriptor.relabel(relabeling)): descriptor.relabel(
                relabeling
            )
        }
        assert candidate_mapping_stereo_matches(
            pattern,
            host,
            relabeling,
            semantics="compare",
            diagnostics=diagnostics,
        )

        classify_stereo_change(
            descriptor,
            opposite,
            semantics="compare",
            diagnostics=diagnostics,
        )
        change = StereoChange.from_endpoints(descriptor, opposite)
        change.apply_to(
            descriptor,
            semantics="compare",
            diagnostics=diagnostics,
        )
        change.reverse(
            semantics="compare",
            diagnostics=diagnostics,
        )

        backward = nx.relabel_nodes(pattern, relabeling, copy=True)
        for old, new in relabeling.items():
            backward.nodes[new]["atom_map"] = new
        backward_descriptor = opposite.relabel(relabeling)
        backward.graph["stereo_descriptors"] = {
            descriptor_id(backward_descriptor): backward_descriptor
        }
        interface = FusionInterface.from_mapping(
            pattern,
            backward,
            relabeling,
        ).with_stereo_effects(
            {
                (
                    "backward",
                    "state",
                    descriptor_id(backward_descriptor),
                ): StereoEffect.INVERT
            }
        )
        construction = construct_pushout(
            pattern,
            backward,
            interface,
            stereo_semantics="compare",
            stereo_diagnostics=diagnostics,
        )
        assert all(
            evidence.replay().valid
            for evidence in construction.provenance.stereo_evidence
        )

    report = summarize_stereo_comparisons(diagnostics)

    assert report["unexpected_divergences"] == 0
    assert report["expected_divergences"] == 4
    assert report["total"] == 62
    assert {
        stage: values["total"]
        for stage, values in report["stages"].items()
    } == {
        "candidate_mapping": 6,
        "descriptor_identity": 6,
        "descriptor_query": 6,
        "fusion_stereo_relation": 12,
        "fusion_stereo_transport": 12,
        "reaction_stereo_application": 6,
        "reaction_stereo_change": 6,
        "reaction_stereo_relation": 2,
        "reaction_stereo_reverse": 6,
    }
    assert set(report["stages"]) == {
        "candidate_mapping",
        "descriptor_identity",
        "descriptor_query",
        "fusion_stereo_relation",
        "fusion_stereo_transport",
        "reaction_stereo_application",
        "reaction_stereo_change",
        "reaction_stereo_relation",
        "reaction_stereo_reverse",
    }
