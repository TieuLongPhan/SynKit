import copy

import networkx as nx
import pytest

from synkit.Graph.Fusion import (
    FUSION_PROOF_SCHEMA,
    LEGACY_FUSION_PROOF_SCHEMA,
    FusionCandidate,
    FusionConstructionError,
    FusionInterface,
    FusionProofError,
    construct_pushout,
    fusion_candidate_from_construction,
    fusion_candidates_exactly_equivalent,
    read_fusion_proof,
)
from synkit.Graph.Morphism import StereoEffect
from synkit.Graph.Stereo import StereoRelationKind, TetrahedralStereo
from synkit.Graph.Stereo import TrigonalBipyramidalStereo


def _stereo_sources(
    backward_parity: int = 1,
) -> tuple[nx.Graph, nx.Graph, dict[int, int]]:
    forward = nx.Graph()
    backward = nx.Graph()
    mapping = {}
    for node, element in enumerate(("C", "F", "Cl", "Br", "I"), start=1):
        forward.add_node(node, element=element, atom_map=node)
        backward.add_node(node + 10, element=element, atom_map=node + 10)
        mapping[node] = node + 10
    for ligand in range(2, 6):
        forward.add_edge(1, ligand, order=1.0)
        backward.add_edge(11, ligand + 10, order=1.0)
    forward.graph["stereo_descriptors"] = {
        "atom:1": TetrahedralStereo((1, 2, 3, 4, 5), 1)
    }
    backward.graph["stereo_descriptors"] = {
        "atom:11": TetrahedralStereo(
            (11, 12, 13, 14, 15),
            backward_parity,
        )
    }
    return forward, backward, mapping


def _candidate(backward_parity: int = 1) -> FusionCandidate:
    forward, backward, mapping = _stereo_sources(backward_parity)
    interface = FusionInterface.from_mapping(forward, backward, mapping)
    if backward_parity == -1:
        interface = interface.with_stereo_effects(
            {("backward", "state", "atom:11"): StereoEffect.INVERT}
        )
    return fusion_candidate_from_construction(
        construct_pushout(forward, backward, interface)
    )


def test_proof_v2_contains_replayable_local_orbit_evidence():
    candidate = _candidate()

    assert candidate.proof_schema == FUSION_PROOF_SCHEMA
    assert candidate.proof_schema == "synkit.fusion-proof/2"
    assert len(candidate.stereo_evidence) == 2
    for evidence in candidate.stereo_evidence:
        assert evidence.replay().valid
        assert evidence.source_to_interface.witness is not None
        assert evidence.interface_to_candidate.witness is not None
        assert evidence.direct_relation.witness is not None
        assert evidence.composed_witness is not None
        assert evidence.endpoint_orbit == (
            evidence.target_configuration.canonical_frame
        )

    serialized = candidate.to_dict()
    assert len(serialized["provenance"]["stereo_evidence"]) == 2
    assert read_fusion_proof(serialized).schema == FUSION_PROOF_SCHEMA


def test_declared_inversion_records_opposite_relation_and_replays():
    candidate = _candidate(-1)
    by_side = {evidence.side: evidence for evidence in candidate.stereo_evidence}

    assert by_side["forward"].interface_to_candidate.kind is (
        StereoRelationKind.EQUIVALENT
    )
    assert by_side["backward"].effect is StereoEffect.INVERT
    assert by_side["backward"].interface_to_candidate.kind is (
        StereoRelationKind.OPPOSITE
    )
    assert all(evidence.replay().valid for evidence in candidate.stereo_evidence)


def test_v2_reader_rejects_a_relation_label_without_witness():
    payload = _candidate().to_dict()
    corrupted = copy.deepcopy(payload)
    corrupted["provenance"]["stereo_evidence"][0][
        "interface_to_candidate"
    ]["witness"] = None

    with pytest.raises(FusionProofError) as excinfo:
        read_fusion_proof(corrupted)

    assert excinfo.value.issue_code == "FUSION_PROOF_STEREO_WITNESS_MISSING"


def test_v2_reader_rejects_a_present_but_false_witness():
    corrupted = _candidate().to_dict()
    corrupted["provenance"]["stereo_evidence"][0][
        "interface_to_candidate"
    ]["witness"] = [0, 2, 1, 3, 4]

    with pytest.raises(FusionProofError) as excinfo:
        read_fusion_proof(corrupted)

    assert excinfo.value.issue_code == "FUSION_PROOF_STEREO_WITNESS_REPLAY"


def test_v2_reader_rejects_forged_relation_endpoint_claims():
    corrupted = _candidate().to_dict()
    corrupted["provenance"]["stereo_evidence"][0][
        "direct_relation"
    ]["target_canonical"] = ["forged"]

    with pytest.raises(FusionProofError) as excinfo:
        read_fusion_proof(corrupted)

    assert excinfo.value.issue_code == "FUSION_PROOF_STEREO_WITNESS_REPLAY"


def test_proof_v1_remains_readable_as_a_compatibility_projection():
    current = _candidate().to_dict()
    legacy = copy.deepcopy(current)
    legacy["proof_schema"] = LEGACY_FUSION_PROOF_SCHEMA
    legacy["provenance"].pop("stereo_evidence")

    document = read_fusion_proof(legacy)

    assert document.schema == LEGACY_FUSION_PROOF_SCHEMA
    assert document.stereo_evidence == ()
    assert document.compatibility_projection["canonical_signature"] == (
        current["canonical_signature"]
    )
    assert document.to_dict()["proof_schema"] == LEGACY_FUSION_PROOF_SCHEMA


def test_orbit_evidence_participates_in_digest_and_exact_candidate_identity():
    retained = _candidate(1)
    inverted = _candidate(-1)

    assert retained.canonical_signature == inverted.canonical_signature
    assert retained.proof_digest != inverted.proof_digest
    assert not fusion_candidates_exactly_equivalent(retained, inverted)


def test_stereo_proof_digest_is_operand_and_map_relabeling_invariant():
    forward, backward, mapping = _stereo_sources(-1)
    interface = FusionInterface.from_mapping(forward, backward, mapping)
    interface = interface.with_stereo_effects(
        {("backward", "state", "atom:11"): StereoEffect.INVERT}
    )
    original = fusion_candidate_from_construction(
        construct_pushout(forward, backward, interface)
    )

    swapped_interface = FusionInterface.from_mapping(
        backward,
        forward,
        {right: left for left, right in mapping.items()},
    ).with_stereo_effects(
        {("forward", "state", "atom:11"): StereoEffect.INVERT}
    )
    swapped = fusion_candidate_from_construction(
        construct_pushout(backward, forward, swapped_interface)
    )

    forward_labels = {node: node + 100 for node in forward}
    backward_labels = {node: node + 200 for node in backward}
    relabeled_forward = nx.relabel_nodes(forward, forward_labels, copy=True)
    relabeled_backward = nx.relabel_nodes(backward, backward_labels, copy=True)
    relabeled_mapping = {
        forward_labels[left]: backward_labels[right]
        for left, right in mapping.items()
    }
    relabeled_interface = FusionInterface.from_mapping(
        relabeled_forward,
        relabeled_backward,
        relabeled_mapping,
    ).with_stereo_effects(
        {("backward", "state", "atom:11"): StereoEffect.INVERT}
    )
    relabeled = fusion_candidate_from_construction(
        construct_pushout(
            relabeled_forward,
            relabeled_backward,
            relabeled_interface,
        )
    )

    assert original.proof_digest == swapped.proof_digest
    assert original.proof_digest == relabeled.proof_digest


def test_declared_effect_without_source_descriptor_is_rejected():
    forward = nx.Graph()
    backward = nx.Graph()
    forward.add_node(1, element="C")
    backward.add_node(10, element="C")
    interface = FusionInterface.from_mapping(forward, backward, {1: 10})
    interface = interface.with_stereo_effects(
        {("forward", "state", "atom:1"): StereoEffect.INVERT}
    )

    with pytest.raises(FusionConstructionError, match="no source descriptor"):
        construct_pushout(forward, backward, interface)


def test_fusion_compare_mode_returns_orbit_graph_and_registered_audit():
    forward, backward, mapping = _stereo_sources(-1)
    interface = FusionInterface.from_mapping(
        forward,
        backward,
        mapping,
    ).with_stereo_effects(
        {("backward", "state", "atom:11"): StereoEffect.INVERT}
    )
    diagnostics = []

    orbit = construct_pushout(forward, backward, interface)
    compared = construct_pushout(
        forward,
        backward,
        interface,
        stereo_semantics="compare",
        stereo_diagnostics=diagnostics,
    )
    legacy = construct_pushout(
        forward,
        backward,
        interface,
        stereo_semantics="legacy",
    )

    assert compared.graph.graph["stereo_descriptors"] == (
        orbit.graph.graph["stereo_descriptors"]
    )
    assert legacy.graph.graph["stereo_descriptors"] == (
        orbit.graph.graph["stereo_descriptors"]
    )
    assert len(diagnostics) == 4
    assert all(comparison.registered for comparison in diagnostics)


def test_tbp_fusion_registers_only_the_nonbinary_relation_classification():
    forward = nx.Graph()
    backward = nx.Graph()
    mapping = {}
    elements = ("P", "F", "Cl", "Br", "I", "H")
    for node, element in enumerate(elements, start=1):
        forward.add_node(node, element=element, atom_map=node)
        backward.add_node(node + 10, element=element, atom_map=node + 10)
        mapping[node] = node + 10
    for ligand in range(2, 7):
        forward.add_edge(1, ligand, order=1.0)
        backward.add_edge(11, ligand + 10, order=1.0)
    forward.graph["stereo_descriptors"] = {
        "atom:1": TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1)
    }
    backward.graph["stereo_descriptors"] = {
        "atom:11": TrigonalBipyramidalStereo(
            (11, 12, 13, 14, 15, 16),
            -1,
        )
    }
    interface = FusionInterface.from_mapping(
        forward,
        backward,
        mapping,
    ).with_stereo_effects(
        {("backward", "state", "atom:11"): StereoEffect.INVERT}
    )
    diagnostics = []

    construction = construct_pushout(
        forward,
        backward,
        interface,
        stereo_semantics="compare",
        stereo_diagnostics=diagnostics,
    )

    assert all(evidence.replay().valid for evidence in construction.provenance.stereo_evidence)
    divergences = [item for item in diagnostics if not item.agreement]
    assert len(divergences) == 1
    assert divergences[0].stage == "fusion_stereo_relation"
    assert divergences[0].expected_divergence == (
        "nonbinary_orbit_reconfiguration"
    )
    assert divergences[0].registered
