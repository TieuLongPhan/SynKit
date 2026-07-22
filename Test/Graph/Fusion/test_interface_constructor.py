"""Verified fusion-interface and pushout-construction laws."""

import copy

import networkx as nx
import pytest

from synkit.Graph.Fusion import (
    FusionConstructionError,
    FusionConstructionIssueCode,
    FusionInterface,
    FusionInterfaceError,
    FusionInterfaceIssueCode,
    construct_pushout,
    fusion_candidate_from_construction,
    graphs_exactly_equivalent,
)
from synkit.Graph.Morphism import WildcardRole
from synkit.Graph.Morphism import StereoEffect
from synkit.Graph.Stereo import TetrahedralStereo


def _sources() -> tuple[nx.Graph, nx.Graph, dict[int, int]]:
    forward = nx.Graph()
    forward.add_node(1, element="C", charge=0, radical=0, aromatic=False)
    forward.add_node(2, element="O", charge=0, radical=0, aromatic=False)
    forward.add_node(3, element="F", charge=0, radical=0, aromatic=False)
    forward.add_edge(1, 2, order=1.0)
    forward.add_edge(1, 3, order=1.0)

    backward = nx.Graph()
    backward.add_node(10, element="C", charge=0, radical=0, aromatic=False)
    backward.add_node(20, element="O", charge=0, radical=0, aromatic=False)
    backward.add_node(30, element="Cl", charge=0, radical=0, aromatic=False)
    backward.add_edge(10, 20, order=1.0)
    backward.add_edge(10, 30, order=1.0)
    return forward, backward, {1: 10, 2: 20}


def test_interface_builds_two_injective_sprint14_morphisms() -> None:
    forward, backward, mapping = _sources()
    interface = FusionInterface.from_mapping(forward, backward, mapping)

    assert interface.forward_morphism.mapping == {0: 1, 1: 2}
    assert interface.backward_morphism.mapping == {0: 10, 1: 20}
    assert len(interface.edges) == 1


def test_pushout_has_complete_provenance_and_commuting_inclusions() -> None:
    forward, backward, mapping = _sources()
    forward_before = copy.deepcopy(forward)
    backward_before = copy.deepcopy(backward)
    interface = FusionInterface.from_mapping(forward, backward, mapping)

    result = construct_pushout(forward, backward, interface)

    assert result.graph.number_of_nodes() == 4
    assert result.graph.number_of_edges() == 3
    assert result.endpoint_certificate.forward_nodes_verified == 3
    assert result.endpoint_certificate.backward_nodes_verified == 3
    assert len(result.provenance.node_sources) == 4
    assert len(result.provenance.edge_sources) == 3
    assert nx.utils.graphs_equal(forward, forward_before)
    assert nx.utils.graphs_equal(backward, backward_before)


def test_operand_permutation_yields_same_graph_and_proof_digest() -> None:
    forward, backward, mapping = _sources()
    first_interface = FusionInterface.from_mapping(forward, backward, mapping)
    second_interface = FusionInterface.from_mapping(
        backward, forward, {right: left for left, right in mapping.items()}
    )
    first = fusion_candidate_from_construction(
        construct_pushout(forward, backward, first_interface)
    )
    second = fusion_candidate_from_construction(
        construct_pushout(backward, forward, second_interface)
    )

    assert graphs_exactly_equivalent(first.its, second.its)
    assert first.proof_digest == second.proof_digest


def test_relabeling_sources_preserves_graph_identity_and_proof_digest() -> None:
    forward, backward, mapping = _sources()
    original_interface = FusionInterface.from_mapping(forward, backward, mapping)
    original = fusion_candidate_from_construction(
        construct_pushout(forward, backward, original_interface)
    )

    f_labels = {1: "fc", 2: "fo", 3: "ff"}
    b_labels = {10: "bc", 20: "bo", 30: "bcl"}
    relabeled_forward = nx.relabel_nodes(forward, f_labels, copy=True)
    relabeled_backward = nx.relabel_nodes(backward, b_labels, copy=True)
    relabeled_mapping = {
        f_labels[left]: b_labels[right] for left, right in mapping.items()
    }
    relabeled_interface = FusionInterface.from_mapping(
        relabeled_forward,
        relabeled_backward,
        relabeled_mapping,
    )
    relabeled = fusion_candidate_from_construction(
        construct_pushout(
            relabeled_forward,
            relabeled_backward,
            relabeled_interface,
        )
    )

    assert relabeled.canonical_signature == original.canonical_signature
    assert relabeled.proof_digest == original.proof_digest


def test_concrete_node_and_edge_corruption_fail_before_construction() -> None:
    forward, backward, mapping = _sources()
    backward.nodes[10]["charge"] = 1
    with pytest.raises(FusionInterfaceError) as node_error:
        FusionInterface.from_mapping(forward, backward, mapping)
    assert node_error.value.issues[0].code is FusionInterfaceIssueCode.NODE_CONFLICT

    backward.nodes[10]["charge"] = 0
    backward.edges[10, 20]["order"] = 2.0
    with pytest.raises(FusionInterfaceError) as edge_error:
        FusionInterface.from_mapping(forward, backward, mapping)
    assert edge_error.value.issues[0].code is FusionInterfaceIssueCode.EDGE_CONFLICT


def test_wildcard_roles_and_owner_incidence_cannot_be_conflated() -> None:
    forward = nx.Graph()
    forward.add_node(1, element="*", wildcard_role="attachment_port", owner=2)
    forward.add_node(2, element="C")
    backward = nx.Graph()
    backward.add_node(10, element="*", wildcard_role="radical_completion")
    backward.add_node(20, element="C")
    with pytest.raises(FusionInterfaceError) as role_error:
        FusionInterface.from_mapping(forward, backward, {1: 10, 2: 20})
    assert role_error.value.issues[0].code is (
        FusionInterfaceIssueCode.WILDCARD_CONFLICT
    )

    backward.nodes[10]["wildcard_role"] = "attachment_port"
    backward.nodes[10]["owner"] = 999
    with pytest.raises(FusionInterfaceError) as owner_error:
        FusionInterface.from_mapping(forward, backward, {1: 10, 2: 20})
    assert owner_error.value.issues[0].code is (
        FusionInterfaceIssueCode.OWNER_OUTSIDE_INTERFACE
    )


@pytest.mark.parametrize("role", tuple(WildcardRole))
def test_every_wildcard_role_has_a_valid_typed_interface(role: WildcardRole) -> None:
    forward = nx.Graph()
    backward = nx.Graph()
    forward_attrs = {"element": "*", "wildcard_role": role.value}
    backward_attrs = {"element": "*", "wildcard_role": role.value}
    mapping = {1: 10}
    if role in {
        WildcardRole.ATTACHMENT_PORT,
        WildcardRole.STEREO_LIGAND_PORT,
    }:
        forward_attrs["owner"] = 2
        backward_attrs["owner"] = 20
        if role is WildcardRole.STEREO_LIGAND_PORT:
            forward_attrs["stereo_slot"] = 0
            backward_attrs["stereo_slot"] = 0
        forward.add_node(2, element="C")
        backward.add_node(20, element="C")
        forward.add_edge(1, 2, order=1.0)
        backward.add_edge(10, 20, order=1.0)
        mapping[2] = 20
    forward.add_node(1, **forward_attrs)
    backward.add_node(10, **backward_attrs)

    interface = FusionInterface.from_mapping(forward, backward, mapping)
    assert interface.substitutions[0].role is role


@pytest.mark.parametrize(
    "role,concrete,valid",
    (
        (WildcardRole.QUERY_ATOM, "C", True),
        (WildcardRole.HYDROGEN_COMPLETION, "H", True),
        (WildcardRole.HYDROGEN_COMPLETION, "C", False),
    ),
)
def test_wildcard_to_concrete_resolution_checks_chemical_domain(
    role: WildcardRole,
    concrete: str,
    valid: bool,
) -> None:
    forward = nx.Graph()
    forward.add_node(1, element="*", wildcard_role=role.value)
    backward = nx.Graph()
    backward.add_node(10, element=concrete, charge=0, radical=0)

    if valid:
        interface = FusionInterface.from_mapping(forward, backward, {1: 10})
        result = construct_pushout(forward, backward, interface)
        assert result.graph.nodes[1]["element"] == concrete
    else:
        with pytest.raises(FusionInterfaceError) as error:
            FusionInterface.from_mapping(forward, backward, {1: 10})
        assert error.value.issues[0].code is (
            FusionInterfaceIssueCode.WILDCARD_CONFLICT
        )


def test_direct_constructor_rechecks_interface_target_graphs() -> None:
    forward, backward, mapping = _sources()
    interface = FusionInterface.from_mapping(forward, backward, mapping)
    backward.add_node(999, element="H")

    with pytest.raises(FusionConstructionError) as error:
        construct_pushout(forward, backward, interface)
    assert error.value.issues[0].code is FusionConstructionIssueCode.TARGET_MISMATCH


def test_noninvertible_replay_refusal_is_proof_evidence() -> None:
    forward, backward, mapping = _sources()
    interface = FusionInterface.from_mapping(forward, backward, mapping)
    result = construct_pushout(
        forward,
        backward,
        interface,
        replay_status="refused_non_invertible",
        replay_reason="unspecified_stereo_effect",
    )

    assert result.endpoint_certificate.replay_status == "refused_non_invertible"
    assert result.endpoint_certificate.replay_reason == "unspecified_stereo_effect"


def test_opposite_stereo_frames_require_an_explicit_interface_effect() -> None:
    forward = nx.Graph()
    backward = nx.Graph()
    mapping = {}
    for node, element in enumerate(("C", "F", "Cl", "Br", "I"), start=1):
        forward.add_node(node, element=element)
        backward.add_node(node + 10, element=element)
        mapping[node] = node + 10
    for ligand in range(2, 6):
        forward.add_edge(1, ligand, order=1.0)
        backward.add_edge(11, ligand + 10, order=1.0)
    forward.graph["stereo_descriptors"] = {
        "atom:1": TetrahedralStereo((1, 2, 3, 4, 5), 1)
    }
    backward.graph["stereo_descriptors"] = {
        "atom:11": TetrahedralStereo((11, 12, 13, 14, 15), -1)
    }
    interface = FusionInterface.from_mapping(forward, backward, mapping)

    with pytest.raises(FusionConstructionError) as conflict:
        construct_pushout(forward, backward, interface)
    assert conflict.value.issues[0].code is (
        FusionConstructionIssueCode.STEREO_CONFLICT
    )

    declared = interface.with_stereo_effects(
        {("backward", "state", "atom:11"): StereoEffect.INVERT}
    )
    result = construct_pushout(forward, backward, declared)
    assert result.graph.graph["stereo_descriptors"]["state"]["atom:1"].parity == 1
