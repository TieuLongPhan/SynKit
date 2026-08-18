"""Soundness and completeness checks for verified RBL search."""

from __future__ import annotations

import copy
import hashlib
import json

import networkx as nx
import pytest

from Experiment.RBL.exhaustive_overlap_oracle import (
    brute_force_typed_overlaps,
)
from synkit.Graph.Fusion import (
    DEFAULT_INTERFACE_EDGE_KEYS,
    DEFAULT_INTERFACE_NODE_KEYS,
)
from synkit.Synthesis.RBL import (
    FusionIssueCode,
    RBLEngine,
    read_rbl_proof,
    validate_strict_rbl_candidate,
)
from synkit.Synthesis.RBL.overlap import (
    TypedOverlapLimits,
    enumerate_typed_overlaps,
)

from Test.Synthesis.RBL.test_fusion_contract import CASES


def _node(element: str, atom_map: int = 0) -> dict[str, object]:
    return {
        "element": element,
        "atom_map": atom_map,
        "isotope": 0,
        "charge": 0,
        "radical": 0,
        "aromatic": False,
        "lone_pairs": 0,
    }


def _tiny_pair(*, anchored: bool = False) -> tuple[nx.Graph, nx.Graph]:
    forward = nx.Graph()
    backward = nx.Graph()
    forward.add_node("c", **_node("C", 1 if anchored else 0))
    forward.add_node("o", **_node("O", 2 if anchored else 0))
    forward.add_edge("c", "o", order=1.0)
    backward.add_node(10, **_node("C", 1 if anchored else 0))
    backward.add_node(20, **_node("O", 2 if anchored else 0))
    backward.add_edge(10, 20, order=1.0)
    return forward, backward


@pytest.mark.parametrize("anchored", [False, True])
def test_incremental_typed_overlap_search_matches_independent_oracle(
    anchored: bool,
) -> None:
    forward, backward = _tiny_pair(anchored=anchored)
    expected = brute_force_typed_overlaps(
        forward,
        backward,
        node_keys=DEFAULT_INTERFACE_NODE_KEYS,
        edge_keys=DEFAULT_INTERFACE_EDGE_KEYS,
    )
    observed = enumerate_typed_overlaps(
        forward,
        backward,
        node_keys=DEFAULT_INTERFACE_NODE_KEYS,
        edge_keys=DEFAULT_INTERFACE_EDGE_KEYS,
    )

    normalize = lambda mappings: {  # noqa: E731 - compact comparison helper
        tuple(sorted(mapping.items(), key=lambda item: repr(item[0])))
        for mapping in mappings
    }
    assert observed.certificate.complete is True
    assert normalize(observed.mappings) == normalize(expected)
    if not anchored:
        assert {len(mapping) for mapping in observed.mappings} == {1, 2}

    relabeled = enumerate_typed_overlaps(
        nx.relabel_nodes(forward, {"c": "left-C", "o": "left-O"}),
        nx.relabel_nodes(backward, {10: "right-C", 20: "right-O"}),
        node_keys=DEFAULT_INTERFACE_NODE_KEYS,
        edge_keys=DEFAULT_INTERFACE_EDGE_KEYS,
    )
    assert relabeled.certificate.complete is True
    assert sorted(map(len, relabeled.mappings)) == sorted(map(len, expected))


def test_typed_overlap_limit_is_explicit_and_never_claims_exhaustion() -> None:
    forward, backward = _tiny_pair()
    observed = enumerate_typed_overlaps(
        forward,
        backward,
        node_keys=DEFAULT_INTERFACE_NODE_KEYS,
        edge_keys=DEFAULT_INTERFACE_EDGE_KEYS,
        limits=TypedOverlapLimits(max_states=1, max_overlaps=10),
    )

    assert observed.certificate.complete is False
    assert observed.certificate.termination == "state_limit"
    assert observed.certificate.states_explored == 1


def test_strict_reconstruction_rejects_fragment_embedding_impostors() -> None:
    validation = validate_strict_rbl_candidate("C>>C", "C.O>>C.O")

    assert validation.valid is False
    assert FusionIssueCode.REACTANT_COMPONENT_NOT_PRESERVED not in {
        issue.code for issue in validation.issues
    }
    assert FusionIssueCode.UNMAPPED_MATERIAL_ATOM in {
        issue.code for issue in validation.issues
    }

    changed = validate_strict_rbl_candidate("CO>>CC", "N>>CC.O")
    assert changed.valid is False
    assert FusionIssueCode.REACTANT_COMPONENT_NOT_PRESERVED in {
        issue.code for issue in changed.issues
    }


def test_open_boundary_requires_the_exact_declared_resource_delta() -> None:
    reaction = "[Na+:1]>>[Na:1]"
    closed = validate_strict_rbl_candidate(reaction, reaction)
    open_valid = validate_strict_rbl_candidate(
        reaction,
        reaction,
        boundary="open",
        environment_delta={"formal_charge": -1},
    )
    open_wrong = validate_strict_rbl_candidate(
        reaction,
        reaction,
        boundary="open",
        environment_delta={"formal_charge": 1},
    )

    assert FusionIssueCode.CHARGE_IMBALANCE in {
        issue.code for issue in closed.issues
    }
    assert open_valid.valid is True
    assert FusionIssueCode.ENVIRONMENT_DELTA_MISMATCH in {
        issue.code for issue in open_wrong.issues
    }


def _digest(payload: dict[str, object]) -> str:
    normalized = copy.deepcopy(payload)
    normalized.pop("document_digest", None)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_replayable_proof_round_trip_and_structural_tamper_detection() -> None:
    _name, reaction, template, _expected = CASES[1]
    engine = RBLEngine(mode="verified").process(reaction, template)
    payload = engine.rbl_proofs[0].to_dict()

    restored = read_rbl_proof(json.dumps(payload))
    assert restored.replay().valid is True
    assert all(
        graph.graph["application_provenance"]["mapping"]
        for graph in (*engine.forward_its, *engine.backward_its)
    )

    forged = copy.deepcopy(payload)
    forged["mapping"][0][1], forged["mapping"][1][1] = (
        forged["mapping"][1][1],
        forged["mapping"][0][1],
    )
    forged["document_digest"] = _digest(forged)
    with pytest.raises(ValueError, match="proof replay failed"):
        read_rbl_proof(forged)


def test_verified_result_separates_outcomes_from_derivation_proofs() -> None:
    _name, reaction, template, _expected = CASES[0]
    engine = RBLEngine(mode="verified").process(reaction, template)
    result = engine.result

    assert result["search_status"] == "FOUND"
    assert result["complete"] is True
    assert result["reason_incomplete"] == []
    assert len(result["outcomes"]) == 1
    assert result["outcomes"][0]["proof_count"] >= 1
    assert result["outcomes"][0]["proof_digests"]


def test_verified_engine_reports_typed_overlap_limit_as_incomplete() -> None:
    _name, reaction, template, _expected = CASES[0]
    engine = RBLEngine(
        mode="verified",
        overlap_max_states=1,
    ).process(reaction, template)

    assert engine.result["complete"] is False
    assert engine.result["search_status"] == "INCOMPLETE"
    assert "typed_overlap_limit" in engine.result["reason_incomplete"]
