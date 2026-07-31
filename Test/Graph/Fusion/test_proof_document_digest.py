import copy

import pytest

from synkit.Graph.Fusion import FusionProofError, read_fusion_proof

from Test.Graph.Fusion.test_proof_v2 import _candidate


def test_current_fusion_proof_binds_the_complete_document():
    payload = _candidate().to_dict()
    document = read_fusion_proof(payload)

    assert len(payload["document_digest"]) == 64
    assert document.document_digest_verified is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("rsmi", "forged"),
        ("interface", {}),
        ("wildcard_substitution", [{"forged": True}]),
        ("validation", [{"valid": False}]),
        ("endpoint_proof", {}),
        ("canonical_signature", "forged"),
        ("proof_digest", "0" * 64),
        ("score", {"added_nodes": 999}),
        ("document_digest", "f" * 64),
    ],
)
def test_every_top_level_fusion_proof_field_is_tamper_evident(
    field,
    replacement,
):
    corrupted = copy.deepcopy(_candidate().to_dict())
    corrupted[field] = replacement

    with pytest.raises(FusionProofError):
        read_fusion_proof(corrupted)


def test_non_stereo_provenance_tamper_is_detected():
    corrupted = copy.deepcopy(_candidate().to_dict())
    corrupted["provenance"]["node_sources"] = []

    with pytest.raises(FusionProofError) as captured:
        read_fusion_proof(corrupted)

    assert captured.value.issue_code == "FUSION_PROOF_DOCUMENT_DIGEST_MISMATCH"
