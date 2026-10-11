"""Independent registration truth, real merging defects and information boundaries."""

from dataclasses import replace

from Experiment.Stereo.Canonicalization import registration
from Experiment.Stereo.Canonicalization.registration import boundary_examples, collection, workload


def test_both_registration_methods_reproduce_independent_global_groups():
    rows, evidence = collection(32)
    assert all(value["reference"]["status"] == "complete" for value in evidence)
    for method in ("pairwise", "certificates"):
        result = workload(rows, method, 10)
        assert result["status"] == "complete"
        assert result["partition_passed"]
        assert result["observed_partition"] == result["expected_partition"]
    assert any(len(group) > 1 for group in result["expected_partition"])
    assert len(result["expected_partition"]) > 1


def test_constant_certificates_cannot_pass_the_registration_truth_gate(monkeypatch):
    original = registration.canonicalize_configured_registry

    def constant(*args, **kwargs):
        return replace(original(*args, **kwargs), canonical_code="constant identity defect")

    monkeypatch.setattr(registration, "canonicalize_configured_registry", constant)
    rows, _ = collection(8)
    result = workload(rows, "certificates", 10)
    assert result["status"] == "complete"
    assert not result["partition_passed"]
    assert result["group_count"] == 1


def test_a_pairwise_always_equal_defect_fails_the_same_independent_partition(monkeypatch):
    monkeypatch.setattr(registration, "stereo_isomorphic", lambda *_args, **_kwargs: True)
    rows, _ = collection(8)
    result = workload(rows, "pairwise", 10)
    assert not result["partition_passed"]
    assert result["group_count"] == 1


def test_profile_sensitive_identity_and_strict_incomplete_refusal_are_separate():
    result = boundary_examples()
    assert result["passed"]
    assert {row["profile"]: row["observed_equal"] for row in result["profile_pair"]} == {
        "chemical": True, "lewis_state": False, "acs_topology": True,
    }
    assert result["strict_partial_input_refuses"]
    assert result["strict_partial_input_status"] == "incomplete"
