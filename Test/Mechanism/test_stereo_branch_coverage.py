"""RX12 reviewed claim-to-witness coverage gate."""

from __future__ import annotations

import json
from pathlib import Path

from synkit.Graph.Stereo import (
    CONFIGURED_STEREO_DESCRIPTOR_CLASSES,
    StereoDeterminacy,
    StereoEvidenceKind,
    StereoLifecycle,
    StereoPopulation,
    StereoReactionDecision,
    StereoReactionSemantics,
    StereoRefusalCode,
)
from synkit.Graph.Stereo.couplings import (
    SUPPORTED_STEREO_COUPLING_KINDS,
)

ROOT = Path(__file__).parents[2]
DATA = ROOT / "Experiment/Lewis/mech_path/Data/MechanismBench"


def _manifest():
    return json.loads(
        (DATA / "stereo_branch_coverage.json").read_text(encoding="utf-8")
    )


def _case_ids(filename):
    payload = json.loads((DATA / filename).read_text(encoding="utf-8"))
    return {case["case_id"] for case in payload["cases"]}


def test_coverage_manifest_schema_review_and_case_fields_are_complete():
    payload = _manifest()

    assert payload["schema"] == "MechanismBench-stereo-branch-coverage-v1"
    assert payload["status"] == "reviewed_claim_index"
    assert payload["review"]["decision"] == ("ACCEPT_WITH_CLAIM_BOUNDARIES")
    assert payload["review"]["chemical_rationale"]
    assert payload["review"]["mathematical_rationale"]
    assert payload["review"]["independence_boundary"]
    for case in payload["cases"]:
        assert {
            "forward_expectation",
            "reverse_expectation",
            "chemical_rationale",
            "mathematical_rationale",
            "limitations",
            "reviewer_decision",
            "license",
        } <= set(case)
        decision = StereoReactionDecision(
            assertion=StereoReactionSemantics.from_dict(case["assertion"])
        )
        assert decision.accepted
        assert StereoReactionDecision.from_dict(decision.to_dict()) == decision


def test_every_public_semantic_branch_has_accepted_and_boundary_witnesses():
    coverage = _manifest()["coverage"]
    expected = {
        "lifecycle": {item.value for item in StereoLifecycle},
        "population": {item.value for item in StereoPopulation},
        "determinacy": {item.value for item in StereoDeterminacy},
        "evidence": {item.value for item in StereoEvidenceKind},
        "descriptor_family": set(CONFIGURED_STEREO_DESCRIPTOR_CLASSES),
        "coupling_kind": set(SUPPORTED_STEREO_COUPLING_KINDS),
        "coupling_relation": {"SYN", "ANTI"},
        "verification_mode": {"off", "endpoint", "stepwise"},
        "route": {
            "forward",
            "reverse",
            "exact_rule",
            "generic_rule",
            "its",
            "reactor",
            "fusion",
            "mechanism",
            "interchange",
        },
        "refusal_code": {item.value for item in StereoRefusalCode},
    }

    assert set(coverage) == set(expected)
    for axis, branches in expected.items():
        assert set(coverage[axis]) == branches
        for branch, witnesses in coverage[axis].items():
            assert len(witnesses) == 2, (axis, branch)
            assert all(
                isinstance(witness, str) and ":" in witness for witness in witnesses
            )
            assert witnesses[0] != witnesses[1]


def test_all_claim_references_resolve_to_reviewed_cases_or_executable_tests():
    payload = _manifest()
    known_cases = {
        "stereo": _case_ids("stereo.json"),
        "coupling": _case_ids("stereo_couplings.json"),
        "electrocyclic": _case_ids("electrocyclic_machinery.json"),
        "coverage": {case["case_id"] for case in payload["cases"]},
    }
    test_files = {
        alias: ROOT / path for alias, path in payload["test_evidence"].items()
    }

    for path in test_files.values():
        assert path.is_file()
        assert "def test_" in path.read_text(encoding="utf-8")

    references = {
        witness
        for branches in payload["coverage"].values()
        for witnesses in branches.values()
        for witness in witnesses
    }
    for reference in references:
        source, identity, *_qualifiers = reference.split(":")
        if source in known_cases:
            assert identity in known_cases[source], reference
        elif source == "test":
            assert identity in test_files, reference
        elif source == "mechanism":
            assert identity == "polar"
        else:
            raise AssertionError(f"Unknown evidence source: {reference}")


def test_serialization_variants_do_not_count_as_chemistry_evidence():
    payload = _manifest()
    excluded = payload["excluded_from_case_count"]
    references = {
        witness
        for branches in payload["coverage"].values()
        for witnesses in branches.values()
        for witness in witnesses
    }

    assert excluded == [f"SER-{index:02d}" for index in range(1, 15)]
    assert not set(excluded) & references


def test_deferred_families_name_exact_blocker_and_promotion_oracle():
    deferred = _manifest()["deferred_families"]

    assert {item["family"] for item in deferred} == {
        "isotope_pseudoasymmetry",
        "enhanced_stereo_population",
        "ring_relative_locked_elimination",
        "facial_pericyclic_selectivity",
        "advanced_allene_spiro_coordination",
    }
    for item in deferred:
        assert len(item["blocker"]) >= 40
        assert len(item["required_to_promote"]) >= 40
