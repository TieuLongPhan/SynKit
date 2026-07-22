"""Sprint 15 proof-bearing RBL compatibility projections."""

import json

import pytest

import synkit.Synthesis.Reactor.rbl_engine as rbl_engine_module
from synkit.Graph.Fusion import FUSION_PROOF_SCHEMA, FusionCandidate
from synkit.Synthesis.Reactor.rbl_engine import RBLEngine

from Test.Synthesis.Reactor.test_rbl_fusion_contract import CASES


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_full_rbl_outputs_are_projections_of_proof_bearing_candidates(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    engine = RBLEngine(mode="full").process(reaction, template)

    assert engine.fusion_candidates
    assert all(
        isinstance(candidate, FusionCandidate) for candidate in engine.fusion_candidates
    )
    assert engine.fused_rsmis == [
        candidate.rsmi for candidate in engine.fusion_candidates
    ]
    assert len(engine.fused_its) == len(engine.fusion_candidates)
    for candidate in engine.fusion_candidates:
        assert candidate.proof_schema == FUSION_PROOF_SCHEMA
        assert candidate.forward_morphism.source == candidate.backward_morphism.source
        assert candidate.validation[0]["valid"] is True
        assert len(candidate.proof_digest) == 64
        assert candidate.endpoint_proof.forward_nodes_verified > 0
        assert candidate.endpoint_proof.backward_nodes_verified > 0

    json.dumps(engine.result)


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_uncapped_unpruned_rbl_search_declares_completeness(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    engine = RBLEngine(
        mode="full",
        prune_automorphisms=False,
        max_mappings_per_pair=0,
    ).process(reaction, template)

    assert engine.result["fusion_search"]["complete"] is True
    assert engine.result["fusion_search"]["termination"] == "exhaustive"
    assert engine.result["fusion_search"]["mappings_truncated"] == 0


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_verified_mode_selects_the_complete_mapping_profile(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    engine = RBLEngine(mode="verified").process(reaction, template)

    assert engine.prune_automorphisms is False
    assert engine.max_mappings_per_pair == 0
    assert engine.result["verified_fusion_mode"] is True
    assert engine.result["fusion_search"]["complete"] is True


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_rbl_proof_digests_are_repeatable(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    first = RBLEngine(mode="full").process(reaction, template)
    second = RBLEngine(mode="full").process(reaction, template)

    assert [candidate.proof_digest for candidate in first.fusion_candidates] == [
        candidate.proof_digest for candidate in second.fusion_candidates
    ]


def test_fast_path_compatibility_output_is_not_misreported_as_graph_fusion() -> None:
    class QuickEngine(RBLEngine):
        def _quick_check(self, rsmi: str, template: object) -> str:
            return rsmi

    engine = QuickEngine(mode="early_stop").process("CC>>CO", "[C:1]>>[C:1]")

    assert engine.fused_rsmis == ["CC>>CO"]
    assert engine.fusion_candidates == []
    assert engine.result["fusion_candidates"] == []
    assert engine.result["fusion_search"] == {}


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_candidate_proof_reuses_postprocess_endpoint_validation(
    monkeypatch: pytest.MonkeyPatch,
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    original = rbl_engine_module.validate_rbl_candidate
    calls = 0

    def counted_validation(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rbl_engine_module,
        "validate_rbl_candidate",
        counted_validation,
    )
    engine = RBLEngine(mode="full").process(reaction, template)
    postprocess_validations = sum(
        item.get("source") == "postprocess" for item in engine.diagnostics["fusion"]
    )

    assert engine.fusion_candidates
    assert calls == postprocess_validations
