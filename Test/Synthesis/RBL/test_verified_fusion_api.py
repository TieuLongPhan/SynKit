"""Sprint 15 proof-bearing RBL compatibility projections."""

import json

import pytest

import synkit.Synthesis.RBL.engine as rbl_engine_module
from synkit.Graph.Fusion import FUSION_PROOF_SCHEMA, FusionCandidate
from synkit.Graph.Fusion.identity import graphs_exactly_equivalent
from synkit.IO import rsmi_to_its
from synkit.Synthesis.RBL import RBLEngine, SearchScope

from Test.Synthesis.RBL.test_fusion_contract import CASES


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_full_rbl_proof_candidates_are_a_sound_subset_of_outputs(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    engine = RBLEngine(mode="full").process(reaction, template)

    assert engine.fused_rsmis
    assert all(
        isinstance(candidate, FusionCandidate) for candidate in engine.fusion_candidates
    )
    assert {candidate.rsmi for candidate in engine.fusion_candidates} <= set(
        engine.fused_rsmis
    )
    for candidate in engine.fusion_candidates:
        assert candidate.proof_schema == FUSION_PROOF_SCHEMA
        assert candidate.forward_morphism.source == candidate.backward_morphism.source
        assert candidate.validation[0]["valid"] is True
        assert len(candidate.proof_digest) == 64
        assert candidate.endpoint_proof.forward_nodes_verified > 0
        assert candidate.endpoint_proof.backward_nodes_verified > 0

    json.dumps(engine.result)


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_uncapped_unpruned_rbl_search_exhausts_only_its_declared_mcs_scope(
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

    assert engine.result["fusion_search"]["complete"] is False
    assert engine.result["fusion_search"]["complete_within_mapping_scope"] is True
    assert engine.result["fusion_search"]["termination"] == ("mapping_scope_exhausted")
    assert engine.result["fusion_search"]["mapping_scope"] == (
        "maximum_common_subgraphs"
    )
    assert engine.result["fusion_search"]["mappings_truncated"] == 0
    assert engine.result["fusion_search"]["overlap_scope"] == (
        "maximum_common_subgraphs"
    )
    assert engine.result["fusion_search"][
        "globally_complete_over_all_overlaps"
    ] is False
    assert engine.result["fusion_search"]["incomplete_reasons"] == [
        "maximum_common_subgraphs_only"
    ]


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
    assert engine.implicit_temp is False
    assert engine.explicit_h is True
    assert engine.result["verified_fusion_mode"] is True
    assert engine.result["fusion_search"]["complete"] is True
    assert engine.result["fusion_search"]["complete_within_mapping_scope"] is True
    assert {candidate.rsmi for candidate in engine.fusion_candidates} == set(
        engine.fused_rsmis
    )
    assert engine.result["fusion_search"]["proof_candidates"] == len(
        engine.fusion_candidates
    )


def test_proof_interface_is_stricter_than_configurable_matcher_labels() -> None:
    engine = RBLEngine(
        mode="verified",
        node_attrs=("element",),
    )

    assert engine.node_attrs == ["element"]
    assert {"element", "isotope", "radical", "lone_pairs"} <= set(
        engine.interface_node_attrs
    )
    assert "hcount" not in engine.interface_node_attrs


def test_verified_mode_preserves_explicit_mapped_hydrogen_in_template() -> None:
    template = "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]"

    engine = RBLEngine(mode="verified").prepare_template(template)

    hydrogen_nodes = [
        data
        for _, data in engine.template_its.nodes(data=True)
        if data.get("element") == "H"
    ]
    assert len(hydrogen_nodes) == 1
    assert hydrogen_nodes[0]["atom_map"] == 4


def test_verified_mode_overrides_unsafe_hydrogen_flags() -> None:
    engine = RBLEngine(
        mode="verified",
        implicit_temp=True,
        explicit_h=False,
    )

    assert engine.implicit_temp is False
    assert engine.explicit_h is True


def test_verified_transesterification_resolves_typed_attachment_ports() -> None:
    _name, reaction, template, _legacy_expected = CASES[1]
    rule_correct = (
        "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH3:6]."
        "[CH3:9][CH2:8][O:7][H:10]>>"
        "[CH3:1][CH2:2][C:3](=[O:4])[O:7][CH2:8][CH3:9]."
        "[O:5]([CH3:6])[H:10]"
    )

    engine = RBLEngine(mode="verified").process(
        reaction,
        template,
        replace_wc=True,
    )

    assert len(engine.fused_rsmis) == 1
    assert len(engine.fusion_candidates) == 1
    assert engine.result["fusion_search"]["interface_completion"] == (
        "all_typed_leaf_port_assignments"
    )
    assert graphs_exactly_equivalent(
        rsmi_to_its(engine.fused_rsmis[0], format="tuple"),
        rsmi_to_its(rule_correct, format="tuple"),
    )


def test_verified_esterification_has_a_typed_hydrogen_materialization_proof() -> None:
    _name, reaction, template, _expected = CASES[0]
    prepared = RBLEngine().prepare_template(template).template_its

    engine = RBLEngine(mode="verified").process(
        reaction,
        prepared,
        replace_wc=True,
    )

    assert len(engine.fused_rsmis) == 1
    assert len(engine.fusion_candidates) == 1
    proof = engine.fusion_candidates[0].validation[0]["evidence"]["postprocess_proof"]
    assert proof["kind"] == "typed_wildcard_hydrogen_materialization"
    assert proof["normalized_digest"] == proof["target_digest"]


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


def test_fast_fusion_falls_back_to_bounded_ranked_pushout_fusion() -> None:
    class FusionOnlyEngine(RBLEngine):
        def _quick_check(self, rsmi: str, template: object) -> None:
            return None

        def _early_stop_on_nonwildcard(self, *args: object, **kwargs: object) -> bool:
            return False

    _name, reaction, template, _expected = CASES[0]
    engine = FusionOnlyEngine(mode="fast_fusion", max_pairs=2).process(
        reaction,
        template,
    )

    assert engine.fused_rsmis
    assert engine.result["search_policy"]["scope"] == "bounded_fusion"
    assert engine.result["fusion_search"]["pair_candidates"] == 4
    assert engine.result["fusion_search"]["pairs_explored"] <= 2
    assert engine.result["fusion_search"]["pair_ordering"] == "wl_no_cutoff"
    assert engine.result["fusion_search"]["fusion_backend"] == ("categorical_pushout")
    assert engine.result["fusion_search"]["complete"] is False


def test_fast_paths_only_compatibility_flag_still_skips_fusion() -> None:
    engine = RBLEngine(fast_paths_only=True)

    assert engine.search_policy.scope is SearchScope.FAST_PATHS_ONLY
    assert engine.max_pairs is None


def test_fast_track_never_invokes_fusion() -> None:
    class NoFusionEngine(RBLEngine):
        def _fuse_and_postprocess(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("fast_track must not invoke fusion")

    _name, reaction, template, _expected = CASES[0]
    engine = NoFusionEngine(mode="fast_track").process(reaction, template)

    assert engine.result["search_policy"]["scope"] == "fast_paths_only"
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
        item.get("source") == "postprocess"
        and not any(
            issue["code"]
            in {
                "FUSION_SERIALIZATION_FAILED",
                "FUSION_POSTPROCESS_FAILED",
            }
            for issue in item.get("issues", ())
        )
        for item in engine.diagnostics["fusion"]
    )

    assert engine.fused_rsmis
    assert calls == postprocess_validations
