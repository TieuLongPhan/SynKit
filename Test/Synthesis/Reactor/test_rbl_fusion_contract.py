"""Permanent compatibility and validation contracts for RBL fusion."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.IO import its_to_rsmi, rsmi_to_its
from synkit.Synthesis.Reactor.fusion_validation import (
    FusionIssueCode,
    WildcardRole,
    validate_endpoint_preservation,
    validate_fusion_rsmi,
    validate_rbl_candidate,
    validate_wildcard_mapping_roles,
)
from synkit.Synthesis.Reactor.rbl_engine import RBLEngine
from synkit.Synthesis.Reactor.rbl_policy import (
    RBLSearchPolicy,
    SearchScope,
    TerminationPolicy,
)

CASES = (
    (
        "esterification",
        "CCC(=O)(O)>>CCC(=O)OC",
        (
            "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]>>"
            "[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        ),
        (
            "[CH3:1][CH2:2][C:3](=[O:4])[OH:5].[CH3:6][O:7][H:8]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH3:6].[OH:7][H:8]"
        ),
    ),
    (
        "transesterification",
        "CCC(=O)OC>>CCC(=O)OCC",
        "[C:1][O:2].[O:3][H:4]>>[C:1][O:3].[O:2][H:4]",
        (
            "[CH3:1][CH2:2][C:3](=[O:4])[O:8][H:9]."
            "[OH:5][CH2:6][CH3:7]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:5][CH2:6][CH3:7]."
            "[OH:8][H:9]"
        ),
    ),
)


@pytest.mark.parametrize("name,reaction,template,expected", CASES)
def test_rbl_preserves_explicit_hydrogen_aam_contract(
    name: str,
    reaction: str,
    template: str,
    expected: str,
) -> None:
    prepared = RBLEngine().prepare_template(template).template_its
    engine = RBLEngine(mode="full").process(reaction, prepared, replace_wc=True)

    assert engine.fused_rsmis, f"{name}: no fused candidates"
    assert any(
        AAMValidator.smiles_check(candidate, expected)
        for candidate in engine.fused_rsmis
    ), f"{name}: no candidate is AAM-equivalent to the compatibility golden"
    assert all(
        validate_fusion_rsmi(candidate).valid for candidate in engine.fused_rsmis
    )
    assert engine.result["acceptance_policy"] == {
        "preserve_original_sides": ["products"],
        "relation": "component_injective_subgraph",
        "use_chirality": True,
    }


def test_balanced_isolated_wildcards_are_removed_symmetrically() -> None:
    rsmi = "[CH3:1].[*:7].[*:8]>>[CH3:1].[*:8].[*:9]"
    assert RBLEngine._strip_balanced_isolated_wildcards(rsmi) == (
        "[CH3:1].[*:7]>>[CH3:1].[*:9]"
    )


def test_fusion_validation_has_stable_machine_readable_issue_codes() -> None:
    validation = validate_fusion_rsmi("[CH3:1].[H:2]>>[CH3:1]")

    assert not validation.valid
    assert FusionIssueCode.SIDE_ONLY_STANDALONE_HYDROGEN in {
        issue.code for issue in validation.issues
    }
    payload = validation.to_dict()
    assert payload["valid"] is False
    assert payload["issues"][0]["code"].startswith("FUSION_")


def test_endpoint_preservation_returns_injective_embedding_proof() -> None:
    validation = validate_endpoint_preservation(
        "CC.O>>CCO",
        "CC.O.N>>CCO.Cl",
    )

    assert validation.valid
    proof = validation.evidence
    assert proof["matcher"] == "synkit.SubgraphSearchEngine"
    assert proof["stereo_policy"] == "synkit.relative_stereo_subgraph"
    assert len(proof["reactant_embeddings"]) == 2
    assert len(proof["product_embeddings"]) == 1
    assert (
        len({entry["candidate_component"] for entry in proof["reactant_embeddings"]})
        == 2
    )


def test_endpoint_preservation_rejects_a_changed_original_product() -> None:
    validation = validate_endpoint_preservation("CC>>CO", "CC>>CN")

    assert not validation.valid
    assert validation.issues[0].code == (FusionIssueCode.PRODUCT_ENDPOINT_NOT_PRESERVED)


def test_endpoint_preservation_respects_component_multiplicity() -> None:
    validation = validate_endpoint_preservation("C.C>>C", "CC>>C")

    assert not validation.valid
    assert validation.issues[0].code == (
        FusionIssueCode.REACTANT_ENDPOINT_NOT_PRESERVED
    )


def test_endpoint_preservation_ignores_atom_map_labels() -> None:
    validation = validate_endpoint_preservation(
        "C>>[CH3:1][OH:2]",
        "N>>[CH3:7][OH:9].Cl",
        required_sides=("products",),
    )

    assert validation.valid


def test_endpoint_preservation_respects_specified_stereo() -> None:
    validation = validate_endpoint_preservation(
        "C>>F[C@](Cl)(Br)I",
        "N>>F[C@@](Cl)(Br)I",
        required_sides=("products",),
    )

    assert not validation.valid
    assert validation.issues[0].code == (FusionIssueCode.PRODUCT_ENDPOINT_NOT_PRESERVED)


def test_rbl_can_request_conservative_preservation_of_both_sides() -> None:
    validation = validate_rbl_candidate(
        "CO>>CC",
        "N>>CC.O",
        preserve_sides=("reactants", "products"),
    )

    assert not validation.valid
    assert FusionIssueCode.REACTANT_ENDPOINT_NOT_PRESERVED in {
        issue.code for issue in validation.issues
    }


def test_wildcard_role_inventory_is_explicit() -> None:
    assert len(set(WildcardRole)) == 5
    assert WildcardRole.QUERY_ATOM != WildcardRole.HYDROGEN_COMPLETION


@pytest.mark.parametrize("role", tuple(WildcardRole))
def test_each_wildcard_role_rejects_conflation(role: WildcardRole) -> None:
    roles = tuple(WildcardRole)
    incompatible = roles[(roles.index(role) + 1) % len(roles)]
    query = nx.Graph()
    query.add_node(
        1,
        element=("*", "*"),
        wildcard_role=role.value,
    )
    completion = nx.Graph()
    completion.add_node(
        2,
        element=("*", "*"),
        wildcard_role=incompatible.value,
    )

    validation = validate_wildcard_mapping_roles(query, completion, {1: 2})
    assert not validation.valid
    assert validation.issues[0].code == FusionIssueCode.WILDCARD_ROLE_CONFLICT

    completion.nodes[2]["wildcard_role"] = role.value
    assert validate_wildcard_mapping_roles(query, completion, {1: 2}).valid


@pytest.mark.parametrize(
    "mode,scope,termination",
    (
        (
            "fast_track",
            SearchScope.FAST_PATHS_ONLY,
            TerminationPolicy.FIRST_VALID,
        ),
        ("early_stop", SearchScope.FUSION, TerminationPolicy.FIRST_VALID),
        ("full", SearchScope.FUSION, TerminationPolicy.EXHAUSTIVE),
    ),
)
def test_legacy_modes_are_explicit_search_policy_presets(
    mode: str,
    scope: SearchScope,
    termination: TerminationPolicy,
) -> None:
    engine = RBLEngine(mode=mode)

    assert engine.search_policy == RBLSearchPolicy(scope, termination)
    assert engine.result["search_policy"] == {
        "scope": scope.value,
        "termination": termination.value,
    }


def test_explicit_search_policy_is_accepted_without_a_mode() -> None:
    policy = RBLSearchPolicy(
        SearchScope.FUSION,
        TerminationPolicy.FIRST_VALID,
    )
    engine = RBLEngine(search_policy=policy)

    assert engine.search_policy is policy
    with pytest.raises(ValueError, match="either mode or search_policy"):
        RBLEngine(mode="full", search_policy=policy)


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
@pytest.mark.parametrize("mode", ("fast_track", "early_stop", "full"))
@pytest.mark.parametrize("replace_wc", (True, False))
def test_every_mode_uses_the_same_fusion_acceptance_contract(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
    mode: str,
    replace_wc: bool,
) -> None:
    engine = RBLEngine(mode=mode).process(
        reaction,
        template,
        replace_wc=replace_wc,
    )

    assert all(
        validate_rbl_candidate(
            reaction,
            candidate,
            allow_wildcards=not replace_wc,
        ).valid
        for candidate in engine.fused_rsmis
    )


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_full_mode_candidate_order_is_deterministic(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    first = RBLEngine(mode="full").process(reaction, template).fused_rsmis
    second = RBLEngine(mode="full").process(reaction, template).fused_rsmis

    assert first == second


@pytest.mark.parametrize("_name,reaction,template,expected", CASES)
def test_template_fragment_permutation_preserves_compatibility_result(
    _name: str,
    reaction: str,
    template: str,
    expected: str,
) -> None:
    reactants, products = template.split(">>", 1)
    permuted = (
        ".".join(reversed(reactants.split(".")))
        + ">>"
        + ".".join(reversed(products.split(".")))
    )

    results = RBLEngine(mode="full").process(reaction, permuted).fused_rsmis
    assert any(AAMValidator.smiles_check(candidate, expected) for candidate in results)


@pytest.mark.parametrize("_name,reaction,template,_expected", CASES)
def test_explicit_h_fusion_serialization_round_trip_preserves_aam(
    _name: str,
    reaction: str,
    template: str,
    _expected: str,
) -> None:
    results = RBLEngine(mode="full").process(reaction, template).fused_rsmis

    for candidate in results:
        replayed = its_to_rsmi(
            rsmi_to_its(candidate, format="tuple"),
            format="tuple",
            explicit_hydrogen=True,
        )
        assert AAMValidator.smiles_check(candidate, replayed)


def test_explicit_and_implicit_input_h_presentations_are_equivalent() -> None:
    _name, _reaction, template, expected = CASES[0]
    explicit_reaction = "CCC(=O)[OH]>>CCC(=O)OC"

    results = (
        RBLEngine(mode="full")
        .process(
            explicit_reaction,
            template,
        )
        .fused_rsmis
    )
    assert any(AAMValidator.smiles_check(candidate, expected) for candidate in results)


def test_fast_track_rejects_an_invalid_quick_candidate() -> None:
    class InvalidQuickEngine(RBLEngine):
        def _quick_check(self, rsmi: str, template: object) -> str:
            # Chemically parseable, but it changes the observed product CO
            # into CN and therefore lacks an endpoint-preservation proof.
            return "CC>>CN"

        def _run_reaction(
            self,
            substrate: object,
            pattern: object,
            invert: bool,
        ) -> list[object]:
            return []

    engine = InvalidQuickEngine(mode="fast_track").process(
        "CC>>CO",
        "[C:1]>>[C:1]",
    )

    assert engine.fused_rsmis == []
    assert engine.result["reason"] == "fast_paths_no_solution"
    assert any(
        issue["code"] == FusionIssueCode.PRODUCT_ENDPOINT_NOT_PRESERVED.value
        for report in engine.diagnostics["fusion"]
        for issue in report["issues"]
    )
