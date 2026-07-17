"""Executable review gates for promoted native stereo benchmark rewrites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem

from synkit.Chem.Reaction import audit_explicit_h_reaction
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Stereo import (
    StereoChange,
    StereoOutcome,
    classify_stereo_change,
    descriptor_graph_support_errors,
    stereo_from_dict,
    stereo_isomorphic,
)
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Mechanism.audit import audit_local_electron_state
from synkit.Rule import NonInvertibleStereoEffectError, SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ROOT = Path(__file__).parents[3]
PAYLOAD = json.loads((ROOT / "Data/Mech/stereo.json").read_text(encoding="utf-8"))
CASES = tuple(
    case
    for case in PAYLOAD["cases"]
    if case.get("representation") == "native_stereo_rewrite"
)
SMILES_PARAMS = Chem.SmilesParserParams()
SMILES_PARAMS.removeHs = False
ALLOWED_H_AUDIT_ERRORS = {
    "NO_EXPLICIT_MAPPED_HYDROGEN",
    "NO_CHANGED_EXPLICIT_HYDROGEN",
}


def _descriptor(value: dict[str, Any] | None) -> Any:
    return stereo_from_dict(value) if value is not None else None


def _registry(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {key: _descriptor(value) for key, value in values.items()}


def _target_key(target: list[Any]) -> str:
    if target[0] == "atom":
        return f"atom:{target[1]}"
    left, right = sorted(target[1:])
    return f"bond:{left}-{right}"


def _rule_and_endpoints(step: dict[str, Any]) -> tuple[SynRule, Any, Any]:
    its = rsmi_to_its(
        step["mapped_reaction"],
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    endpoints = step["endpoint_stereo"]
    before = _registry(endpoints["reactant"])
    after = _registry(endpoints["product"])
    effects = {}
    for serialized in step["stereo_effects"]:
        old = _descriptor(serialized["before"])
        new = _descriptor(serialized["after"])
        key = _target_key(serialized["descriptor_target"])
        effects[key] = StereoChange(
            classify_stereo_change(old, new),
            old,
            new,
        )
    its.graph["stereo_descriptors"] = {
        "reactant": before,
        "product": after,
    }
    its.graph["stereo_changes"] = effects
    outcomes = {
        key: StereoOutcome.from_value(value)
        for key, value in step.get("rule_metadata", {})
        .get("stereo_outcomes", {})
        .items()
    }
    rule = SynRule(
        its,
        format="tuple",
        implicit_h=False,
        stereo_outcomes=outcomes or None,
    )
    reactant = ITSReverter(its).to_reactant_graph()
    product = ITSReverter(its).to_product_graph()
    reactant.graph["stereo_descriptors"] = before
    product.graph["stereo_descriptors"] = after
    return rule, reactant, product


def _reactor(
    host: Any,
    rule: SynRule,
    *,
    reactor_options: dict[str, Any] | None = None,
) -> SynReactor:
    options = reactor_options or {}
    return SynReactor(
        host,
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        **options,
    )


def test_promotion_manifest_has_the_reviewed_80_positive_boundary():
    legacy_metadata = PAYLOAD["candidate_promotion_metadata"]
    final_metadata = PAYLOAD["final_eight_promotion_metadata"]
    legacy_cases = [case for case in CASES if case["case_id"] <= "ST-69"]
    final_cases = [case for case in CASES if case["case_id"] >= "ST-70"]

    assert len(CASES) == 29
    assert len(legacy_cases) == legacy_metadata["accepted_case_count"] == 21
    assert [case["case_id"] for case in legacy_cases] == [
        f"ST-{number}" for number in range(49, 70)
    ]
    assert len(final_cases) == final_metadata["accepted_case_count"] == 8
    assert [case["case_id"] for case in final_cases] == [
        f"ST-{number}" for number in range(70, 78)
    ]
    assert all(
        case["status"] == "executable"
        and case["provenance"]["candidate_review"]["decision"] == "accepted_executable"
        for case in CASES
    )
    assert legacy_metadata["not_promoted_case_count"] == 45
    assert sum(legacy_metadata["decision_summary"].values()) == 45
    assert final_metadata["engine_blocked_source_ids"] == []
    assert final_metadata["resolved_engine_blocker_source_ids"] == [
        "ST-FINAL-CAND-006",
        "ST-FINAL-CAND-007",
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_promoted_case_parses_balances_and_deserializes(case):
    assert case["steps"]
    for step in case["steps"]:
        left, right = step["mapped_reaction"].split(">>", 1)
        assert Chem.MolFromSmiles(left, SMILES_PARAMS) is not None
        assert Chem.MolFromSmiles(right, SMILES_PARAMS) is not None
        report = audit_explicit_h_reaction(step["mapped_reaction"])
        assert not set(report.errors) - ALLOWED_H_AUDIT_ERRORS

        rule, reactant, product = _rule_and_endpoints(step)
        assert audit_local_electron_state(reactant).valid
        assert audit_local_electron_state(product).valid
        assert {
            key: change.change for key, change in rule.stereo_effects.items()
        } == step["expected_stereo_changes"]

        for graph, endpoint in (
            (reactant, step["endpoint_stereo"]["reactant"]),
            (product, step["endpoint_stereo"]["product"]),
        ):
            for key, serialized in endpoint.items():
                descriptor = _descriptor(serialized)
                assert stereo_from_dict(descriptor.to_dict()) == descriptor
                assert not descriptor_graph_support_errors(
                    graph,
                    descriptor,
                    registry_key=key,
                )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_promoted_case_applies_forward_reverse_and_guards_orientation(case):
    for step in case["steps"]:
        rule, reactant, expected_product = _rule_and_endpoints(step)
        reactor_options = step["application"].get("reactor_options", {})
        forward = _reactor(reactant, rule, reactor_options=reactor_options)

        assert forward.mapping_count >= 1
        assert (
            len(forward.its_list)
            == step["application"]["expected_unique_stereoisomer_count"]
        )
        expected_registry = expected_product.graph["stereo_descriptors"]
        for result in forward.its_list:
            product = ITSReverter(result).to_product_graph()
            assert audit_local_electron_state(product).valid
            actual_registry = product.graph.get("stereo_descriptors", {})
            assert set(actual_registry) == set(expected_registry)
            for key, expected in expected_registry.items():
                assert actual_registry[key] in {expected, expected.invert()}

            expected_reverse_count = step["application"].get(
                "expected_reverse_unique_product_count", 1
            )
            if expected_reverse_count == 0:
                assert rule.is_stereo_reversible is False
                expected_targets = tuple(
                    sorted(
                        key
                        for key, value in step["expected_stereo_changes"].items()
                        if value == "UNSPECIFIED"
                    )
                )
                assert rule.non_invertible_stereo_targets() == expected_targets
                with pytest.raises(NonInvertibleStereoEffectError) as excinfo:
                    rule.reversed()
                assert (
                    excinfo.value.reason
                    == step["application"]["reverse_expected_failure"]
                )
                assert excinfo.value.targets == expected_targets
            else:
                assert rule.is_stereo_reversible is True
                reverse = _reactor(
                    product,
                    rule.reversed(),
                    reactor_options=reactor_options,
                )
                assert reverse.mapping_count >= 1
                assert len(reverse.its_list) == expected_reverse_count
                assert all(
                    stereo_isomorphic(
                        ITSReverter(reverse_result).to_product_graph(),
                        reactant,
                    )
                    for reverse_result in reverse.its_list
                )

        original_registry = reactant.graph.get("stereo_descriptors", {})
        if original_registry:
            wrong = reactant.copy()
            wrong.graph["stereo_descriptors"] = {
                key: descriptor.invert()
                for key, descriptor in original_registry.items()
            }
            assert (
                _reactor(
                    wrong,
                    rule,
                    reactor_options=reactor_options,
                ).mapping_count
                == 0
            )


def test_promoted_mixture_cases_keep_two_weighted_inverse_branches():
    mixture_cases = [
        case
        for case in CASES
        if case["oracle"]["product_population"] == "enantiomeric_mixture"
    ]

    assert len(mixture_cases) == 4
    for case in mixture_cases:
        step = case["steps"][0]
        outcomes = step["rule_metadata"]["stereo_outcomes"]
        assert len(outcomes) == 1
        outcome = StereoOutcome.from_value(next(iter(outcomes.values())))
        assert outcome.kind == "ENANTIOMERIC_MIXTURE"
        assert len(outcome.weights) == 2
        assert outcome.weights[0] != outcome.weights[1]
        assert sum(outcome.weights) == pytest.approx(1.0)

        rule, reactant, _ = _rule_and_endpoints(step)
        products = [
            ITSReverter(result).to_product_graph()
            for result in _reactor(reactant, rule).its_list
        ]
        assert len(products) == 2
        assert not stereo_isomorphic(products[0], products[1])
