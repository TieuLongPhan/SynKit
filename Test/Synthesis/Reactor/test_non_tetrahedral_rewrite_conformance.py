"""Compact forward/reverse non-tetrahedral rule conformance fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Stereo import (
    StereoChange,
    StereoOutcome,
    classify_stereo_change,
    descriptor_graph_support_errors,
    descriptors_from_rdkit,
    stereo_from_dict,
    stereo_isomorphic,
)
from synkit.IO.chem_converter import rsmi_to_its
from synkit.IO.graph_to_mol import GraphToMol
from synkit.Mechanism.audit import audit_local_electron_state
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

DATA_PATH = Path(__file__).parents[3] / "Data" / "Mech" / "stereo.json"
PAYLOAD = json.loads(DATA_PATH.read_text())
MANIFEST = PAYLOAD["non_tetrahedral_fixture_metadata"]
CASES = tuple(
    case
    for case in PAYLOAD["cases"]
    if case.get("representation") == "non_tetrahedral_rewrite"
)


def _descriptor(value: dict[str, Any] | None) -> Any:
    return stereo_from_dict(value) if value is not None else None


def _rule_and_endpoints(
    case: dict[str, Any],
) -> tuple[SynRule, Any, Any]:
    rc = rsmi_to_its(
        case["mapped_reaction"],
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    before = _descriptor(case["before"])
    after = _descriptor(case["after"])
    target = case["target"]
    rc.graph["stereo_descriptors"] = {
        "reactant": {target: before} if before is not None else {},
        "product": {target: after} if after is not None else {},
    }
    rc.graph["stereo_changes"] = {
        target: StereoChange(
            classify_stereo_change(before, after),
            before,
            after,
        )
    }
    outcome = case.get("stereo_outcome")
    rule = SynRule(
        rc,
        format="tuple",
        implicit_h=False,
        stereo_outcomes=(
            {target: StereoOutcome.from_value(outcome)} if outcome is not None else None
        ),
    )
    return (
        rule,
        ITSReverter(rc).to_reactant_graph(),
        ITSReverter(rc).to_product_graph(),
    )


def _reactor(host: Any, rule: SynRule) -> SynReactor:
    return SynReactor(
        host,
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_compact_fixture_applies_forward_and_reverse(case):
    rule, reactant, _ = _rule_and_endpoints(case)
    target = case["target"]

    assert audit_local_electron_state(reactant).valid
    assert rule.stereo_effects[target].change == case["expected_change"]

    forward = _reactor(reactant, rule)

    assert forward.mapping_count >= 1
    assert len(forward.its_list) == case["expected_forward_products"]
    forward_products = [ITSReverter(its).to_product_graph() for its in forward.its_list]
    assert all(
        audit_local_electron_state(product).valid for product in forward_products
    )

    expected = _descriptor(case["after"])
    expected_orientations = (
        {expected, expected.invert()}
        if case.get("stereo_outcome") is not None
        else {expected}
    )
    assert {
        product.graph["stereo_descriptors"].get(target) for product in forward_products
    } == expected_orientations

    for product in forward_products:
        rebuilt = GraphToMol().graph_to_mol(product)
        assert descriptors_from_rdkit(rebuilt)[target] == (
            product.graph["stereo_descriptors"][target]
        )

        reverse = _reactor(product, rule.reversed())
        assert reverse.mapping_count >= 1
        assert len(reverse.its_list) == case["expected_reverse_products_per_branch"]
        restored = ITSReverter(reverse.its_list[0]).to_product_graph()
        assert audit_local_electron_state(restored).valid
        assert stereo_isomorphic(restored, reactant)


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CASES if case["before"] is not None),
    ids=lambda case: case["case_id"],
)
def test_wrong_orientation_is_rejected_by_the_exact_guard(case):
    rule, reactant, _ = _rule_and_endpoints(case)
    reactant.graph["stereo_descriptors"] = {case["target"]: _descriptor(case["after"])}

    assert _reactor(reactant, rule).mapping_count == 0


def test_wrong_locus_is_localized_by_descriptor_topology_validation():
    case = next(case for case in CASES if case["stereo_family"] == "octahedral")
    rule, reactant, _ = _rule_and_endpoints(case)
    wrong_locus = _descriptor(case["before"]).relabel({1: 2, 2: 1})
    reactant.graph["stereo_descriptors"] = {"atom:2": wrong_locus}

    errors = descriptor_graph_support_errors(
        reactant,
        wrong_locus,
        registry_key="atom:2",
    )

    assert errors
    assert any("not adjacent to owner 2" in error for error in errors)
    assert _reactor(reactant, rule).mapping_count == 0


def test_lost_descriptor_is_rejected_at_the_declared_target():
    case = next(
        case for case in CASES if case["stereo_family"] == "trigonal_bipyramidal"
    )
    rule, reactant, _ = _rule_and_endpoints(case)
    reactant.graph["stereo_descriptors"] = {}

    assert set(rule.stereo_guards) == {case["target"]}
    assert _reactor(reactant, rule).mapping_count == 0


def test_racemic_atrop_branches_cannot_accidentally_merge():
    case = next(case for case in CASES if case["stereo_family"] == "atrop_bond")
    rule, reactant, _ = _rule_and_endpoints(case)
    forward = _reactor(reactant, rule)

    assert len(forward.its_list) == 2
    assert len(SynReactor._deduplicate_structural_its(forward.its_list)) == 2
    first, second = (ITSReverter(its).to_product_graph() for its in forward.its_list)
    assert not stereo_isomorphic(first, second)
    assert {
        product.graph["stereo_descriptors"][case["target"]]
        for product in (first, second)
    } == {
        _descriptor(case["after"]),
        _descriptor(case["after"]).invert(),
    }


def test_manifest_records_scope_provenance_and_all_negative_controls():
    assert MANIFEST["status"] == "project_owned_compact_conformance"
    assert MANIFEST["provenance"]["license"] == "project-owned"
    assert {case["stereo_family"] for case in CASES} == {
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "atrop_bond",
    }
    assert all(
        case["application_directions"] == ["forward", "reverse"]
        and case["conversion_boundary"]
        and case["claim_boundary"]
        for case in CASES
    )
    assert {control["control_id"] for control in MANIFEST["negative_controls"]} == {
        "wrong-orientation",
        "wrong-locus",
        "lost-descriptor",
        "accidental-merge",
    }
