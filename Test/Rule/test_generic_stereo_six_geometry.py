"""Paired exact/local and extracted/typed rule conformance for six geometries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Stereo import (
    AtropBondStereo,
    StereoChange,
    StereoRelationKind,
    TetrahedralStereo,
    classify_stereo_change,
    stereo_from_dict,
    stereo_isomorphic,
)
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import GenericStereoRuleExtractor, SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

DATA_PATH = Path(__file__).parents[2] / "Data" / "Mech" / "stereo.json"
SN2 = "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"
PLANAR = (
    "[F:1]/[C:2]([Cl:3])=[C:4]([Br:5])/[I:6]>>"
    "[F:1]/[C:2]([Cl:3])=[C:4]([Br:5])\\[I:6]"
)


def _annotated_rule_graph(case: dict[str, Any]) -> nx.Graph:
    rc = rsmi_to_its(
        case["mapped_reaction"],
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    before = stereo_from_dict(case["before"])
    after = stereo_from_dict(case["after"])
    target = case["target"]
    rc.graph["stereo_descriptors"] = {
        "reactant": {target: before},
        "product": {target: after},
    }
    rc.graph["stereo_changes"] = {
        target: StereoChange(classify_stereo_change(before, after), before, after)
    }
    return rc


def _atrop_rule_graph(payload: dict[str, Any]) -> nx.Graph:
    case = next(
        case
        for case in payload["cases"]
        if case.get("case_id") == "nt-atrop-biaryl-radical-coupling"
    )
    product = case["mapped_reaction"].split(">>", 1)[1]
    rc = rsmi_to_its(
        f"{product}>>{product}",
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    before = AtropBondStereo((1, 3, 2, 4, 5, 6), 1, "Sprint 22 fixture")
    after = before.invert()
    target = "bond:2-4"
    rc.graph["stereo_descriptors"] = {
        "reactant": {target: before},
        "product": {target: after},
    }
    rc.graph["stereo_changes"] = {
        target: StereoChange(classify_stereo_change(before, after), before, after)
    }
    return rc


def _six_rule_graphs() -> dict[str, nx.Graph]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    result = {
        "tetrahedral": rsmi_to_its(
            SN2,
            format="tuple",
            drop_non_aam=False,
            use_index_as_atom_map=True,
        ),
        "planar_bond": rsmi_to_its(
            PLANAR,
            format="tuple",
            drop_non_aam=False,
            use_index_as_atom_map=True,
        ),
        "atrop_bond": _atrop_rule_graph(payload),
    }
    for family in (
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    ):
        case = next(
            case
            for case in payload["cases"]
            if case.get("representation") == "non_tetrahedral_rewrite"
            and case.get("stereo_family") == family
        )
        result[family] = _annotated_rule_graph(case)
    return result


RULE_GRAPHS = _six_rule_graphs()


def _reactor(host: nx.Graph, rule: SynRule) -> SynReactor:
    return SynReactor(
        host,
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        automorphism=False,
    )


@pytest.mark.parametrize("family", tuple(RULE_GRAPHS))
def test_extracted_and_exact_rules_have_identical_source_populations(family: str):
    rc = RULE_GRAPHS[family]
    exact_rule = SynRule(rc, format="tuple", implicit_h=False)
    extracted = GenericStereoRuleExtractor().extract(rc)
    reactant = ITSReverter(rc).to_reactant_graph()

    exact = _reactor(reactant, exact_rule)
    generic = _reactor(reactant, extracted.rule)

    assert extracted.certificate.source_replay_exact
    assert extracted.certificate.reverse_status == "exact"
    assert exact.mapping_count == generic.mapping_count == 1
    assert len(exact.its_list) == len(generic.its_list) == 1
    exact_product = ITSReverter(exact.its_list[0]).to_product_graph()
    generic_product = ITSReverter(generic.its_list[0]).to_product_graph()
    assert stereo_isomorphic(exact_product, generic_product)

    exact_reverse = _reactor(exact_product, exact_rule.reversed())
    generic_reverse = _reactor(generic_product, extracted.rule.reversed())
    assert exact_reverse.mapping_count == generic_reverse.mapping_count == 1
    assert stereo_isomorphic(
        ITSReverter(exact_reverse.its_list[0]).to_product_graph(),
        ITSReverter(generic_reverse.its_list[0]).to_product_graph(),
    )


def test_paired_corpus_covers_all_six_supported_geometry_classes() -> None:
    assert set(RULE_GRAPHS) == {
        "tetrahedral",
        "planar_bond",
        "atrop_bond",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }


def test_nonbinary_square_planar_relation_changes_downstream_rule_selection() -> None:
    """A retained-only projection loses a downstream coordination match."""
    rc = RULE_GRAPHS["square_planar"]
    extracted = GenericStereoRuleExtractor().extract(rc)
    reactant = ITSReverter(rc).to_reactant_graph()
    product = ITSReverter(
        _reactor(reactant, extracted.rule).its_list[0]
    ).to_product_graph()
    change = next(iter(extracted.rule.stereo_effects.values()))

    assert change.relation.kind is StereoRelationKind.RECONFIGURED

    expected_descriptor = change.after
    assert expected_descriptor is not None
    selector_rc = rc.copy()
    selector_rc.graph["stereo_descriptors"] = {
        "reactant": {"atom:1": expected_descriptor},
        "product": {"atom:1": expected_descriptor},
    }
    selector_rc.graph["stereo_changes"] = {
        "atom:1": StereoChange.from_endpoints(
            expected_descriptor,
            expected_descriptor,
        )
    }
    selector = SynRule(selector_rc, format="tuple", implicit_h=False)

    retained_only = product.copy()
    retained_only.graph["stereo_descriptors"] = {"atom:1": change.before}

    assert _reactor(product, selector).mapping_count == 1
    assert _reactor(retained_only, selector).mapping_count == 0


def test_virtual_h_is_retained_as_owner_scoped_identity_not_a_material_port() -> None:
    extracted = GenericStereoRuleExtractor().extract(RULE_GRAPHS["tetrahedral"])
    guard = next(iter(extracted.rule.stereo_guards.values()))

    assert "@H:2" in guard.atoms
    assert {port.reference for port in extracted.certificate.ports} == {1, 3}


def test_virtual_lp_is_retained_while_material_ligands_become_typed_ports() -> None:
    reaction = "[P:1]([F:2])([Cl:3])[Br:4]>>[P:1]([F:2])([Cl:3])[Br:4]"
    rc = rsmi_to_its(
        reaction,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    before = TetrahedralStereo((1, 2, 3, 4, "@LP:1"), 1)
    after = before.invert()
    rc.graph["stereo_descriptors"] = {
        "reactant": {"atom:1": before},
        "product": {"atom:1": after},
    }
    rc.graph["stereo_changes"] = {"atom:1": StereoChange.from_endpoints(before, after)}

    extracted = GenericStereoRuleExtractor().extract(rc)

    assert extracted.certificate.source_replay_exact
    assert "@LP:1" in next(iter(extracted.rule.stereo_guards.values())).atoms
    assert {port.reference for port in extracted.certificate.ports} == {2, 3, 4}
