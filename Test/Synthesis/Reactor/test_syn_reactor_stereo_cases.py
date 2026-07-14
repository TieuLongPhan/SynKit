import pytest
from rdkit import Chem

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Stereo import (
    AtropBondStereo,
    StereoChange,
    StereoOutcome,
    TetrahedralStereo,
)
from synkit.IO.chem_converter import rsmi_to_its
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

SN2_RULE = (
    "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"
)


@pytest.mark.parametrize(
    ("reaction", "expected_outcome"),
    [
        (
            "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>"
            "[CH3:1][C@H:2]([F:3])[OH:5].[Cl-:4]",
            "RETAINED",
        ),
        (SN2_RULE, "INVERTED"),
        (
            "[CH3:1][C@H:2]([F:3])[Cl:4]>>" "[CH3:1][CH+:2][F:3].[Cl-:4]",
            "BROKEN",
        ),
        (
            "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]",
            "FORMED",
        ),
    ],
)
def test_rule_stores_chemical_stereo_outcome(reaction, expected_outcome):
    its = rsmi_to_its(
        reaction,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )

    assert its.graph["stereo_changes"]["atom:2"].change == expected_outcome


def _reactor(substrate, rule, **kwargs):
    return SynReactor(
        substrate,
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="strict",
        **kwargs,
    )


def test_unmapped_radical_substrate_preserves_explicit_product_state():
    rule = "[CH3:1][CH2:2][Cl:3]>>[CH3:1][CH2:2].[Cl:3]"
    reactor = _reactor("CCCl", rule)
    product = ITSReverter(reactor.its_list[0]).to_product_graph()

    assert reactor.mapping_count == 1
    assert sorted(data["radical"] for _, data in product.nodes(data=True)) == [0, 1, 1]
    assert product.number_of_edges() == 1


def test_sn2_stereo_rule_has_complete_rc_and_inverted_effect():
    rc = rsmi_to_its(
        SN2_RULE,
        core=True,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )

    assert set(rc.nodes) == {1, 2, 3, 4, 5}
    assert rc.graph["stereo_changes"]["atom:2"].change == "INVERTED"


def test_sn2_forward_and_inverse_accept_only_expected_enantiomer():
    forward = _reactor("C[C@H](F)Cl.[OH-]", SN2_RULE)
    wrong_forward = _reactor("C[C@@H](F)Cl.[OH-]", SN2_RULE)
    inverse = _reactor("C[C@@H](F)O.[Cl-]", SN2_RULE, invert=True)
    wrong_inverse = _reactor("C[C@H](F)O.[Cl-]", SN2_RULE, invert=True)

    assert set(forward.rule.stereo_guards) == {"atom:2"}
    assert forward.rule.stereo_effects["atom:2"].change == "INVERTED"
    assert forward.mapping_count == inverse.mapping_count == 1
    assert wrong_forward.mapping_count == wrong_inverse.mapping_count == 0
    assert forward.its_list[0].graph["stereo_changes"]["atom:2"].change == ("INVERTED")
    assert "@@" in forward.smarts[0].split(">>", 1)[1]
    assert "@" in inverse.smarts[0].split(">>", 1)[0]


def test_sn1_breaks_then_nonselective_capture_forms_both_enantiomers():
    ionization = "[CH3:1][C@H:2]([F:3])[Cl:4]>>" "[CH3:1][CH+:2][F:3].[Cl-:4]"
    capture_seed = "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"
    capture_rule = SynRule.from_smart(
        capture_seed,
        format="tuple",
        stereo_outcomes={"atom:2": StereoOutcome.racemic()},
    )
    broken = _reactor("C[C@H](F)Cl", ionization)
    formed = _reactor("C[CH+]F.[OH-]", capture_rule)
    broken_product = ITSReverter(broken.its_list[0]).to_product_graph()
    formed_products = [ITSReverter(its).to_product_graph() for its in formed.its_list]

    cip_codes = set()
    constitution_only = set()
    for reaction in formed.smarts:
        smiles = reaction.split(">>", 1)[1]
        molecule = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        constitution_only.add(Chem.MolToSmiles(molecule, isomericSmiles=False))
        cip_codes.update(
            atom.GetProp("_CIPCode")
            for atom in molecule.GetAtoms()
            if atom.HasProp("_CIPCode")
        )

    assert broken.rule.stereo_effects["atom:2"].change == "BROKEN"
    assert capture_rule.stereo_guards == {}
    assert capture_rule.stereo_effects["atom:2"].change == "FORMED"
    assert capture_rule.stereo_outcomes == {"atom:2": StereoOutcome.racemic()}
    assert capture_rule.rc.raw.graph["stereo_outcomes"] == {
        "atom:2": {"kind": "RACEMIC", "weights": [0.5, 0.5]}
    }
    assert formed.mapping_count == 1
    assert len(formed.its_list) == len(formed.smarts) == 2
    assert {
        its.graph["stereo_branch"]["atom:2"]["weight"] for its in formed.its_list
    } == {0.5}
    assert all(
        its.graph["stereo_outcomes"]["atom:2"]["kind"] == "RACEMIC"
        for its in formed.its_list
    )
    assert broken_product.graph["stereo_descriptors"] == {}
    assert all(
        len(product.graph["stereo_descriptors"]) == 1 for product in formed_products
    )
    assert cip_codes == {"R", "S"}
    assert len(constitution_only) == 1


def test_ez_descriptor_break_and_formation_and_strict_guard():
    hydrogenation = (
        "[CH3:1]/[CH:2]=[CH:3]/[CH3:4].[H:5][H:6]>>"
        "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]"
    )
    elimination = (
        "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]>>"
        "[CH3:1]/[CH:2]=[CH:3]/[CH3:4].[H:5][H:6]"
    )
    broken = _reactor("C/C=C/C.[H][H]", hydrogenation)
    wrong_isomer = _reactor("C/C=C\\C.[H][H]", hydrogenation)
    formed = _reactor("CCCC", elimination)
    broken_product = ITSReverter(broken.its_list[0]).to_product_graph()
    formed_product = ITSReverter(formed.its_list[0]).to_product_graph()
    formed_molecule = GraphToMol().graph_to_mol(formed_product)
    formed_double_bond = next(
        bond
        for bond in formed_molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )

    # The two atom mappings of H2 are symmetry-equivalent and deduplicate to
    # one structural hydrogenation product.
    assert broken.mapping_count == 2
    assert len(broken.smarts) == 1
    assert formed.mapping_count == 1
    assert wrong_isomer.mapping_count == 0
    assert broken_product.graph["stereo_descriptors"] == {}
    assert len(formed_product.graph["stereo_descriptors"]) == 1
    assert formed_double_bond.GetStereo() == Chem.BondStereo.STEREOE


def test_stereo_distinct_rules_are_not_equal():
    same = SN2_RULE.replace("C@@H:2", "C@H:2")
    left = SynRule.from_smart(SN2_RULE, format="tuple", implicit_h=False)
    right = SynRule.from_smart(same, format="tuple", implicit_h=False)

    assert left != right
    assert hash(left) != hash(right)


def test_stereo_outcome_participates_in_rule_identity():
    capture = "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"
    single = SynRule.from_smart(capture, format="tuple")
    racemic = SynRule.from_smart(
        capture,
        format="tuple",
        stereo_outcomes={"atom:2": "RACEMIC"},
    )

    assert single != racemic
    assert hash(single) != hash(racemic)


def test_racemic_outcome_rejects_nonformation_effect():
    with pytest.raises(ValueError, match="newly formed chiral"):
        SynRule.from_smart(
            SN2_RULE,
            format="tuple",
            stereo_outcomes={"atom:2": "RACEMIC"},
        )


def test_racemic_is_exactly_equal_and_unequal_is_enantiomeric_mixture():
    with pytest.raises(ValueError, match="equal 0.5/0.5"):
        StereoOutcome("RACEMIC", (0.7, 0.3))
    with pytest.raises(ValueError, match="are RACEMIC"):
        StereoOutcome("ENANTIOMERIC_MIXTURE", (0.5, 0.5))

    mixture = StereoOutcome.enantiomeric_mixture(0.7, 0.3)

    assert mixture.kind == "ENANTIOMERIC_MIXTURE"
    assert mixture.weights == (0.7, 0.3)


def test_enantiomeric_mixture_uses_one_rule_and_preserves_branch_weights():
    capture = "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"
    rule = SynRule.from_smart(
        capture,
        format="tuple",
        stereo_outcomes={"atom:2": StereoOutcome.enantiomeric_mixture(0.7, 0.3)},
    )
    reactor = _reactor("C[CH+]F.[OH-]", rule)

    assert len(reactor.its_list) == 2
    assert sorted(
        its.graph["stereo_branch"]["atom:2"]["weight"] for its in reactor.its_list
    ) == [0.3, 0.7]
    assert sorted(its.graph["stereo_branch_weight"] for its in reactor.its_list) == [
        0.3,
        0.7,
    ]
    assert all(
        its.graph["stereo_outcomes"]["atom:2"]["kind"] == "ENANTIOMERIC_MIXTURE"
        for its in reactor.its_list
    )


def test_atropisomeric_formation_uses_one_racemic_rule_and_two_branches():
    unchanged = (
        "[F:1][CH:2]([Cl:3])[CH:4]([Br:5])[I:6]>>"
        "[F:1][CH:2]([Cl:3])[CH:4]([Br:5])[I:6]"
    )
    rc = rsmi_to_its(
        unchanged,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    formed = AtropBondStereo((1, 3, 2, 4, 5, 6), 1, "rule")
    rc.graph["stereo_descriptors"] = {
        "reactant": {},
        "product": {"bond:2-4": formed},
    }
    rc.graph["stereo_changes"] = {"bond:2-4": StereoChange("FORMED", None, formed)}
    rule = SynRule(
        rc,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"bond:2-4": StereoOutcome.racemic()},
    )
    reactor = _reactor("FC(Cl)C(Br)I", rule)

    assert reactor.mapping_count == 1
    assert len(reactor.its_list) == 2
    products = [
        its.graph["stereo_descriptors"]["product"]["bond:2-4"]
        for its in reactor.its_list
    ]
    assert set(products) == {formed, formed.invert()}
    assert {its.graph["stereo_branch_weight"] for its in reactor.its_list} == {0.5}

    reverse = rule.reversed()
    assert reverse.stereo_query_policies == {"bond:2-4": "either"}
    for orientation in (formed, formed.invert()):
        host = MolToGraph().transform(
            Chem.MolFromSmiles("[F:1][CH:2]([Cl:3])[CH:4]([Br:5])[I:6]")
        )
        host.graph["stereo_descriptors"] = {"bond:2-4": orientation}
        reverse_reactor = _reactor(host, reverse)

        assert reverse_reactor.mapping_count == 1
        assert len(reverse_reactor.its_list) == 1
        assert reverse_reactor.its_list[0].graph["stereo_descriptors"]["product"] == {}


def test_reverse_racemic_rule_accepts_either_enantiomer_without_branching():
    capture = "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"
    forward = SynRule.from_smart(
        capture,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"atom:2": StereoOutcome.racemic()},
    )
    reverse = forward.reversed()

    first = _reactor("C[C@H](F)O", reverse)
    inverse = _reactor("C[C@@H](F)O", reverse)
    serialized_graph_inverse = _reactor(
        "C[C@@H](F)O",
        forward.rc.raw,
        invert=True,
    )

    assert reverse.stereo_effects["atom:2"].change == "BROKEN"
    assert reverse.stereo_outcomes == {}
    assert reverse.stereo_query_policies == {"atom:2": "either"}
    assert first.mapping_count == inverse.mapping_count == 1
    assert serialized_graph_inverse.mapping_count == 1
    assert serialized_graph_inverse.rule.stereo_query_policies == {"atom:2": "either"}
    assert len(first.its_list) == len(inverse.its_list) == 1
    assert reverse.reversed() == forward


def test_stereo_sensitive_host_symmetry_preserves_distinct_applications():
    capture = "[CH3:1][CH+:2][F:3].[OH-:4]>>" "[CH3:1][C@H:2]([F:3])[OH:4]"
    reverse = SynRule.from_smart(
        capture,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"atom:2": StereoOutcome.racemic()},
    ).reversed()
    # Connectivity alone can exchange the two components. Reacting the R or S
    # component leaves the opposite enantiomer untouched, so both applications
    # are stereochemically distinct and must survive host symmetry pruning.
    reactor = _reactor(
        "C[C@H](F)O.C[C@@H](F)O",
        reverse,
        automorphism=True,
    )

    assert reactor.mapping_count == 2
    assert len(reactor.its_list) == len(reactor.smarts) == 2
    assert {
        tuple(product_registry)
        for product_registry in (
            its.graph["stereo_descriptors"]["product"] for its in reactor.its_list
        )
    } == {("atom:2",), ("atom:6",)}


@pytest.mark.parametrize("radical_policy", ["strict", "lower_bound", "ignore"])
def test_radical_policy_never_bypasses_strict_stereo_guard(radical_policy):
    wrong = _reactor(
        "C[C@@H](F)Cl.[OH-]",
        SN2_RULE,
        radical_policy=radical_policy,
    )

    assert wrong.mapping_count == 0


def test_rule_application_separates_unknown_exact_and_wildcard_queries():
    rc = rsmi_to_its(
        SN2_RULE,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    change = rc.graph["stereo_changes"]["atom:2"]
    unknown = TetrahedralStereo(
        change.before.atoms,
        None,
        "unknown-rule-query",
    )
    rc.graph["stereo_descriptors"]["reactant"]["atom:2"] = unknown
    rc.graph["stereo_changes"]["atom:2"] = StereoChange(
        change.change,
        unknown,
        change.after,
    )
    rule = SynRule(rc, format="tuple", implicit_h=False)

    exact = _reactor("C[C@H](F)Cl.[OH-]", rule)
    wildcard = _reactor(
        "C[C@H](F)Cl.[OH-]",
        rule,
        stereo_query_mode="wildcard",
    )

    assert exact.mapping_count == 0
    assert wildcard.mapping_count == 1


def test_fleeting_rule_stereo_is_preserved_on_applied_its():
    unchanged = (
        "[CH3:1][C:2]([F:3])([Cl:4])[OH:5]>>" "[CH3:1][C:2]([F:3])([Cl:4])[OH:5]"
    )
    rc = rsmi_to_its(
        unchanged,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    transition = TetrahedralStereo(
        (2, 1, 3, 4, 5),
        1,
        "its-transition",
    )
    rc.graph["stereo_descriptors"]["transition"] = {"atom:2": transition}
    rc.graph["stereo_changes"] = {
        "atom:2": StereoChange("FLEETING", None, None, transition)
    }
    rule = SynRule(rc, format="tuple", implicit_h=False)
    reactor = _reactor("CC(F)(Cl)O", rule)

    assert reactor.mapping_count == 1
    assert len(reactor.its_list) == 1
    result = reactor.its_list[0]
    transition_graph = ITSReverter(result).to_transition_state_graph()
    assert result.graph["stereo_descriptors"]["transition"] == {"atom:2": transition}
    assert result.graph["stereo_changes"]["atom:2"].change == "FLEETING"
    assert transition_graph.graph["stereo_descriptors"] == {"atom:2": transition}
    assert transition_graph.graph["stereo_projection"] == "transition"
