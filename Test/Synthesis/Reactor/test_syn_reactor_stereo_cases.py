import pytest
from rdkit import Chem

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Stereo import (
    StereoOutcome,
    StereoRelationKind,
)
from synkit.IO.chem_converter import rsmi_to_its
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

SN2_RULE = (
    "[CH3:1][C@H:2]([F:3])[Cl:4].[OH-:5]>>" "[CH3:1][C@@H:2]([F:3])[OH:5].[Cl-:4]"
)

GENERIC_SYN_PI_HYDROGENATION_RULE = (
    "[C:1]=[C:2].[H:3][H:4]>>" "[C:1]([H:3])[C:2]([H:4])"
)

GENERIC_ANTI_PI_BROMINATION_RULE = (
    "[C:1]=[C:2].[Br:3][Br:4]>>" "[C:1]([Br:3])[C:2]([Br:4])"
)

GENERIC_SN2_RULE = (
    "[*:1][C@H:2]([*:3])[Cl:4].[OH-:5]>>" "[*:1][C@@H:2]([*:3])[OH:5].[Cl-:4]"
)

GENERIC_SN1_CAPTURE_RULE = "[*:1][CH+:2][*:3].[OH-:4]>>" "[*:1][C@H:2]([*:3])[OH:4]"


def _vicinal_addition(relation):
    return {"bond:1-2": relation}


def _cip_codes_by_atom_map(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    atom_maps = {atom.GetIdx(): atom.GetAtomMapNum() for atom in molecule.GetAtoms()}
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return {
        atom_maps[atom.GetIdx()]: atom.GetProp("_CIPCode")
        for atom in molecule.GetAtoms()
        if atom.HasProp("_CIPCode")
    }


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


def _typed_generic_sn2_rule(
    *,
    first_elements=("C",),
    first_owner=2,
    first_slot=1,
    first_side="any",
):
    rc = rsmi_to_its(
        GENERIC_SN2_RULE,
        format="tuple",
        drop_non_aam=False,
        use_index_as_atom_map=True,
    )
    rc.nodes[1].update(
        wildcard_role="stereo_ligand_port",
        owner=first_owner,
        stereo_slot=first_slot,
        elements=set(first_elements),
        side=first_side,
    )
    rc.nodes[3].update(
        wildcard_role="stereo_ligand_port",
        owner=2,
        stereo_slot=0,
        elements={"F"},
    )
    return SynRule(rc, format="tuple", implicit_h=False)


def test_reactor_enforces_typed_stereo_ligand_domains() -> None:
    accepted = _reactor(
        "C[C@H](F)Cl.[OH-]",
        _typed_generic_sn2_rule(),
    )
    rejected = _reactor(
        "C[C@H](F)Cl.[OH-]",
        _typed_generic_sn2_rule(first_elements=("Xe",)),
    )

    assert accepted.mapping_count == 1
    assert len(accepted.its_list) == 1
    assert rejected.mapping_count == 0
    assert rejected.its_list == []
    assert {issue.code.value for issue in rejected.stereo_morphism_issues} == {
        "STEREO_MORPHISM_PORT_DOMAIN_MISMATCH"
    }


def test_reactor_enforces_typed_stereo_port_endpoint_side() -> None:
    reactant = _reactor(
        "C[C@H](F)Cl.[OH-]",
        _typed_generic_sn2_rule(first_side="reactant"),
    )
    product_only = _reactor(
        "C[C@H](F)Cl.[OH-]",
        _typed_generic_sn2_rule(first_side="product"),
    )

    assert reactant.mapping_count == 1
    assert product_only.mapping_count == 0


def test_typed_stereo_ports_realign_on_reverse_and_double_reverse() -> None:
    rule = _typed_generic_sn2_rule()
    forward = _reactor("C[C@H](F)Cl.[OH-]", rule)
    product = forward.smarts[0].split(">>", 1)[1]
    reverse_rule = rule.reversed()
    reverse = _reactor(product, reverse_rule)

    assert reverse.mapping_count == 1
    assert reverse_rule.left.raw.nodes[1]["stereo_slot"] == 0
    assert reverse_rule.left.raw.nodes[3]["stereo_slot"] == 1
    assert reverse_rule.reversed() == rule


@pytest.mark.parametrize(
    ("owner", "slot"),
    ((3, 1), (2, 99)),
)
def test_reactor_rejects_wrong_typed_stereo_owner_or_slot(owner, slot) -> None:
    reactor = _reactor(
        "C[C@H](F)Cl.[OH-]",
        _typed_generic_sn2_rule(first_owner=owner, first_slot=slot),
    )

    assert reactor.mapping_count == 0
    assert reactor.its_list == []
    assert reactor.stereo_morphism_issues


def test_unmapped_radical_substrate_preserves_explicit_product_state():
    rule = "[CH3:1][CH2:2][Cl:3]>>[CH3:1][CH2:2].[Cl:3]"
    reactor = _reactor("CCCl", rule)
    product = ITSReverter(reactor.its_list[0]).to_product_graph()

    assert reactor.mapping_count == 1
    assert sorted(data["radical"] for _, data in product.nodes(data=True)) == [0, 1, 1]
    assert product.number_of_edges() == 1


def test_propagate_mode_drops_descriptor_after_a_ligand_disconnects():
    rule = "[CH3:1][CH:2]([F:3])[Cl:4]>>" "[CH3:1][CH:2][F:3].[Cl:4]"
    reactor = SynReactor(
        "C[C@H](F)Cl",
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="propagate",
    )

    product_registry = reactor.its_list[0].graph["stereo_descriptors"]["product"]

    assert "atom:2" not in product_registry


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


def test_sn2_exact_guard_mode_accepts_only_template_enantiomer():
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


def test_sn2_default_propagation_inverts_either_enantiomer():
    rule = SynRule.from_smart(SN2_RULE, format="tuple", implicit_h=False)
    first = SynReactor(
        "C[C@H](F)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    mirror = SynReactor(
        "C[C@@H](F)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    first_product = first.its_list[0].graph["stereo_descriptors"]["product"]["atom:2"]
    mirror_product = mirror.its_list[0].graph["stereo_descriptors"]["product"]["atom:2"]

    assert first.stereo_mode == mirror.stereo_mode == "propagate"
    assert first.mapping_count == mirror.mapping_count == 1
    assert first_product.invert() == mirror_product
    assert "@@" in first.smarts[0].split(">>", 1)[1]
    assert "@H" in mirror.smarts[0].split(">>", 1)[1]


def test_reactor_compare_mode_audits_application_without_changing_products():
    rule = SynRule.from_smart(SN2_RULE, format="tuple", implicit_h=False)
    orbit = SynReactor(
        "C[C@@H](F)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    compared = SynReactor(
        "C[C@@H](F)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
        stereo_semantics="compare",
    )

    assert compared.smarts == orbit.smarts
    assert [record.stage for record in compared.stereo_semantic_diagnostics] == [
        "reaction_stereo_application"
    ]
    assert all(record.registered for record in compared.stereo_semantic_diagnostics)


def test_reactor_compare_mode_audits_mapping_population_and_reverse_rule():
    forward = _reactor(
        "C[C@H](F)Cl.[OH-]",
        SN2_RULE,
        stereo_semantics="compare",
    )
    reverse = _reactor(
        "C[C@@H](F)O.[Cl-]",
        SN2_RULE,
        invert=True,
        stereo_semantics="compare",
    )

    assert forward.mapping_count == reverse.mapping_count == 1
    assert {record.stage for record in forward.stereo_semantic_diagnostics} == {
        "candidate_mapping"
    }
    assert {record.stage for record in reverse.stereo_semantic_diagnostics} == {
        "candidate_mapping",
        "reaction_stereo_reverse",
    }
    assert all(
        record.registered
        for reactor in (forward, reverse)
        for record in reactor.stereo_semantic_diagnostics
    )


def test_reactor_rejects_unknown_stereo_semantics_mode():
    with pytest.raises(ValueError, match="stereo_semantics"):
        SynReactor("CC", "[C:1]>>[C:1]", stereo_semantics="fallback")


def test_sn2_default_propagation_keeps_unspecified_orientation_unknown():
    reactor = SynReactor(
        "CC(F)Cl.[OH-]",
        SN2_RULE,
        template_format="tuple",
        explicit_h=False,
    )
    product = reactor.its_list[0].graph["stereo_descriptors"]["product"]["atom:2"]

    assert reactor.mapping_count == 1
    assert product.parity is None
    assert "@" not in reactor.smarts[0]


def test_generic_sn2_binds_wildcard_stereo_references_to_host_neighbors():
    rule = SynRule.from_smart(
        GENERIC_SN2_RULE,
        format="tuple",
        implicit_h=False,
    )
    methyl_fluoro = SynReactor(
        "C[C@H](F)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    ethyl_bromo = SynReactor(
        "CC[C@@H](Br)Cl.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert methyl_fluoro.mapping_count == ethyl_bromo.mapping_count == 1
    assert "[C@@H:2]" in methyl_fluoro.smarts[0].split(">>", 1)[1]
    assert "[C@H:3]" in ethyl_bromo.smarts[0].split(">>", 1)[1]
    assert {
        attrs["atom_map"]
        for _, attrs in rule.left.raw.nodes(data=True)
        if attrs["element"] == "*"
    } == {1, 3}


def test_generic_sn1_capture_branches_on_different_substituents():
    rule = SynRule.from_smart(
        GENERIC_SN1_CAPTURE_RULE,
        format="tuple",
        implicit_h=False,
        stereo_outcomes={"atom:2": StereoOutcome.racemic()},
    )
    first = SynReactor(
        "C[CH+]F.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    second = SynReactor(
        "CC[CH+]Br.[OH-]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert first.mapping_count == second.mapping_count == 1
    assert len(first.smarts) == len(second.smarts) == 2
    assert {
        tuple(sorted(_cip_codes_by_atom_map(smarts.split(">>", 1)[1]).values()))
        for smarts in second.smarts
    } == {("R",), ("S",)}


def test_sn2_inversion_can_keep_the_same_cip_label():
    """Physical inversion is distinct from an R/S label after priority changes."""
    reactor = _reactor("C[C@H](F)Cl.[OH-]", SN2_RULE)

    reactant_cip = _cip_codes_by_atom_map(SN2_RULE.split(">>", 1)[0])
    product_cip = _cip_codes_by_atom_map(reactor.smarts[0].split(">>", 1)[1])

    assert reactor.rule.stereo_effects["atom:2"].change == "INVERTED"
    assert reactor.rule.stereo_effects["atom:2"].reference_mapping == ((4, 5),)
    assert reactor.rule.stereo_effects["atom:2"].relation.kind is (
        StereoRelationKind.OPPOSITE
    )
    assert reactant_cip[2] == product_cip[2] == "R"


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


def test_default_propagation_consumes_either_ez_when_no_product_stereo_is_coupled():
    hydrogenation = (
        "[CH3:1]/[CH:2]=[CH:3]/[CH3:4].[H:5][H:6]>>"
        "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]"
    )
    e_reactor = SynReactor(
        "C/C=C/C.[H][H]",
        hydrogenation,
        template_format="tuple",
        explicit_h=False,
    )
    z_reactor = SynReactor(
        "C/C=C\\C.[H][H]",
        hydrogenation,
        template_format="tuple",
        explicit_h=False,
    )

    assert e_reactor.mapping_count == z_reactor.mapping_count == 2
    assert len(e_reactor.smarts) == len(z_reactor.smarts) == 1
    assert all(
        its.graph["stereo_descriptors"]["product"] == {}
        for reactor in (e_reactor, z_reactor)
        for its in reactor.its_list
    )


def test_relative_hcount_rule_hydrogenates_fully_substituted_alkene():
    hydrogenation = (
        "[CH3:1]/[CH:2]=[CH:3]/[CH3:4].[H:5][H:6]>>"
        "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]"
    )
    rule = SynRule.from_smart(
        hydrogenation,
        format="tuple",
        implicit_h=True,
    )
    reactor = SynReactor(
        "C/C(CC)=C(CC)/C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    alkene_centers = (2, 3)
    assert all(rule.left.raw.nodes[node]["hcount"] == 0 for node in alkene_centers)
    assert reactor.mapping_count > 0
    assert len(reactor.smarts) == 1
    assert reactor.its_list[0].graph["stereo_descriptors"]["product"] == {}


def test_relative_hcount_coupling_collapses_fully_substituted_meso_faces():
    hydrogenation = (
        "[CH3:1][CH:2]=[CH:3][CH3:4].[H:5][H:6]>>"
        "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]"
    )
    absolute_rule = SynRule.from_smart(
        hydrogenation,
        format="tuple",
        implicit_h=False,
        stereo_couplings={"bond:2-3": "SYN"},
    )
    relative_rule = SynRule.from_smart(
        hydrogenation,
        format="tuple",
        implicit_h=True,
        stereo_couplings={"bond:2-3": "SYN"},
    )
    substrate = "C/C(CC)=C(CC)\\C.[H][H]"
    absolute = SynReactor(
        substrate,
        absolute_rule,
        template_format="tuple",
        explicit_h=False,
    )
    relative = SynReactor(
        substrate,
        relative_rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert absolute.mapping_count == 0
    assert relative.mapping_count == 1
    assert len(relative.smarts) == 1
    unmapped_products = set()
    for smarts in relative.smarts:
        molecule = Chem.MolFromSmiles(smarts.split(">>", 1)[1])
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        unmapped_products.add(Chem.MolToSmiles(molecule, isomericSmiles=True))
    assert len(unmapped_products) == 1
    assert set(
        _cip_codes_by_atom_map(relative.smarts[0].split(">>", 1)[1]).values()
    ) == {"R", "S"}
    symmetry = relative.its_list[0].graph["stereo_coupling_branch"]["bond:2-5"]
    assert symmetry["equivalent_face_branches"] == [0, 1]
    assert symmetry["symmetry_multiplicity"] == 2
    assert all(
        len(its.graph["stereo_descriptors"]["product"]) == 2
        and {change.change for change in its.graph["stereo_changes"].values()}
        == {"BROKEN", "FORMED"}
        and its.graph["stereo_couplings"]
        for its in relative.its_list
    )


def test_coupling_face_dedup_keeps_the_enantiomer_pair_without_smiles_authority(
    monkeypatch,
):
    rule = SynRule.from_smart(
        GENERIC_SYN_PI_HYDROGENATION_RULE,
        format="tuple",
        implicit_h=True,
        stereo_couplings={"bond:1-2": "SYN"},
    )
    reactor = SynReactor(
        "C/C(CC)=C(CC)/C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert reactor.mapping_count == 1
    assert len(reactor.smarts) == 2
    assert {
        tuple(
            sorted(
                _cip_codes_by_atom_map(smarts.split(">>", 1)[1])[atom_map]
                for atom_map in (2, 5)
            )
        )
        for smarts in reactor.smarts
    } == {("R", "R"), ("S", "S")}

    def fail_if_serialized(*args, **kwargs):
        raise AssertionError("canonical SMILES is not a deduplication key")

    candidates = list(reactor.its_list)
    monkeypatch.setattr(Chem, "MolToSmiles", fail_if_serialized)
    deduplicated = SynReactor._deduplicate_coupling_face_products(candidates)

    assert deduplicated == candidates


def test_potential_stereo_reconstruction_failure_has_explicit_sentinel(
    monkeypatch,
):
    product = MolToGraph().transform(Chem.MolFromSmiles("[CH3:1][CH3:2]"))

    def fail_reconstruction(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(GraphToMol, "graph_to_mol", fail_reconstruction)

    assert SynReactor._potential_tetrahedral_atom_maps(product) is None


def test_potential_stereo_failure_retains_rule_derived_candidates(monkeypatch):
    rule = SynRule.from_smart(
        GENERIC_SYN_PI_HYDROGENATION_RULE,
        format="tuple",
        implicit_h=True,
        stereo_couplings={"bond:1-2": "SYN"},
    )
    baseline = SynReactor(
        "C/C=C/C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert len(baseline.its_list) == 1
    assert baseline.its_list[0].graph["stereo_descriptors"]["product"] == {}

    monkeypatch.setattr(
        SynReactor,
        "_potential_tetrahedral_atom_maps",
        staticmethod(lambda product: None),
    )
    conservative = SynReactor(
        "C/C=C/C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert len(conservative.its_list) == 2
    assert all(
        len(its.graph["stereo_descriptors"]["product"]) == 2
        for its in conservative.its_list
    )


@pytest.mark.parametrize(
    ("relation", "expected_geometry"),
    [
        ("SYN", Chem.BondStereo.STEREOZ),
        ("ANTI", Chem.BondStereo.STEREOE),
    ],
)
def test_generic_pi_rule_semihydrogenates_but_2_yne_with_syn_anti_coupling(
    relation,
    expected_geometry,
):
    """The same alkyne graph edit produces Z/E according to one coupling."""
    rule = SynRule.from_smart(
        GENERIC_SYN_PI_HYDROGENATION_RULE,
        format="tuple",
        implicit_h=False,
        stereo_couplings=_vicinal_addition(relation),
    )
    reactor = SynReactor(
        "CC#CC.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert reactor.mapping_count == 1
    assert len(reactor.smarts) == 1
    product_molecule = Chem.MolFromSmiles(reactor.smarts[0].split(">>", 1)[1])
    product_double_bond = next(
        bond
        for bond in product_molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    assert product_double_bond.GetStereo() == expected_geometry
    assert rule.stereo_effects == {}
    assert all(
        its.graph["stereo_changes"]["bond:2-3"].change == "FORMED"
        for its in reactor.its_list
    )
    assert {
        value["relation"]
        for its in reactor.its_list
        for value in its.graph["stereo_couplings"].values()
    } == {relation}


def test_one_generic_syn_pi_rule_reduces_alkyne_and_alkene():
    """C=C is a relative one-pi-bond locus for a coupled addition rule."""
    rule = SynRule.from_smart(
        GENERIC_SYN_PI_HYDROGENATION_RULE,
        format="tuple",
        implicit_h=True,
        stereo_couplings={"bond:1-2": "SYN"},
    )

    alkyne = SynReactor(
        "CC#CC.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    alkene = SynReactor(
        "C/C(CC)=C(CC)\\C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    alkyne_product = Chem.MolFromSmiles(alkyne.smarts[0].split(">>", 1)[1])
    product_double_bond = next(
        bond
        for bond in alkyne_product.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    assert alkyne.mapping_count == alkene.mapping_count == 1
    assert len(alkyne.smarts) == 1
    assert len(alkene.smarts) == 1
    assert product_double_bond.GetStereo() == Chem.BondStereo.STEREOZ
    assert all(
        set(its.graph["stereo_descriptors"]["product"]) == {"atom:2", "atom:5"}
        and all(
            descriptor.descriptor_class == "tetrahedral"
            for descriptor in its.graph["stereo_descriptors"]["product"].values()
        )
        for its in alkene.its_list
    )

    simple_e = SynReactor(
        "C/C=C/C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    simple_z = SynReactor(
        "C/C=C\\C.[H][H]",
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    assert len(simple_e.smarts) == len(simple_z.smarts) == 1
    assert all(
        its.graph["stereo_descriptors"]["product"] == {}
        and set(its.graph["stereo_changes"]) == {"bond:2-3"}
        for reactor in (simple_e, simple_z)
        for its in reactor.its_list
    )
    assert rule.stereo_summary() == {
        "bond:1-2": {
            "coupling": {
                "kind": "VICINAL_ADDITION",
                "relation": "SYN",
                "centers": [1, 2],
                "ligands": [3, 4],
            }
        }
    }
    assert all(
        "_minimum_pi_order" not in attrs
        for _, _, attrs in rule.left.raw.edges(data=True)
    )
    assert all(
        "_coupled_pi_center_query" not in attrs
        for _, attrs in rule.left.raw.nodes(data=True)
    )


def test_relative_pi_matching_is_scoped_to_declared_coupling():
    literal_rule = SynRule.from_smart(
        GENERIC_SYN_PI_HYDROGENATION_RULE,
        format="tuple",
        implicit_h=True,
    )
    reactor = SynReactor(
        "CC#CC.[H][H]",
        literal_rule,
        template_format="tuple",
        explicit_h=False,
        radical_policy="ignore",
    )

    assert reactor.mapping_count == 0


def test_bromination_rule_states_anti_relation_instead_of_implying_it():
    rule = SynRule.from_smart(
        GENERIC_ANTI_PI_BROMINATION_RULE,
        format="tuple",
        implicit_h=False,
        stereo_couplings=_vicinal_addition("ANTI"),
    )
    reactor = SynReactor(
        "C/C=C/C.BrBr",
        rule,
        template_format="tuple",
        explicit_h=False,
    )

    assert rule.stereo_summary()["bond:1-2"]["coupling"]["relation"] == "ANTI"
    assert rule.stereo_effects == {}
    assert reactor.mapping_count > 0
    assert all(
        its.graph["stereo_couplings"]["bond:2-3"]["relation"] == "ANTI"
        for its in reactor.its_list
    )
