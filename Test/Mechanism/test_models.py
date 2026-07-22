import networkx as nx
import pytest
from rdkit import Chem
import json
from pathlib import Path

from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismModelError,
    MechanismRecord,
    MechanisticStep,
    PI,
    RADICAL,
    SIGMA,
    RadicalStateAudit,
    COMPATIBLE_LEWIS_GRAPH_ACRONYMS,
    PREVIOUS_PUBLIC_LEWIS_GRAPH_ACRONYM,
    PUBLIC_LEWIS_GRAPH_ACRONYM,
    audit_local_electron_state,
    complete_radical_aam,
    external_locus_symbol,
    group_from_legacy_epd,
    mechanism_from_legacy_epd,
    mechanism_record_schema,
    radical_match,
    StereoDescriptor,
    StereoEffect,
    normalize_locus_symbol,
    normalize_radical_row,
)
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Vis.epd.models import transition_from_epd_step


def test_record_round_trip_preserves_public_contract():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(2, 1)),
        2,
        "curved",
        "g1",
    )
    record = MechanismRecord(
        "[OH-:1].[CH3+:2]>>[CH3:2][OH:1]",
        (MechanisticStep("s1", (ElectronMoveGroup("g1", (move,)),)),),
    )

    restored = MechanismRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.steps[0].groups[0].moves[0].target.atom_maps == (1, 2)


def test_legacy_epd_adapter_preserves_action_order_and_types():
    record = mechanism_from_legacy_epd(
        "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]",
        [
            ["LP-/Sigma+", [1], [1, 2]],
            ["Sigma-/LP+", [2, 3], [3]],
        ],
    )

    assert [step.step_id for step in record.steps] == ["s1", "s2"]
    assert record.steps[0].groups[0].moves[0].source.kind == "lp"
    assert record.steps[1].groups[0].moves[0].target.kind == "lp"


def test_arrow_type_and_electron_count_must_agree():
    with pytest.raises(MechanismModelError, match="ARROW_ELECTRON_COUNT_MISMATCH"):
        ElectronMove(
            ElectronLocus.atom("rad", atom_map=1),
            ElectronLocus.atom("rad", atom_map=2),
            1,
            "curved",
            "g1",
        )


def test_electron_moves_reject_noops_and_two_electron_radical_loci():
    radical = ElectronLocus.atom("rad", atom_map=1)
    sigma = ElectronLocus.bond("sigma", atom_maps=(1, 2))

    with pytest.raises(MechanismModelError, match="source and target are identical"):
        ElectronMove(radical, radical, 1, "fishhook", "g1")
    with pytest.raises(MechanismModelError, match="RADICAL_LOCUS_REQUIRES_FISHHOOK"):
        ElectronMove(sigma, radical, 2, "curved", "g1")


def test_homolysis_requires_two_coupled_fishhooks_and_accepts_valid_group():
    source = ElectronLocus.bond("sigma", atom_maps=(1, 2))
    one = ElectronMove(
        source,
        ElectronLocus.atom("rad", atom_map=1),
        1,
        "fishhook",
        "g1",
        coupling_id="c1",
    )
    assert (
        ElectronMoveGroup("g1", (one,), macro="HOMOLYSIS").issues()[0].code
        == "MISSING_COUPLED_FISHHOOK"
    )

    two = ElectronMove(
        source,
        ElectronLocus.atom("rad", atom_map=2),
        1,
        "fishhook",
        "g1",
        coupling_id="c1",
    )
    group = ElectronMoveGroup("g1", (one, two), macro="HOMOLYSIS")
    assert group.issues() == ()


def test_homolysis_requires_radicals_on_the_broken_bond_endpoints():
    source = ElectronLocus.bond("sigma", atom_maps=(1, 2))
    moves = tuple(
        ElectronMove(
            source,
            ElectronLocus.atom("rad", atom_map=atom_map),
            1,
            "fishhook",
            "g1",
            coupling_id="c1",
        )
        for atom_map in (3, 4)
    )

    issues = ElectronMoveGroup("g1", moves, macro="HOMOLYSIS").issues()

    assert {issue.code for issue in issues} == {
        "NONLOCAL_ELECTRON_MOVE",
        "UNBALANCED_EVENT_GROUP",
    }


def test_group_rejects_simultaneous_overconsumption_before_replay():
    graph = nx.Graph()
    graph.add_node(1, atom_map=1, radical=1)
    graph.add_node(2, atom_map=2, radical=1)
    graph.add_edge(1, 2, sigma_order=1.0, pi_order=0.0)
    source = ElectronLocus.bond("sigma", atom_maps=(1, 2))
    moves = (
        ElectronMove(source, ElectronLocus.atom("lp", atom_map=1), 2, "curved", "g1"),
        ElectronMove(source, ElectronLocus.atom("lp", atom_map=2), 2, "curved", "g1"),
    )

    assert "LOCUS_OVERCONSUMED" in {
        issue.code for issue in ElectronMoveGroup("g1", moves).validate_pre_state(graph)
    }


def test_group_rejects_nonlocal_primitive_move():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(2, 3)),
        2,
        "curved",
        "g1",
    )

    assert "NONLOCAL_ELECTRON_MOVE" in {
        issue.code for issue in ElectronMoveGroup("g1", (move,)).issues()
    }


def test_lone_pair_relocation_requires_pre_state_adjacency():
    graph = nx.Graph()
    graph.add_node(1, atom_map=1, lone_pairs=1, radical=0)
    graph.add_node(2, atom_map=2, lone_pairs=0, radical=1)
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.atom("lp", atom_map=2),
        1,
        "fishhook",
        "g1",
    )

    issues = ElectronMoveGroup(
        "g1", (move,), macro="LONE_PAIR_RADICAL_RELOCATION"
    ).validate_pre_state(graph)

    assert "NONLOCAL_ELECTRON_MOVE" in {issue.code for issue in issues}


def test_radical_audit_and_policies():
    mol = Chem.MolFromSmiles("[CH3:1]")
    audit = RadicalStateAudit.molecule_round_trip(mol)

    assert audit.matches
    assert radical_match({1: 1}, {1: 2}, policy="lower_bound")
    assert not radical_match({1: 1}, {1: 2}, policy="strict")
    assert radical_match({1: 1}, {1: 2}, policy="ignore")


def test_group_from_legacy_epd_is_a_valid_polar_group():
    group = group_from_legacy_epd([["LP-/Sigma+", [1], [1, 2]]])
    assert group.issues() == ()


def test_v2_stereo_contract_keeps_relative_tag_separate_from_legacy_label():
    mol = Chem.MolFromSmiles("C[C@H](O)F")
    graph = MolToGraph().transform(mol)
    center = graph.nodes[2]

    assert center["chiral_tag"] in {"CW", "CCW"}
    assert center["stereo_descriptor"] is None
    assert "cip_label" in center


@pytest.mark.parametrize(
    ("external", "canonical"),
    [
        ("LP", "lp"),
        ("lp", "lp"),
        ("sigma", "σ"),
        ("σ", "σ"),
        ("pi", "π"),
        ("π", "π"),
        ("rad", "∙"),
        ("∙", "∙"),
        ("·", "∙"),
        ("•", "∙"),
    ],
)
def test_external_locus_aliases_normalize_to_internal_symbols(external, canonical):
    assert normalize_locus_symbol(external) == canonical
    assert (
        ElectronLocus(external, (1,) if canonical in {"lp", "∙"} else (1, 2)).kind
        == canonical
    )


def test_locus_serialization_uses_internal_symbol_and_reads_ascii_alias():
    locus = ElectronLocus.from_dict({"locus": "sigma", "atom_maps": [2, 1]})

    assert locus.kind == SIGMA
    assert locus.to_dict() == {"locus": "σ", "atom_maps": [1, 2]}
    assert external_locus_symbol(PI, style="ascii") == "pi"
    assert external_locus_symbol(RADICAL, style="legacy_epd") == "Rad"


def test_visual_transition_accepts_electron_move_and_displays_internal_kind():
    move = ElectronMove(
        ElectronLocus.atom("LP", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(1, 2)),
        2,
        "curved",
        "g1",
    )

    transition = transition_from_epd_step(move)

    assert transition.kind == "LP-/B+"
    assert transition.data["internal_kind"] == "lp-/σ+"
    assert transition.data["electron_count"] == 2


def test_canonical_schema_exposes_only_internal_locus_symbols():
    locus_enum = mechanism_record_schema()["$defs"]["electronLocus"]["properties"][
        "locus"
    ]["enum"]

    assert locus_enum == ["lp", "σ", "π", "∙"]


def test_generic_legacy_bond_locus_requires_context():
    with pytest.raises(MechanismModelError, match="does not identify σ versus π"):
        group_from_legacy_epd([["B-/LP+", [1, 2], [2]]])


def test_attribute_propagation_contract_is_a_data_artifact():
    path = Path(__file__).parents[2] / "Data/Schema/attribute_propagation_v1_5.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    assert PUBLIC_LEWIS_GRAPH_ACRONYM == "LLG"
    assert PREVIOUS_PUBLIC_LEWIS_GRAPH_ACRONYM == "LSG"
    assert COMPATIBLE_LEWIS_GRAPH_ACRONYMS == ("LLG", "LSG", "LWG")
    assert contract["public_v2_acronym"] == "LLG"
    assert contract["previous_public_acronym"] == "LSG"
    assert "radical" in contract["attributes"]


def test_local_electron_state_audit_reports_and_repairs_charge():
    graph = nx.Graph()
    graph.add_node(
        1,
        atom_map=1,
        valence_electrons=4,
        lone_pairs=0,
        radical=1,
        hcount=3,
        charge=1,
    )

    audit = audit_local_electron_state(graph)
    repaired = audit_local_electron_state(graph, repair=True)

    assert audit.issues[0].code == "LOCAL_ELECTRON_MISMATCH"
    assert repaired.repaired_atom_maps == (1,)
    assert graph.nodes[1]["charge"] == 0


def test_local_electron_state_audit_rejects_negative_and_pure_pi_resources():
    graph = nx.Graph()
    graph.add_node(
        1,
        atom_map=1,
        valence_electrons=4,
        lone_pairs=0,
        radical=0,
        hcount=-1,
        charge=0,
    )
    graph.add_node(
        2,
        atom_map=2,
        valence_electrons=4,
        lone_pairs=0,
        radical=0,
        hcount=2,
        charge=0,
    )
    graph.add_edge(1, 2, sigma_order=0.0, pi_order=1.0)

    codes = {issue.code for issue in audit_local_electron_state(graph).issues}

    assert "INVALID_ELECTRON_RESOURCE" in codes
    assert "PI_WITHOUT_SIGMA" in codes


def test_bond_stereo_effect_target_round_trips_and_is_schema_legal():
    descriptor = StereoDescriptor(
        "planar_bond",
        (1, "@H:2", 2, 3, "@H:3", 4),
        0,
    )
    effect = StereoEffect(("bond", (3, 2)), "FORM", after=descriptor)

    assert effect.descriptor_target == ("bond", (2, 3))
    assert StereoEffect.from_dict(effect.to_dict()) == effect
    target_schema = mechanism_record_schema()["$defs"]["stereoEffect"]["properties"][
        "descriptor_target"
    ]
    assert target_schema["prefixItems"][1]["oneOf"][1]["minItems"] == 2


def test_endpoint_stereo_sidecar_round_trips_and_is_schema_legal():
    descriptor = StereoDescriptor(
        "tetrahedral",
        (2, 1, 3, 4, "@H:2"),
        None,
        "unknown",
        "reviewed",
    )
    record = MechanismRecord(
        "[CH:2]([F:1])([Cl:3])[CH3:4]>>[CH:2]([F:1])([Cl:3])[CH3:4]",
        (),
        endpoint_stereo={"product": {"atom:2": descriptor}},
    )

    assert MechanismRecord.from_dict(record.to_dict()) == record
    endpoint_schema = mechanism_record_schema()["properties"]["endpoint_stereo"]
    assert endpoint_schema["properties"]["product"] == {
        "$ref": "#/$defs/stereoRegistry"
    }


def test_unspecified_stereo_reversal_fails_explicitly():
    unknown = StereoDescriptor(
        "tetrahedral",
        (2, 1, 3, 4, "@H:2"),
        None,
        "unknown",
    )
    effect = StereoEffect(("atom", 2), "UNSPECIFIED", after=unknown)

    with pytest.raises(MechanismModelError, match="NONREVERSIBLE_STEREO_EFFECT"):
        effect.reversed()


@pytest.mark.parametrize(
    ("logical_row", "row"),
    [
        (
            1,
            [
                "CC(C)(C[O:11][N+:10]([O-])=O)C.[Ar]>>CC(C)(C[O:11])C."
                "[O-][N+:10]=O.[Ar] 10,11-10;10,11-11",
                "Heat",
                "Initiation",
                "homolyze",
            ],
        ),
        (
            2,
            [
                "C[C:21](C)(C)[CH2:20][O:10]>>C[C:21](C)C.[CH2:20]=[O:10] "
                "10-10,20;20,21-10,20;20,21-21",
                "Room Temperature",
                "Propagation",
                "retroaddition",
            ],
        ),
        (
            3,
            [
                "C[C:10](C)C.[N+:20](=O)[O-]>>C[C:10](C)(C)[N+:20](=O)[O-] "
                "10-10,20;20-10,20",
                "Room Temperature",
                "Termination",
                "recombine",
            ],
        ),
        (
            23,
            [
                "[H:20][CH:21]=O.N(=O)[O:10]>>"
                "[H:20][O:10]N=O.[CH:21]=O "
                "10-10,20;20,21-10,20;20,21-21",
                "Room Temperature",
                "Propagation",
                "abstraction",
            ],
        ),
        (
            34,
            [
                "C=[C:10]=[CH2:11].[Br:20]>>[CH2:11][C:10]([Br:20])=C "
                "20-10,20;10,11-10,20;10,11-11",
                "Room Temperature",
                "Propagation",
                "addition",
            ],
        ),
        (
            37,
            [
                "C[CH:10][C:20](=[CH2:21])Br>>C[CH:10]=[C:20]([CH2:21])Br "
                "10-10,20;20,21-10,20;20,21-21",
                "Room Temperature",
                "Propagation",
                "resonance",
            ],
        ),
    ],
)
def test_radical_dataset_adapter_normalizes_representative_macros(logical_row, row):
    normalized = normalize_radical_row(row, row_number=logical_row)

    assert normalized.accepted, normalized.report.issues
    assert normalized.mechanism is not None
    group = normalized.mechanism.steps[0].groups[0]
    assert group.issues() == ()
    assert all(move.electron_count == 1 for move in group.moves)
    assert all(move.arrow_type == "fishhook" for move in group.moves)
    assert all(
        move.source.kind in {"lp", "σ", "π", "∙"}
        and move.target.kind in {"lp", "σ", "π", "∙"}
        for move in group.moves
    )


def test_radical_dataset_adapter_types_lone_pair_radical_relocation():
    relocated = normalize_radical_row(
        ["[O-:1][N+:2]=O>>[O:1][N:2]=O 1-2", "", "", "ha resonance"],
        row_number=1,
    )

    assert relocated.accepted
    group = relocated.mechanism.steps[0].groups[0]
    assert group.macro == "LONE_PAIR_RADICAL_RELOCATION"
    assert group.issues() == ()
    move = group.moves[0]
    assert move.source == ElectronLocus.atom("lp", atom_map=1)
    assert move.target == ElectronLocus.atom("lp", atom_map=2)
    assert "CompactAtomArrow→state-resolved:lp→lp" in (
        relocated.report.aliases_normalized
    )


def test_radical_dataset_adapter_can_reconstruct_without_source_macro_policy():
    row = [
        "CC(C)(C[O:11][N+:10]([O-])=O)C.[Ar]>>CC(C)(C[O:11])C."
        "[O-][N+:10]=O.[Ar] 10,11-10;10,11-11",
        "Heat",
        "Initiation",
        "recombine",
    ]

    class_checked = normalize_radical_row(row, row_number=1)
    reconstruction_only = normalize_radical_row(
        row, row_number=1, enforce_source_macro=False
    )

    assert not class_checked.accepted
    assert reconstruction_only.accepted, reconstruction_only.report.issues
    assert reconstruction_only.mechanism.steps[0].groups[0].macro is None


def test_radical_dataset_adapter_rejects_reversed_atom_transfer_annotation():
    reversed_arrow = normalize_radical_row(
        [
            "[N:10](=O)[O:21]>>[N+:10](=O)[O-:21] 21-10",
            "",
            "",
            "ha resonance",
        ],
        row_number=2046,
    )

    assert not reversed_arrow.accepted
    assert reversed_arrow.report.issues[0].code == ("FLOW_ATOM_TRANSFER_STATE_MISMATCH")


def test_radical_dataset_adapter_quarantines_bad_separator():
    bad_separator = normalize_radical_row(
        ["[B:2].[O:1]>>[B:2][O:1] 1=2-2", "", "", "recombine"],
        row_number=2,
    )

    assert not bad_separator.accepted
    assert bad_separator.report.issues[0].code == "MALFORMED_ELECTRON_FLOW"
    assert "MALFORMED_FLOW_SEPARATOR" in bad_separator.report.issues[0].message


def test_radical_dataset_adapter_normalizes_unambiguous_legacy_flow_syntax():
    equals_flow = normalize_radical_row(
        ["[B:2].[O:1]>>[B:2][O:1] 1=2", "", "", "addition"],
        row_number=540,
    )
    pair_hyphen = normalize_radical_row(
        [
            "C[C:21](=[CH2:20])[CH:22]=[CH2:23].[OH:10]>>"
            "[CH2:23]/[CH:22]=[C:21]([CH2:20][OH:10])\\C "
            "10-10,20;20,21-10,20;20,21-21,22;"
            "22,23-21-22;22,23-23",
            "Room Temperature",
            "Propagation",
            "addition",
        ],
        row_number=2928,
    )

    assert not equals_flow.accepted
    assert "FlowSeparator=→-" in equals_flow.report.aliases_normalized
    assert all(
        issue.code != "MALFORMED_ELECTRON_FLOW" for issue in equals_flow.report.issues
    )
    assert not pair_hyphen.accepted
    assert "FlowPairSeparator-→," in pair_hyphen.report.aliases_normalized
    assert all(
        issue.code != "MALFORMED_ELECTRON_FLOW" for issue in pair_hyphen.report.issues
    )


def test_radical_dataset_adapter_retains_flow_steps_after_whitespace():
    compact = normalize_radical_row(
        [
            "[H:20][Br:21].[OH:10]>>[H:20][OH:10].[Br:21] "
            "10-10,20;20,21-10,20;20,21-21",
            "Room Temperature",
            "Propagation",
            "abstraction",
        ]
    )
    spaced = normalize_radical_row(
        [
            "[H:20][Br:21].[OH:10]>>[H:20][OH:10].[Br:21] "
            "10-10,20; 20,21-10,20; 20,21-21",
            "Room Temperature",
            "Propagation",
            "abstraction",
        ]
    )

    assert compact.accepted, compact.report.issues
    assert spaced.accepted, spaced.report.issues
    assert spaced.mechanism is not None
    assert compact.mechanism is not None
    assert spaced.mechanism.mapped_reaction == compact.mechanism.mapped_reaction
    assert spaced.mechanism.steps == compact.mechanism.steps
    assert "FlowWhitespace→canonical" not in compact.report.aliases_normalized
    assert "FlowWhitespace→canonical" in spaced.report.aliases_normalized


def test_radical_dataset_adapter_explicitly_ignores_stereo_during_aam():
    normalized = normalize_radical_row(
        [
            "[H:20][Br:21].CCC/[C:10]=C/Br>>"
            "[H:20]/[C:10](=C\\Br)/CCC.[Br:21] "
            "10-10,20;20,21-10,20;20,21-21",
            "Room Temperature",
            "Propagation",
            "abstraction",
        ],
        row_number=861,
    )

    assert normalized.accepted, normalized.report.issues
    assert normalized.report.stereochemistry_ignored_for_expansion
    assert "Stereo→constitution-only" in normalized.report.aliases_normalized
    assert normalized.report.constitution_checked
    assert normalized.report.radical_state_preserved
    assert normalized.report.spin_assessment == "UNASSESSED_NO_SOURCE_EVIDENCE"


def test_radical_dataset_adapter_classifies_unrepairable_flow_references():
    missing_map = normalize_radical_row(
        ["[OH:1]>>[OH:1] 1-1; 1-2", "", "", "recombine"]
    )
    missing_bond = normalize_radical_row(
        ["[O:1].[O:2]>>[O:1].[O:2] 1-1,2", "", "", "recombine"]
    )

    assert missing_map.report.issues[0].code == "FLOW_ATOM_MAP_MISSING"
    assert "FlowWhitespace→canonical" in missing_map.report.aliases_normalized
    assert missing_bond.report.issues[0].code == "FLOW_BOND_ABSENT_FROM_ITS"


def test_radical_dataset_adapter_preserves_mapped_radical_state():
    normalized = normalize_radical_row(
        [
            "[CH2:10][C:20]1=[CH:21]C=CC=C1>>"
            "[CH2:10]=[C:20]1C=CC=C[CH:21]1 "
            "10-10,20;20,21-10,20;20,21-21",
            "Room Temperature",
            "Propagation",
            "resonance",
        ],
        row_number=1491,
    )

    assert normalized.accepted, normalized.report.issues
    assert normalized.report.aam_expanded
    assert normalized.report.aam_expansion_side == "reactant"
    assert not normalized.report.aam_fallback_used
    assert normalized.report.constitution_checked
    assert normalized.mechanism is not None


def test_radical_dataset_adapter_does_not_remap_complete_aam():
    reaction = "[H:20][Br:21].[OH:10]>>[H:20][OH:10].[Br:21]"
    normalized = normalize_radical_row(
        [
            f"{reaction} 10-10,20;20,21-10,20;20,21-21",
            "Room Temperature",
            "Propagation",
            "abstraction",
        ],
        row_number=32,
    )

    assert normalized.accepted, normalized.report.issues
    assert not normalized.report.aam_expanded
    assert normalized.report.aam_expansion_side is None
    assert normalized.report.constitution_checked
    assert normalized.mechanism is not None
    assert normalized.mechanism.mapped_reaction == reaction


def test_radical_dataset_adapter_folds_unmapped_explicit_hydrogens():
    normalized = normalize_radical_row(
        [
            "[H][C:10]([H])[H].[H]/[C:20]([H])=[C:21]([H])/[H]>>"
            "[CH2:21][C:20]([H])([H])[C:10]([H])([H])[H] "
            "10-10,20;20,21-10,20;20,21-21",
            "Room Temperature",
            "Propagation",
            "addition",
        ],
        row_number=184,
    )

    assert normalized.accepted, normalized.report.issues
    assert normalized.report.unmapped_explicit_hydrogens_folded
    assert normalized.report.folded_unmapped_explicit_hydrogen_count > 0
    assert normalized.mechanism is not None


def test_radical_dataset_adapter_preserves_folded_explicit_hydrogen_count():
    normalized = normalize_radical_row(
        [
            "CC(C)(C[O:20])C.CC(C)([CH:11]([H:10])[O:12])C>>"
            "CC(C)(C[O:20][H:10])C.CC(C)([C:11]([H])=[O:12])C "
            "20-20,10;10,11-20,10;10,11-11,12;12-11,12",
            "Room Temperature",
            "Termination",
            "abstraction",
        ],
        row_number=4,
    )

    assert normalized.accepted, normalized.report.issues
    assert normalized.report.unmapped_explicit_hydrogens_folded
    assert normalized.report.folded_unmapped_explicit_hydrogen_count == 1
    assert normalized.mechanism is not None
    assert "[CH:11]=[O:12]" in normalized.mechanism.mapped_reaction


def test_radical_aam_completion_is_independent_of_arrow_grammar():
    reaction = "[H:20][CH:21]=O.N(=O)[O:10]>>" "[H:20][O:10]N=O.[CH:21]=O"

    completed = complete_radical_aam(reaction)

    assert completed.usable, completed.failure_reason
    assert completed.status == "COMPLETED"
    assert completed.method == "its"
    assert completed.all_atoms_mapped
    assert completed.balanced_atom_maps
    assert completed.source_anchors_preserved
    assert completed.constitution_preserved
    assert completed.explicit_hydrogen_serialization
    assert completed.mapped_reaction is not None
    assert all(
        f":{atom_map}]" in completed.mapped_reaction for atom_map in (10, 20, 21)
    )


def test_radical_aam_completion_preserves_mapped_spectator_hydrogens():
    reaction = (
        "[H:1][C:2]([H:3])([H:4])[H:5].[I:6].[He]>>"
        "[He].[H:3][C:2]([H:4])[H:1].[H:5][I:6]"
    )

    completed = complete_radical_aam(reaction)

    assert completed.usable, completed.failure_reason
    assert completed.mapped_reaction is not None
    assert {
        atom.GetAtomMapNum()
        for side in completed.mapped_reaction.split(">>")
        for atom in Chem.MolFromSmiles(side, sanitize=False).GetAtoms()
    } == set(range(1, 8))


def test_radical_aam_completion_validates_complete_source_and_fails_closed():
    complete = complete_radical_aam("[H:20][Br:21].[OH:10]>>[H:20][OH:10].[Br:21]")
    duplicate = complete_radical_aam("[O:1].[C:1]>>[O:1].[C:1]")

    assert complete.usable
    assert complete.status == "ALREADY_COMPLETE"
    assert complete.method == "source"
    assert duplicate.status == "FAILED"
    assert not duplicate.usable
    assert duplicate.mapped_reaction is None
    assert "duplicate" in duplicate.failure_reason.lower()


@pytest.mark.parametrize(
    ("reaction", "flow", "source_class"),
    [
        (
            "CCCC(C(C)[O:20][O:21][O])O[N+](=O)[O-]>>"
            "CCCC(C(C)[O:20])O[N+](=O)[O-].[O:21][O]",
            "20,21-20;20,21-2",
            "homolyze",
        ),
        (
            "[H:20][CH:21](C)CC(=O)CC.[OH:10]>>" "[H:20][OH:10].CCC(=O)C[CH:21]C",
            "10-10,20;20,21-10,20;20,21-2",
            "abstraction",
        ),
    ],
)
def test_radical_aam_completion_ignores_nonexistent_arrow_map(
    reaction, flow, source_class
):
    arrow_normalization = normalize_radical_row(
        [f"{reaction} {flow}", "", "", source_class]
    )
    completed = complete_radical_aam(reaction)

    assert arrow_normalization.report.issues[0].code == "FLOW_ATOM_MAP_MISSING"
    assert completed.usable, completed.failure_reason


def test_radical_aam_completion_retains_non_arrow_skeletal_anchor():
    reaction = (
        "[H:20][O:21][O].CC[C:11]12CC(C(C=C1)(C)O[O])O[O:10]2>>"
        "[H:20][O:10]OC1C[C:11](C=CC1(C)O[O])CC.[O:21][O]"
    )
    arrow_normalization = normalize_radical_row(
        [
            f"{reaction} 10-10,20; 21,20-10,20; 20,21-21",
            "Room Temperature",
            "Initiation",
            "abstraction",
        ],
        row_number=4325,
    )
    completed = complete_radical_aam(reaction)

    assert arrow_normalization.accepted, arrow_normalization.report.issues
    assert completed.usable, completed.failure_reason
    assert completed.mapped_reaction is not None
    assert ":11]" in completed.mapped_reaction


def test_radical_arrow_normalization_repairs_unique_reactant_symmetry_swap():
    normalized = normalize_radical_row(
        [
            "[CH3:1][C:2]([CH3:3])[C:4]1=[CH:5][CH:6]=[CH:7]"
            "[CH:8]=[CH:9]1.[O:10][O:11]>>"
            "[CH3:1][C:2]([CH3:3])([C:4]1=[CH:5][CH:6]=[CH:7]"
            "[CH:8]=[CH:9]1)[O:10][O:11] 11-11,2; 2-11,2",
            "Room Temperature",
            "Termination",
            "recombine",
        ],
        row_number=56,
    )

    assert normalized.accepted, normalized.report.issues
    assert normalized.report.equivalent_map_swap == (10, 11)
    assert normalized.report.arrow_repair_assessment == "REACTANT_SYMMETRY_UNIQUE"
    assert "EquivalentReactantMapSwap:10↔11" in normalized.report.aliases_normalized
    assert normalized.report.to_dict()["equivalent_map_swap"] == [10, 11]


def test_radical_arrow_symmetry_swap_does_not_weaken_macro_grammar():
    normalized = normalize_radical_row(
        [
            "[CH:6]=[CH:7][CH:8].[O:23][O:24]>>"
            "[CH:6]([CH:7]=[CH:8])[O:23][O:24] "
            "8-8,7;6,7-7,8;6,7-6,24;24-24,6",
            "",
            "",
            "recombine",
        ],
        row_number=134,
    )

    assert not normalized.accepted
    assert normalized.report.equivalent_map_swap == (23, 24)
    assert normalized.report.issues[0].code == "UNBALANCED_EVENT_GROUP"


def test_radical_arrow_normalization_rejects_nonequivalent_map_repair():
    normalized = normalize_radical_row(
        [
            "[CH2:22]=[C:21].[S:5]>>[C:21][CH2:22][S:5] " "5-5,21;22,21-5,22;22,21-21",
            "",
            "",
            "addition",
        ],
        row_number=404,
    )

    assert not normalized.accepted
    assert normalized.report.equivalent_map_swap is None
    assert normalized.report.issues[0].code == "FLOW_BOND_ABSENT_FROM_ITS"


def test_event_group_signature_is_permutation_invariant():
    source = ElectronLocus.bond("σ", atom_maps=(1, 2))
    first = ElectronMove(
        source,
        ElectronLocus.atom("∙", atom_map=1),
        1,
        "fishhook",
        "g1",
        coupling_id="c1",
    )
    second = ElectronMove(
        source,
        ElectronLocus.atom("∙", atom_map=2),
        1,
        "fishhook",
        "g1",
        coupling_id="c1",
    )

    left = ElectronMoveGroup("g1", (first, second), macro="HOMOLYSIS")
    right = ElectronMoveGroup("g1", (second, first), macro="HOMOLYSIS")

    assert left.canonical_signature() == right.canonical_signature()
