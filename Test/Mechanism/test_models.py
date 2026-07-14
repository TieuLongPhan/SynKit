import networkx as nx
import pytest
from rdkit import Chem
import csv
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
    PUBLIC_LEWIS_GRAPH_ACRONYM,
    audit_local_electron_state,
    external_locus_symbol,
    group_from_legacy_epd,
    mechanism_from_legacy_epd,
    mechanism_record_schema,
    radical_match,
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


def test_group_rejects_simultaneous_overconsumption_before_replay():
    graph = nx.Graph()
    graph.add_node(1, atom_map=1, radical=1)
    graph.add_node(2, atom_map=2, radical=1)
    graph.add_edge(1, 2, sigma_order=1.0, pi_order=0.0)
    source = ElectronLocus.bond("sigma", atom_maps=(1, 2))
    moves = (
        ElectronMove(source, ElectronLocus.atom("rad", atom_map=1), 2, "curved", "g1"),
        ElectronMove(source, ElectronLocus.atom("rad", atom_map=2), 2, "curved", "g1"),
    )

    assert "LOCUS_OVERCONSUMED" in {
        issue.code for issue in ElectronMoveGroup("g1", moves).validate_pre_state(graph)
    }


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
    contract = json.loads(path.read_text())

    assert PUBLIC_LEWIS_GRAPH_ACRONYM == "LSG"
    assert contract["public_v2_acronym"] == "LSG"
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


@pytest.mark.parametrize("logical_row", [1, 2, 3, 4, 34, 37])
def test_radical_dataset_adapter_normalizes_representative_macros(logical_row):
    path = Path(__file__).parents[2] / "Data/Mech/radical.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = list(csv.reader(handle))[logical_row - 1]

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


def test_radical_dataset_adapter_quarantines_unreviewed_class_and_bad_separator():
    unreviewed = normalize_radical_row(
        ["[O-:1][N+:2]=O>>[O:1][N:2]=O 1-2", "", "", "ha resonance"],
        row_number=1,
    )
    bad_separator = normalize_radical_row(
        ["[B:2].[O:1]>>[B:2][O:1] 1=2", "", "", "recombine"],
        row_number=2,
    )

    assert not unreviewed.accepted
    assert unreviewed.report.issues[0].code == "UNREVIEWED_RADICAL_CLASS"
    assert not bad_separator.accepted
    assert "NONSTANDARD_FLOW_SEPARATOR" in bad_separator.report.issues[0].message


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
