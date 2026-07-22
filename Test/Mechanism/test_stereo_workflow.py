import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo import TetrahedralStereo

from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanisticStep,
    StereoDescriptor,
    StereoEffect,
    apply_stereo_effects,
    mechanism_equivalent,
    project_record,
    corrupt_record,
    benchmark_release_issues,
    BenchmarkCase,
    stereo_graph_from_gml,
    stereo_graph_to_gml,
)


def _polar_record(maps=(1, 2)):
    oxygen, carbon = maps
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=oxygen),
        ElectronLocus.bond("sigma", atom_maps=(oxygen, carbon)),
        2,
        "curved",
        "g1",
    )
    return MechanismRecord(
        f"[OH-:{oxygen}].[CH3+:{carbon}]>>[CH3:{carbon}][OH:{oxygen}]",
        (MechanisticStep("s1", (ElectronMoveGroup("g1", (move,)),)),),
    )


def test_five_line_public_workflow_and_json_round_trip(tmp_path):
    record = _polar_record()
    certificate = record.verify(electron="strict", stereo="stepwise")
    mtg = record.to_mtg()
    json_path = tmp_path / "mechanism.json"
    svg_path = tmp_path / "mechanism.svg"

    record.to_json(json_path)
    restored = MechanismRecord.from_json(json_path)
    figure = record.draw(certificate=certificate, path=svg_path)

    assert certificate.status == "VALID"
    assert mtg.graph["verify_stereo"] == "stepwise"
    assert restored == record
    assert svg_path.read_text().startswith("<?xml")
    plt.close(figure)


def test_equivalence_is_invariant_to_atom_map_permutation():
    assert mechanism_equivalent(
        _polar_record((1, 2)), _polar_record((20, 10)), level="events"
    )
    assert mechanism_equivalent(
        _polar_record((1, 2)), _polar_record((20, 10)), level="trajectory"
    )


def test_equivalence_uses_graph_correspondence_not_rdkit_rank_or_atom_order(
    monkeypatch,
):
    right = _polar_record((20, 10))
    reordered = MechanismRecord(
        "[CH3+:10].[OH-:20]>>[OH:20][CH3:10]",
        right.steps,
    )

    def fail_if_ranked(*args, **kwargs):
        raise AssertionError("RDKit canonical ranks are not an identity authority")

    monkeypatch.setattr(Chem, "CanonicalRankAtoms", fail_if_ranked)

    assert mechanism_equivalent(_polar_record(), reordered, level="events")
    assert mechanism_equivalent(_polar_record(), reordered, level="trajectory")


def test_net_equivalence_requires_one_atom_mapping_across_both_sides():
    left = MechanismRecord(
        "[OH-:1].[CH3:2][O:3][CH2:4][F:5]>>" "[CH3:2][OH:1].[O-:3][CH2:4][F:5]",
        (),
    )
    oxygen_origins_exchanged = MechanismRecord(
        "[OH-:1].[CH3:2][O:3][CH2:4][F:5]>>" "[CH3:2][OH:3].[O-:1][CH2:4][F:5]",
        (),
    )

    assert not mechanism_equivalent(left, oxygen_origins_exchanged, level="net")


def test_net_equivalence_translates_atom_order_tags_to_relative_stereo():
    side = "[CH3:1][C@:2]([F:3])([Cl:4])[Br:5]"
    molecule = Chem.MolFromSmiles(side)
    reordered = Chem.RenumberAtoms(molecule, [4, 2, 1, 3, 0])
    for atom in reordered.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + 10)
    reordered_side = Chem.MolToSmiles(
        reordered,
        canonical=False,
        isomericSmiles=True,
    )
    original = MechanismRecord(f"{side}>>{side}", ())
    relabeled = MechanismRecord(f"{reordered_side}>>{reordered_side}", ())
    enantiomer = MechanismRecord(
        f"{side.replace('C@:', 'C@@:')}>>{side.replace('C@:', 'C@@:')}",
        (),
    )

    assert mechanism_equivalent(original, relabeled, level="net")
    assert not mechanism_equivalent(original, enantiomer, level="net")


def test_net_equivalence_preserves_endpoint_sidecar_state_semantics():
    side = "[CH:2]([F:1])([Cl:3])[CH3:4]"
    relabeled_side = "[Cl:30][CH:20]([CH3:40])[F:10]"
    unknown = StereoDescriptor(
        "tetrahedral",
        (2, 1, 3, 4, "@H:2"),
        None,
        "unknown",
    )
    relabeled_unknown = StereoDescriptor(
        "tetrahedral",
        (20, 10, 30, 40, "@H:20"),
        None,
        "unknown",
    )
    unspecified = StereoDescriptor(
        "tetrahedral",
        unknown.atoms,
        None,
        "unspecified",
    )
    original = MechanismRecord(
        f"{side}>>{side}",
        (),
        endpoint_stereo={"product": {"atom:2": unknown}},
    )
    relabeled = MechanismRecord(
        f"{relabeled_side}>>{relabeled_side}",
        (),
        endpoint_stereo={"product": {"atom:20": relabeled_unknown}},
    )
    different_state = MechanismRecord(
        f"{side}>>{side}",
        (),
        endpoint_stereo={"product": {"atom:2": unspecified}},
    )

    assert mechanism_equivalent(original, relabeled, level="net")
    assert not mechanism_equivalent(original, different_state, level="net")
    assert not mechanism_equivalent(
        original,
        MechanismRecord(f"{side}>>{side}", ()),
        level="net",
    )


def test_net_equivalence_preserves_isotope_identity():
    labeled = MechanismRecord("[13CH3:1]>>[13CH3:1]", ())
    unlabeled = MechanismRecord("[CH3:1]>>[CH3:1]", ())

    assert not mechanism_equivalent(labeled, unlabeled, level="net")


def test_conversion_reports_grouping_loss():
    converted, report = project_record(_polar_record(), "mapped_reaction_smiles")

    assert ">>" in converted
    assert not report.lossless
    assert "event_groups" in report.discarded_fields


def test_illegal_stereo_transition_is_stepwise_error():
    descriptor = StereoDescriptor("tetrahedral", (99, 1, 3, 4, "@H:99"), 1)
    effect = StereoEffect(("atom", 99), "INVERT", before=descriptor)
    record = MechanismRecord(
        "[CH3:1][C@:2]([F:3])([Cl:4])[H:5]>>" "[CH3:1][C@:2]([F:3])([Cl:4])[H:5]",
        (MechanisticStep("s1", (), (effect,)),),
    )

    certificate = record.verify(stereo="stepwise")

    assert certificate.status == "INVALID"
    assert "STEREO_TRANSITION_FROM_ABSENT" in {
        issue.code for issue in certificate.issues
    }


def test_stereo_effects_validate_before_relation_and_apply_atomically():
    graph = nx.Graph()
    for atom_map, element in ((1, "C"), (2, "C"), (3, "F"), (4, "Cl")):
        graph.add_node(
            atom_map,
            atom_map=atom_map,
            element=element,
            hcount=1 if atom_map == 2 else 0,
            lone_pairs=0,
        )
    graph.add_edges_from((2, ligand) for ligand in (1, 3, 4))
    before = StereoDescriptor("tetrahedral", (2, 1, 3, 4, "@H:2"), 1)
    inverted = StereoDescriptor("tetrahedral", before.atoms, -1)
    graph.graph["mechanism_stereo_descriptors"] = {"atom:2": before}

    preserved, preserve_issues = apply_stereo_effects(
        graph,
        (StereoEffect(("atom", 2), "PRESERVE", before, inverted),),
        step_id="s1",
    )
    atomic, atomic_issues = apply_stereo_effects(
        graph,
        (
            StereoEffect(("atom", 2), "BREAK", before),
            StereoEffect(
                ("atom", 99),
                "INVERT",
                StereoDescriptor("tetrahedral", (99, 1, 3, 4, "@H:99"), 1),
            ),
        ),
        step_id="s1",
    )

    assert {issue.code for issue in preserve_issues} == {"INVALID_STEREO_PRESERVATION"}
    assert preserved.graph["mechanism_stereo_descriptors"] == {"atom:2": before}
    assert "STEREO_TRANSITION_FROM_ABSENT" in {issue.code for issue in atomic_issues}
    assert atomic.graph["mechanism_stereo_descriptors"] == {"atom:2": before}


def test_gml_stereo_registry_round_trip():
    graph = nx.Graph()
    graph.add_nodes_from([(1, {"element": "C"}), (2, {"element": "F"})])
    graph.add_edge(1, 2)
    descriptor = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    graph.graph["stereo_descriptors"] = {"atom:1": descriptor}

    text, report = stereo_graph_to_gml(graph)
    restored = stereo_graph_from_gml(text)

    assert report.lossless
    assert restored.graph["stereo_descriptors"]["atom:1"] == descriptor


def test_benchmark_corruptions_and_release_gate_are_explicit():
    record = _polar_record()
    corruptions = corrupt_record(record)
    candidate = BenchmarkCase("polar-1", "polar", record, {"license": "project-owned"})

    assert len(corruptions) == 10
    assert len({item.corruption for item in corruptions}) == 10
    for item in corruptions:
        assert item.expected_issue_code in item.observed_issue_codes(), item.corruption
    assert "PARTITION_COUNT:polar:1/80" in benchmark_release_issues([candidate])
    assert "CHEMISTRY_REVIEW_REQUIRED:1" in benchmark_release_issues([candidate])


def test_event_equivalence_commutes_only_disjoint_adjacent_groups():
    def group(group_id, source, target):
        return ElectronMoveGroup(
            group_id,
            (
                ElectronMove(
                    ElectronLocus.atom("lp", atom_map=source),
                    ElectronLocus.atom("lp", atom_map=target),
                    2,
                    "curved",
                    group_id,
                ),
            ),
        )

    reaction = "[O-:1].[S-:2].[Cl-:3].[Br-:4]>>" "[O-:1].[S-:2].[Cl-:3].[Br-:4]"
    first, second = group("a", 1, 2), group("b", 3, 4)
    left = MechanismRecord(
        reaction, (MechanisticStep("s1", (first,)), MechanisticStep("s2", (second,)))
    )
    right = MechanismRecord(
        reaction, (MechanisticStep("s1", (second,)), MechanisticStep("s2", (first,)))
    )

    assert mechanism_equivalent(left, right, level="events")
    assert not mechanism_equivalent(left, right, level="trajectory")

    dependent = group("c", 2, 3)
    dep_left = MechanismRecord(
        reaction, (MechanisticStep("s1", (first,)), MechanisticStep("s2", (dependent,)))
    )
    dep_right = MechanismRecord(
        reaction, (MechanisticStep("s1", (dependent,)), MechanisticStep("s2", (first,)))
    )
    assert not mechanism_equivalent(dep_left, dep_right, level="events")
