import matplotlib.pyplot as plt
import networkx as nx

from synkit.Graph.Stereo import TetrahedralStereo

from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanisticStep,
    StereoDescriptor,
    StereoEffect,
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


def test_conversion_reports_grouping_loss():
    converted, report = project_record(_polar_record(), "mapped_reaction_smiles")

    assert ">>" in converted
    assert not report.lossless
    assert "event_groups" in report.discarded_fields


def test_illegal_stereo_transition_is_stepwise_error():
    descriptor = StereoDescriptor("tetrahedral", (2, 1, 3, 4, "@H:2"), 1)
    effect = StereoEffect(("atom", 99), "INVERT", before=descriptor)
    record = MechanismRecord(
        "[CH3:1][C@:2]([F:3])([Cl:4])[H]>>[CH3:1][C@:2]([F:3])([Cl:4])[H]",
        (MechanisticStep("s1", (), (effect,)),),
    )

    certificate = record.verify(stereo="stepwise")

    assert certificate.status == "INVALID"
    assert "STEREO_TRANSITION_FROM_ABSENT" in {
        issue.code for issue in certificate.issues
    }


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

    reaction = "[O-:1].[O-:2].[O-:3].[O-:4]>>[O-:1].[O-:2].[O-:3].[O-:4]"
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
