import networkx as nx

from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanismReplayer,
    MechanisticStep,
    StereoDescriptor,
    StereoEffect,
)


def _record(reaction, group):
    return MechanismRecord(reaction, (MechanisticStep("s1", (group,)),))


def test_atomic_polar_lone_pair_to_sigma_replay():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("σ", atom_maps=(1, 2)),
        2,
        "curved",
        "g1",
    )
    group = ElectronMoveGroup("g1", (move,))
    result = MechanismReplayer().replay(
        _record("[OH-:1].[CH3+:2]>>[CH3:2][OH:1]", group)
    )

    assert result.certificate.status == "VALID", result.certificate.issues
    assert len(result.intermediates) == 1
    assert result.certificate.final_match["matches"]
    assert result.mtg.number_of_edges() == 1


def test_atomic_homolysis_replay_never_stores_half_fishhook_state():
    source = ElectronLocus.bond("σ", atom_maps=(1, 2))
    moves = (
        ElectronMove(
            source,
            ElectronLocus.atom("∙", atom_map=1),
            1,
            "fishhook",
            "g1",
            coupling_id="c1",
        ),
        ElectronMove(
            source,
            ElectronLocus.atom("∙", atom_map=2),
            1,
            "fishhook",
            "g1",
            coupling_id="c1",
        ),
    )
    result = MechanismReplayer().replay(
        _record(
            "[CH3:1][Cl:2]>>[CH3:1].[Cl:2]",
            ElectronMoveGroup("g1", moves, macro="HOMOLYSIS"),
        )
    )

    assert result.certificate.status == "VALID", result.certificate.issues
    assert len(result.intermediates) == 1
    assert sorted(
        attrs["radical"] for _, attrs in result.final_graph.nodes(data=True)
    ) == [1, 1]


def test_lone_pair_radical_relocation_commits_whole_resources():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.atom("lp", atom_map=2),
        1,
        "fishhook",
        "g1",
    )
    group = ElectronMoveGroup("g1", (move,), macro="LONE_PAIR_RADICAL_RELOCATION")

    result = MechanismReplayer().replay(
        _record(
            "[O-:1][N+:2]=[O:3]>>[O:1][N:2]=[O:3]",
            group,
        )
    )

    assert result.certificate.status == "VALID", result.certificate.issues
    assert result.certificate.final_match["matches"]
    by_map = {
        attrs["atom_map"]: attrs for _, attrs in result.final_graph.nodes(data=True)
    }
    assert (by_map[1]["lone_pairs"], by_map[1]["radical"]) == (2, 1)
    assert (by_map[2]["lone_pairs"], by_map[2]["radical"]) == (1, 0)


def test_missing_fishhook_partner_fails_before_commit():
    source = ElectronLocus.bond("σ", atom_maps=(1, 2))
    move = ElectronMove(
        source,
        ElectronLocus.atom("∙", atom_map=1),
        1,
        "fishhook",
        "g1",
        coupling_id="c1",
    )
    result = MechanismReplayer().replay(
        _record(
            "[CH3:1][Cl:2]>>[CH3:1].[Cl:2]",
            ElectronMoveGroup("g1", (move,), macro="HOMOLYSIS"),
        )
    )

    assert result.certificate.status == "INVALID"
    assert "MISSING_COUPLED_FISHHOOK" in {
        issue.code for issue in result.certificate.issues
    }
    assert result.intermediates == ()


def test_wrong_endpoint_produces_structured_product_mismatch():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("σ", atom_maps=(1, 2)),
        2,
        "curved",
        "g1",
    )
    result = MechanismReplayer().replay(
        _record("[OH-:1].[CH3+:2]>>[OH-:1].[CH3+:2]", ElectronMoveGroup("g1", (move,)))
    )

    assert result.certificate.status == "INVALID"
    assert result.certificate.issues[-1].code == "FINAL_PRODUCT_MISMATCH"


def test_endpoint_signature_includes_isotopes_and_all_mapped_components():
    isotope = MechanismRecord("[13CH4:1]>>[12CH4:1]", ())
    extra_component = MechanismRecord("[CH4:1]>>[CH4:1].[OH2:2]", ())

    isotope_result = MechanismReplayer().replay(isotope)
    component_result = MechanismReplayer().replay(extra_component)

    assert isotope_result.certificate.status == "INVALID"
    assert isotope_result.certificate.issues[-1].code == "FINAL_PRODUCT_MISMATCH"
    assert component_result.certificate.status == "INVALID"
    assert component_result.certificate.issues[-1].code == "FINAL_PRODUCT_MISMATCH"


def test_endpoint_signature_includes_hydrogen_and_lone_pair_resources():
    atomic_oxygen = nx.Graph()
    water = nx.Graph()
    atomic_oxygen.add_node(
        1,
        atom_map=1,
        element="O",
        isotope=0,
        charge=0,
        radical=0,
        hcount=0,
        lone_pairs=3,
        valence_electrons=6,
    )
    water.add_node(
        1,
        atom_map=1,
        element="O",
        isotope=0,
        charge=0,
        radical=0,
        hcount=2,
        lone_pairs=2,
        valence_electrons=6,
    )

    comparison = MechanismReplayer()._compare_graphs(atomic_oxygen, water)

    assert not comparison["matches"]


def test_default_endpoint_comparison_ignores_aromatic_kekule_phase():
    first = nx.Graph()
    second = nx.Graph()
    for graph in (first, second):
        graph.add_node(
            1,
            atom_map=1,
            element="C",
            isotope=0,
            charge=0,
            radical=0,
            hcount=1,
            lone_pairs=0,
            valence_electrons=4,
        )
        graph.add_node(
            2,
            atom_map=2,
            element="C",
            isotope=0,
            charge=0,
            radical=0,
            hcount=1,
            lone_pairs=0,
            valence_electrons=4,
        )
    first.add_edge(1, 2, order=1.5, sigma_order=1.0, pi_order=1.0)
    second.add_edge(1, 2, order=1.5, sigma_order=1.0, pi_order=0.0)

    assert MechanismReplayer()._compare_graphs(first, second)["matches"]
    assert not MechanismReplayer(aromatic_policy="kekule")._compare_graphs(
        first,
        second,
    )["matches"]


def test_unmapped_and_duplicate_endpoint_atoms_are_structured_errors():
    unmapped = MechanismRecord("[CH4:1]>>[CH4:1].O", ())
    duplicate = MechanismRecord("[OH-:1].[OH-:1]>>[OH-:1].[OH-:1]", ())

    unmapped_result = MechanismReplayer().replay(unmapped)
    duplicate_result = MechanismReplayer().replay(duplicate)

    assert "MISSING_ATOM_MAP" in {
        issue.code for issue in unmapped_result.certificate.issues
    }
    assert "DUPLICATE_ATOM_MAP" in {
        issue.code for issue in duplicate_result.certificate.issues
    }


def test_strict_stereo_failure_rolls_back_the_complete_step():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(1, 2)),
        2,
        "curved",
        "g1",
    )
    absent = StereoDescriptor(
        "tetrahedral",
        (99, 1, 2, 3, "@H:99"),
        1,
    )
    effect = StereoEffect(("atom", 99), "INVERT", before=absent)
    record = MechanismRecord(
        "[OH-:1].[CH3+:2]>>[CH3:2][OH:1]",
        (
            MechanisticStep(
                "s1",
                (ElectronMoveGroup("g1", (move,)),),
                (effect,),
            ),
        ),
    )

    result = MechanismReplayer(verify_stereo="stepwise").replay(record)

    assert result.certificate.status == "INVALID"
    assert result.intermediates == ()
    by_map = {
        attrs["atom_map"]: node for node, attrs in result.final_graph.nodes(data=True)
    }
    assert not result.final_graph.has_edge(by_map[1], by_map[2])
