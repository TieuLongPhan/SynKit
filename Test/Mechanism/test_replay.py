from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanismReplayer,
    MechanisticStep,
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
