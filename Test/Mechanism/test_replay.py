import json

import networkx as nx
import pytest

from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanismReplayer,
    MechanisticStep,
    StereoDescriptor,
    StereoEffect,
    VerificationCertificate,
    mechanism_from_legacy_epd,
)


TI_1023 = (
    "[CH:1]12[CH:2]3[CH:3]4[CH:4]5[CH:5]1[Ti:29]23451678"
    "([CH2:21][c:35]2[cH:18][cH:37][cH:16][cH:38][cH:19]2)"
    "([CH2:22][c:34]2[cH:17][cH:36][cH:15][cH:39][cH:20]2)"
    "[CH:6]2[CH:7]1[CH:8]6[CH:9]7[CH:10]28."
    "[cH:11]1[c:23]([CH+:42][c:24]2[cH:12][c:27]([F:32])"
    "[cH:41][c:28]([F:33])[cH:13]2)[cH:14][c:25]([F:30])"
    "[cH:40][c:26]1[F:31]>>"
    "[CH:1]12[CH:2]3[CH:3]4[CH:4]5[CH:5]1[Ti+:29]23451678"
    "([CH2:22][c:34]2[cH:17][cH:36][cH:15][cH:39][cH:20]2)"
    "[CH:6]2[CH:7]1[CH:8]6[CH:9]7[CH:10]28."
    "[cH:11]1[c:23]([CH:42]([CH2:21][c:35]2[cH:18][cH:37]"
    "[cH:16][cH:38][cH:19]2)[c:24]2[cH:12][c:27]([F:32])"
    "[cH:41][c:28]([F:33])[cH:13]2)[cH:14][c:25]([F:30])"
    "[cH:40][c:26]1[F:31]"
)

CARBENE_1478 = (
    "[C:1]([Cl:2])([Cl:3])([Cl:4])[H:6].[OH-:5]>>"
    "[C:1]([Cl:2])[Cl:3].[Cl-:4].[OH:5][H:6]"
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


def test_delta_authoritative_titanium_transfer_is_delta_consistent():
    record = mechanism_from_legacy_epd(
        TI_1023,
        [["Sigma-/Sigma+", [21, 29], [21, 42]]],
    )

    result = MechanismReplayer().replay(record)
    certificate = result.certificate

    assert certificate.status == "VALID", certificate.issues
    assert certificate.verification_level == "DELTA_CONSISTENT"
    assert certificate.transition_valid
    assert certificate.delta_charge_match
    assert certificate.endpoint_match
    assert not certificate.absolute_lwg_valid
    assert certificate.initial_absolute_residuals == {29: 8}
    assert certificate.product_absolute_residuals == {29: 8}
    assert certificate.initial_global_electron_residual == 8
    assert certificate.product_global_electron_residual == 8
    assert certificate.absolute_residuals_invariant
    assert len(result.intermediates) == 1
    assert certificate.step_reports[0]["charge_updates"] == [
        {"atom_map": 21, "before": 0, "delta": 0, "after": 0},
        {"atom_map": 29, "before": 0, "delta": 1, "after": 1},
        {"atom_map": 42, "before": 1, "delta": -1, "after": 0},
    ]


def test_titanium_transfer_rejects_wrong_product_charge():
    record = mechanism_from_legacy_epd(
        TI_1023.replace("[Ti+:29]", "[Ti:29]"),
        [["Sigma-/Sigma+", [21, 29], [21, 42]]],
    )

    certificate = MechanismReplayer().replay(record).certificate

    assert certificate.status == "INVALID"
    assert certificate.transition_valid
    assert not certificate.delta_charge_match
    assert certificate.verification_level == "INVALID"
    assert "DELTA_CHARGE_MISMATCH" in {
        issue.code for issue in certificate.issues
    }


def test_titanium_transfer_requires_its_declared_source_sigma_bond():
    reactants = TI_1023.split(">>", 1)[0]
    graph = MechanismReplayer._parse_side(reactants)
    lookup = MechanismReplayer._atom_map_lookup(graph)
    graph.remove_edge(lookup[21], lookup[29])
    group = mechanism_from_legacy_epd(
        TI_1023,
        [["Sigma-/Sigma+", [21, 29], [21, 42]]],
    ).steps[0].groups[0]

    _, report = MechanismReplayer()._apply_group(graph, group, step_id="s1")

    assert report.status == "INVALID"
    assert "SOURCE_LOCUS_ABSENT" in {
        issue.code for issue in report.issues
    }


def test_changed_haptic_contact_is_not_hidden_by_delta_consistency():
    reactants, products = TI_1023.split(">>", 1)
    initial = MechanismReplayer._parse_side(reactants)
    group = mechanism_from_legacy_epd(
        TI_1023,
        [["Sigma-/Sigma+", [21, 29], [21, 42]]],
    ).steps[0].groups[0]
    replayed, report = MechanismReplayer()._apply_group(
        initial,
        group,
        step_id="s1",
    )
    changed_product = MechanismReplayer._parse_side(products)
    lookup = MechanismReplayer._atom_map_lookup(changed_product)
    changed_product.remove_edge(lookup[1], lookup[29])

    comparison = MechanismReplayer()._compare_graphs(replayed, changed_product)

    assert report.status == "VALID"
    assert not comparison["matches"]
    assert MechanismReplayer._absolute_state(initial) == MechanismReplayer._absolute_state(
        replayed
    )
    assert (
        MechanismReplayer._absolute_state(initial)
        != MechanismReplayer._absolute_state(changed_product)
    )


def test_exact_main_group_replay_reports_exact_verification():
    move = ElectronMove(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(1, 2)),
        2,
        "curved",
        "g1",
    )

    certificate = MechanismReplayer().replay(
        _record(
            "[OH-:1].[CH3+:2]>>[CH3:2][OH:1]",
            ElectronMoveGroup("g1", (move,)),
        )
    ).certificate

    assert certificate.status == "VALID"
    assert certificate.verification_level == "EXACT"
    assert certificate.absolute_lwg_valid
    assert certificate.initial_absolute_residuals == {}


def test_constant_absolute_charge_offset_is_not_exact():
    endpoint = TI_1023.split(">>", 1)[0].replace("[Ti:29]", "[Ti+:29]")
    certificate = MechanismReplayer().replay(
        MechanismRecord(f"{endpoint}>>{endpoint}", ())
    ).certificate

    assert certificate.status == "VALID"
    assert certificate.verification_level == "DELTA_CONSISTENT"
    assert not certificate.absolute_lwg_valid
    assert certificate.absolute_residuals_invariant


def test_missing_odd_electron_can_only_be_delta_consistent():
    reaction = (
        "[O:12]=[Mn:13]=[O:14]."
        "[c:1]1([C:2]([H:9])([H:10])[O:7][H:11])"
        "[cH:3][cH:5][cH:8][cH:6][cH:4]1>>"
        "[O:12]([H:9])[H:11]."
        "[c:1]1([C:2]([H:10])=[O:7])"
        "[cH:3][cH:5][cH:8][cH:6][cH:4]1.[Mn:13]=[O:14]"
    )
    epd = [
        ["Sigma-/Pi+", [2, 9], [2, 7]],
        ["Sigma-/Sigma+", [7, 11], [11, 12]],
        ["Pi-/Sigma+", [12, 13], [9, 12]],
        ["Sigma-/LP+", [12, 13], [13]],
    ]

    certificate = MechanismReplayer().replay(
        mechanism_from_legacy_epd(reaction, epd)
    ).certificate

    assert certificate.status == "VALID", certificate.issues
    assert certificate.verification_level == "DELTA_CONSISTENT"
    assert certificate.transition_valid
    assert certificate.endpoint_match
    assert certificate.absolute_residuals_invariant
    assert not certificate.absolute_lwg_valid


def test_carbene_endpoint_is_strictly_invalid_and_explicitly_normalized():
    record = mechanism_from_legacy_epd(
        CARBENE_1478,
        [
            ["LP-/Sigma+", [5], [5, 6]],
            ["Sigma-/LP+", [1, 6], [1]],
            ["Sigma-/LP+", [1, 4], [4]],
        ],
    )

    strict = MechanismReplayer().replay(record)
    normalized = MechanismReplayer(
        endpoint_resource_policy="closed_shell_pair",
        closed_shell_atom_maps={1},
    ).replay(record)

    assert strict.certificate.status == "INVALID"
    assert strict.certificate.transition_valid
    assert strict.certificate.delta_charge_match
    assert not strict.certificate.endpoint_match
    assert strict.certificate.verification_level == "INVALID"
    assert normalized.certificate.status == "VALID", normalized.certificate.issues
    assert normalized.certificate.verification_level == "NORMALIZED"
    assert normalized.certificate.endpoint_match
    assert not normalized.certificate.final_match["raw_matches"]
    assert normalized.certificate.normalization_evidence == (
        {
            "atom_map": 1,
            "element": "C",
            "side": "expected",
            "before": {"charge": 0, "lone_pairs": 0, "radical": 2},
            "after": {"charge": 0, "lone_pairs": 1, "radical": 0},
            "policy_name": "closed_shell_pair",
            "policy_version": "1.0",
        },
    )
    final_by_map = {
        attrs["atom_map"]: attrs
        for _, attrs in normalized.final_graph.nodes(data=True)
    }
    assert (final_by_map[1]["lone_pairs"], final_by_map[1]["radical"]) == (1, 0)
    parsed_product = MechanismReplayer._parse_side(CARBENE_1478.split(">>", 1)[1])
    product_by_map = {
        attrs["atom_map"]: attrs for _, attrs in parsed_product.nodes(data=True)
    }
    assert (product_by_map[1]["lone_pairs"], product_by_map[1]["radical"]) == (0, 2)


@pytest.mark.parametrize("element", ["C", "I", "S", "Pb"])
def test_closed_shell_pair_policy_is_element_generic_and_copy_only(element):
    paired = nx.Graph()
    diradical = nx.Graph()
    base = {
        "atom_map": 7,
        "element": element,
        "isotope": 0,
        "charge": 0,
        "hcount": 0,
        "valence_electrons": 4,
    }
    paired.add_node(1, **base, lone_pairs=1, radical=0)
    diradical.add_node(1, **base, lone_pairs=0, radical=2)

    strict = MechanismReplayer()._compare_graphs(paired, diradical)
    normalized = MechanismReplayer(
        endpoint_resource_policy="closed_shell_pair",
        closed_shell_atom_maps={7},
    )._compare_graphs(paired, diradical)

    assert not strict["matches"]
    assert normalized["matches"]
    assert normalized["normalization_evidence"][0]["element"] == element
    assert paired.nodes[1]["lone_pairs"] == 1
    assert diradical.nodes[1]["radical"] == 2


def test_nonlocal_move_remains_invalid_under_delta_replay():
    move = ElectronMove(
        ElectronLocus.bond("sigma", atom_maps=(1, 2)),
        ElectronLocus.bond("sigma", atom_maps=(3, 4)),
        2,
        "curved",
        "g1",
    )
    record = _record(
        "[CH3:1][Cl:2].[CH3:3][Cl:4]>>[CH3:1][Cl:2].[CH3:3][Cl:4]",
        ElectronMoveGroup("g1", (move,)),
    )

    certificate = MechanismReplayer().replay(record).certificate

    assert certificate.status == "INVALID"
    assert not certificate.transition_valid
    assert "NONLOCAL_ELECTRON_MOVE" in {
        issue.code for issue in certificate.issues
    }


def test_verification_certificate_round_trip_preserves_delta_evidence():
    certificate = MechanismReplayer().replay(
        mechanism_from_legacy_epd(
            TI_1023,
            [["Sigma-/Sigma+", [21, 29], [21, 42]]],
        )
    ).certificate

    restored = VerificationCertificate.from_dict(
        json.loads(json.dumps(certificate.to_dict()))
    )

    assert restored.verification_level == certificate.verification_level
    assert restored.transition_valid == certificate.transition_valid
    assert restored.initial_absolute_residuals == {29: 8}
    assert restored.product_absolute_residuals == {29: 8}
    assert restored.absolute_residuals_invariant
    assert restored.step_reports == certificate.step_reports
