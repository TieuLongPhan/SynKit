import inspect

import pytest

from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor.batch_reactor import BatchReactor
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ETHANE_DEHYDROGENATION = "[CH2:1]([H:3])[CH2:2]([H:4])>>[CH2:1]=[CH2:2].[H:3][H:4]"

DIELS_ALDER = (
    "[CH2:1]=[CH:2][CH:3]=[CH2:4]."
    "[CH2:5]=[CH:6][CH:7]=[O:8]>>"
    "[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH:6]1[CH:7]=[O:8]"
)

MESO_HYDROGENATION = (
    "[CH3:1][CH:2]=[CH:3][CH3:4].[H:5][H:6]>>"
    "[CH3:1][CH:2]([H:5])[CH:3]([H:6])[CH3:4]"
)

GENERIC_SYN_HYDROGENATION = "[C:1]=[C:2].[H:3][H:4]>>[C:1]([H:3])[C:2]([H:4])"

AROMATIC_CROSS_COUPLING = (
    "[CH3:10][CH2:11][O:12][C:13](=[O:14])[c:15]1[cH:16][cH:18]"
    "[cH:19][c:20]([B:21]([OH:22])[OH:23])[cH:17]1."
    "[CH3:1][c:2]1[cH:3][cH:5][cH:6][c:7]([Br:8])[c:4]1[I:9]>>"
    "[CH3:1][c:2]1[cH:3][cH:5][cH:6][c:7]([Br:8])[c:4]1-"
    "[c:20]1[cH:17][c:15]([C:13]([O:12][CH2:11][CH3:10])=[O:14])"
    "[cH:16][cH:18][cH:19]1.[I:9][B:21]([OH:22])[OH:23]"
)


def _ethane_reactor(*, automorphism, dedup_its, diagnostics=False):
    return SynReactor(
        "CC",
        ETHANE_DEHYDROGENATION,
        template_format="tuple",
        explicit_h=False,
        automorphism=automorphism,
        dedup_its=dedup_its,
        electron_diagnostics=diagnostics,
    )


@pytest.mark.parametrize(
    ("automorphism", "dedup_its", "expected_mappings", "expected_its"),
    [
        (True, True, 1, 1),
        (True, False, 1, 18),
        (False, True, 2, 1),
        (False, False, 2, 36),
    ],
)
def test_automorphism_and_its_dedup_are_orthogonal(
    automorphism,
    dedup_its,
    expected_mappings,
    expected_its,
):
    reactor = _ethane_reactor(
        automorphism=automorphism,
        dedup_its=dedup_its,
    )

    assert reactor.mapping_count == expected_mappings
    assert len(reactor.its_list) == expected_its
    assert len(reactor.smarts_list) == expected_its
    assert all(
        its.graph.get("_product_electron_fields_current") for its in reactor.its_list
    )

    if dedup_its:
        assert all(
            "application_provenance" not in its.graph for its in reactor.its_list
        )
    else:
        provenance = [its.graph["application_provenance"] for its in reactor.its_list]
        assert [entry["application_index"] for entry in provenance] == list(
            range(expected_its)
        )
        assert all(entry["mapping"] for entry in provenance)


@pytest.mark.parametrize("template_format", ["typesGH", "tuple"])
def test_raw_mode_preserves_equivalent_application_multiplicity(template_format):
    consolidated = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format=template_format,
        explicit_h=False,
        automorphism=False,
    )
    raw = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format=template_format,
        explicit_h=False,
        automorphism=False,
        dedup_its=False,
    )

    assert consolidated.mapping_count == raw.mapping_count == 2
    assert len(consolidated.its_list) == len(consolidated.smarts_list) == 1
    assert len(raw.its_list) == len(raw.smarts_list) == 2
    assert len(SynReactor._deduplicate_structural_its(list(raw.its_list))) == 1
    assert [
        its.graph["application_provenance"]["mapping_index"] for its in raw.its_list
    ] == [0, 1]


def test_raw_mode_preserves_meso_face_branches_before_aggregation():
    rule = SynRule.from_smart(
        MESO_HYDROGENATION,
        format="tuple",
        implicit_h=True,
        stereo_couplings={"bond:2-3": "SYN"},
    )
    substrate = "C/C(CC)=C(CC)\\C.[H][H]"
    consolidated = SynReactor(
        substrate,
        rule,
        template_format="tuple",
        explicit_h=False,
    )
    raw = SynReactor(
        substrate,
        rule,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
    )

    assert len(consolidated.its_list) == 1
    symmetry = consolidated.its_list[0].graph["stereo_coupling_branch"]["bond:2-5"]
    assert symmetry["equivalent_face_branches"] == [0, 1]
    assert symmetry["symmetry_multiplicity"] == 2

    assert len(raw.its_list) == len(raw.smarts_list) == 2
    assert [
        its.graph["stereo_coupling_branch"]["bond:2-5"]["face_branch"]
        for its in raw.its_list
    ] == [0, 1]
    assert [
        its.graph["application_provenance"]["stereo_branch_index"]
        for its in raw.its_list
    ] == [0, 1]


def test_disabling_dedup_does_not_merge_or_drop_enantiomeric_products():
    rule = SynRule.from_smart(
        GENERIC_SYN_HYDROGENATION,
        format="tuple",
        implicit_h=True,
        stereo_couplings={"bond:1-2": "SYN"},
    )
    kwargs = {
        "substrate": "C/C(CC)=C(CC)/C.[H][H]",
        "template": rule,
        "template_format": "tuple",
        "explicit_h": False,
    }
    consolidated = SynReactor(**kwargs)
    raw = SynReactor(**kwargs, dedup_its=False)

    assert len(consolidated.its_list) == len(raw.its_list) == 2
    assert set(consolidated.smarts_list) == set(raw.smarts_list)


def test_raw_aromatic_phase_changing_results_are_fully_refreshed():
    reactants = AROMATIC_CROSS_COUPLING.split(">>", 1)[0]
    template = rsmi_to_its(AROMATIC_CROSS_COUPLING, core=True, format="tuple")
    reactor = SynReactor(
        reactants,
        template,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
    )

    assert reactor.its_list
    assert all(
        its.graph.get("_product_kekule_phase_dirty") is True
        and its.graph.get("_product_electron_fields_current") is True
        for its in reactor.its_list
    )


@pytest.mark.parametrize("explicit_h", [False, True])
@pytest.mark.parametrize(
    ("invert", "substrate"),
    [
        (False, "CC"),
        (True, "C=C.[H][H]"),
    ],
)
def test_raw_mode_preserves_forward_reverse_and_hydrogen_policies(
    explicit_h,
    invert,
    substrate,
):
    reactor = SynReactor(
        substrate,
        ETHANE_DEHYDROGENATION,
        invert=invert,
        template_format="tuple",
        explicit_h=explicit_h,
        dedup_its=False,
    )

    assert reactor.its_list
    assert len(reactor.smarts_list) == len(reactor.its_list)
    assert all("application_provenance" in its.graph for its in reactor.its_list)
    assert all(
        its.graph.get("_product_electron_fields_current") is True
        for its in reactor.its_list
    )


def test_raw_results_and_diagnostics_are_cache_stable_and_aligned():
    reactor = _ethane_reactor(
        automorphism=True,
        dedup_its=False,
        diagnostics=True,
    )
    first_its = reactor.its_list
    first_smarts = reactor.smarts_list
    first_provenance = [its.graph["application_provenance"] for its in first_its]

    assert reactor.its_list is first_its
    assert reactor.smarts_list is first_smarts
    assert [its.graph["application_provenance"] for its in reactor.its_list] == (
        first_provenance
    )
    assert len(reactor.diagnostics) == len(first_its) == len(first_smarts)
    assert [report["index"] for report in reactor.diagnostics] == list(
        range(len(first_its))
    )


def test_raw_smarts_raises_instead_of_silently_losing_its_alignment(monkeypatch):
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
    )
    monkeypatch.setattr(SynReactor, "_to_smarts", staticmethod(lambda its: None))

    with pytest.raises(ValueError, match="raw ITS application"):
        _ = reactor.smarts_list


def test_from_smiles_forwards_policy_and_rejects_non_boolean_values():
    reactor = SynReactor.from_smiles(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
    )
    assert reactor.dedup_its is False
    assert len(reactor.its_list) == 2

    with pytest.raises(TypeError, match="dedup_its must be a bool"):
        SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            explicit_h=False,
            dedup_its="false",
        )


def test_batch_reactor_does_not_expose_synreactor_its_policy():
    assert "dedup_its" not in inspect.signature(BatchReactor).parameters
    batch = BatchReactor(
        ["C=CC=C.C=CC=O"],
        explicit_h=False,
        strategy="bt",
    )
    assert "dedup_its" not in batch.help()
    assert batch.fit([DIELS_ALDER]) == [
        {
            "syn_fw": [DIELS_ALDER],
            "count": 1,
        }
    ]
