import inspect
import warnings

import networkx as nx
import pytest

from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import (
    BatchReactor,
    RawITSApplicationSerializationWarning,
    SynReactor,
)
from synkit.Synthesis.Reactor.core import product as reactor_product_state
from synkit.Synthesis.Reactor.output import serialization as reactor_serialization
from synkit.Synthesis.Reactor.output import structural as reactor_structural
from synkit.Synthesis.Reactor.workflow import batch as batch_reactor_module
from synkit.Synthesis.Reactor.workflow.batch import _RuleApplier

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


def test_deferred_product_dedup_finalizes_without_structural_quotient(
    monkeypatch,
) -> None:
    def unexpected_application_quotient(_mappings, _host, _reaction_center):
        raise AssertionError("deferred mode must not run application quotienting")

    def unexpected_structural_quotient(_graphs):
        raise AssertionError("deferred mode must not run structural ITS quotienting")

    monkeypatch.setattr(
        SynReactor,
        "_deduplicate_rewrite_equivalent_mappings",
        staticmethod(unexpected_application_quotient),
    )
    monkeypatch.setattr(
        SynReactor,
        "_deduplicate_structural_its",
        staticmethod(unexpected_structural_quotient),
    )
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        automorphism=False,
        stereo_mode="ignore",
        product_deduplication="deferred",
    )

    assert reactor.its_list
    assert all(
        graph.graph.get("_product_electron_fields_current")
        for graph in reactor.its_list
    )
    assert reactor.smarts_list


@pytest.mark.parametrize(
    ("substrate", "template", "explicit_h_rule", "expect_cache_hit"),
    [
        ("C=CC=C.C=CC=O", DIELS_ALDER, False, True),
        ("CC", ETHANE_DEHYDROGENATION, True, False),
    ],
)
def test_reactant_serialization_cache_respects_explicit_h_applications(
    monkeypatch,
    substrate,
    template,
    explicit_h_rule,
    expect_cache_hit,
) -> None:
    original = reactor_serialization._to_smarts
    cached_reactants = []

    def observed(
        graph,
        *,
        reactant_smiles=None,
        preserved_hydrogen_maps=None,
    ):
        cached_reactants.append(reactant_smiles)
        return original(
            graph,
            reactant_smiles=reactant_smiles,
            preserved_hydrogen_maps=preserved_hydrogen_maps,
        )

    monkeypatch.setattr(reactor_serialization, "_to_smarts", observed)
    reactor = SynReactor(
        substrate,
        template,
        template_format="tuple",
        explicit_h=False,
        automorphism=False,
        stereo_mode="ignore",
        product_deduplication="deferred",
    )

    assert reactor.smarts_list
    assert reactor._flag_pattern_has_explicit_H is explicit_h_rule
    assert any(value is not None for value in cached_reactants) is expect_cache_hit


def test_deferred_product_dedup_requires_ignored_stereo() -> None:
    with pytest.raises(ValueError, match="requires stereo_mode='ignore'"):
        SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            template_format="tuple",
            explicit_h=False,
            product_deduplication="deferred",
        )


def test_product_deduplication_policy_is_validated() -> None:
    with pytest.raises(ValueError, match="product_deduplication"):
        SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            template_format="tuple",
            explicit_h=False,
            product_deduplication="unknown",
        )


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


def test_rewrite_locus_refresh_matches_conservative_whole_graph_refresh():
    reactants = AROMATIC_CROSS_COUPLING.split(">>", 1)[0]
    reactor = SynReactor(
        reactants,
        AROMATIC_CROSS_COUPLING,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="ignore",
        product_deduplication="deferred",
    )

    assert reactor.its_list
    for localized in reactor.its_list:
        assert localized.graph.get("_product_refresh_nodes")
        conservative = localized.copy()
        conservative.graph.pop("_product_refresh_nodes")
        conservative.graph["_product_electron_fields_current"] = False
        reactor_product_state._refresh_product_electron_fields(conservative)
        assert reactor_serialization._to_smarts(localized) == (
            reactor_serialization._to_smarts(conservative)
        )


def test_electron_refresh_support_contains_presence_change_boundary():
    graph = nx.Graph()
    graph.add_node(
        1,
        element=("C", None),
        present=(True, False),
        hcount=(3, 3),
        charge=(0, 0),
        radical=(0, 0),
        lone_pairs=(0, 0),
        valence_electrons=(4, 4),
    )
    graph.add_node(
        2,
        element=("C", "C"),
        present=(True, True),
        hcount=(3, 3),
        charge=(0, 0),
        radical=(0, 0),
        lone_pairs=(0, 0),
        valence_electrons=(4, 4),
    )
    graph.add_edge(
        1,
        2,
        order=(1.0, 1.0),
        kekule_order=(1.0, 1.0),
        sigma_order=(1.0, 1.0),
        pi_order=(0.0, 0.0),
    )
    graph.graph.update(
        electron_aware_rewrite=True,
        _product_kekule_phase_dirty=False,
        _product_electron_fields_current=False,
    )

    support = reactor_product_state._electron_refresh_support(graph, {1})
    assert support == frozenset({1, 2})

    localized = graph.copy()
    localized.graph["_product_refresh_nodes"] = support
    conservative = graph.copy()
    reactor_product_state._refresh_product_electron_fields(localized)
    reactor_product_state._refresh_product_electron_fields(conservative)
    assert localized.nodes[2]["recomputed_charge"] == (
        conservative.nodes[2]["recomputed_charge"]
    )
    assert localized.nodes[2]["charge"] == conservative.nodes[2]["charge"]


def test_product_only_tuple_projection_matches_joint_projection():
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        stereo_mode="ignore",
        product_deduplication="deferred",
    )
    graph = reactor.its_list[0]

    joint_product = reactor_serialization._tuple_endpoint_graphs(graph)[1]
    product_only = reactor_serialization._tuple_endpoint_graphs(
        graph,
        sides=(1,),
    )[0]

    assert nx.utils.graphs_equal(joint_product, product_only)


def test_exact_structural_quotient_does_not_conflate_container_types():
    tuple_graph = nx.Graph()
    tuple_graph.add_node(
        1,
        element="C",
        aromatic=False,
        hcount=4,
        charge=0,
        radical=0,
        lone_pairs=0,
        valence_electrons=4,
        present=True,
        typesGH=(("C", False, 4, 0, []), ("C", False, 4, 0, [])),
    )
    list_graph = tuple_graph.copy()
    list_graph.nodes[1]["typesGH"] = list(tuple_graph.nodes[1]["typesGH"])

    unique = SynReactor._deduplicate_structural_its([tuple_graph, list_graph])

    assert unique == [tuple_graph, list_graph]
    for graph in (tuple_graph, list_graph):
        assert all(
            not key.startswith("_structural_")
            for _, attrs in graph.nodes(data=True)
            for key in attrs
        )


def test_canonical_certificate_resolves_a_regular_one_wl_collision():
    cycle = nx.cycle_graph(6)
    triangles = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    relabelled_cycle = nx.relabel_nodes(
        cycle,
        {node: f"v{(5 - node) % 6}" for node in cycle},
    )
    graphs = (cycle, triangles, relabelled_cycle)
    for graph in graphs:
        for _, attrs in graph.nodes(data=True):
            attrs.update(
                element=("C", "C"),
                aromatic=(False, False),
                hcount=(2, 2),
                charge=(0, 0),
                radical=(0, 0),
                lone_pairs=(0, 0),
                valence_electrons=(4, 4),
                present=(True, True),
            )
        for _, _, attrs in graph.edges(data=True):
            attrs.update(
                order=(1, 1),
                kekule_order=(1, 1),
                sigma_order=(1, 1),
                pi_order=(0, 0),
            )
        reactor_structural._attach_exact_structural_signatures(graph)

    reactor_structural._refine_structural_colours_in_place(
        list(graphs),
        iterations=1,
    )
    wl_colours = [
        tuple(
            sorted(
                attrs[reactor_structural._REFINED_NODE_COLOUR]
                for _, attrs in graph.nodes(data=True)
            )
        )
        for graph in graphs
    ]
    assert wl_colours[0] == wl_colours[1]

    node_palette = {}
    edge_palette = {}
    certificates = [
        reactor_structural._canonical_attributed_graph_certificate(
            graph,
            node_attribute=reactor_structural._EXACT_NODE_SIG,
            edge_attribute=reactor_structural._EXACT_EDGE_SIG,
            node_palette=node_palette,
            edge_palette=edge_palette,
        )
        for graph in graphs
    ]
    assert certificates[0] != certificates[1]
    assert certificates[0] == certificates[2]


def test_incremental_exact_labels_equal_complete_recomputation():
    base = nx.path_graph(4)
    for _, attrs in base.nodes(data=True):
        attrs.update(
            element=("C", "C"),
            aromatic=(False, False),
            hcount=(2, 2),
            charge=(0, 0),
            radical=(0, 0),
            lone_pairs=(0, 0),
            valence_electrons=(4, 4),
            present=(True, True),
        )
    for _, _, attrs in base.edges(data=True):
        attrs.update(
            order=(1, 1),
            kekule_order=(1, 1),
            sigma_order=(1, 1),
            pi_order=(0, 0),
        )

    node_palette = {}
    edge_palette = {}
    identity_cache = {}
    base.graph[reactor_structural._EXACT_NODE_PALETTE] = node_palette
    base.graph[reactor_structural._EXACT_EDGE_PALETTE] = edge_palette
    base.graph[reactor_structural._EXACT_IDENTITY_CACHE] = identity_cache
    reactor_structural._attach_exact_structural_signatures(base)

    incremental = base.copy()
    complete = base.copy()
    for graph in (incremental, complete):
        graph.nodes[1]["charge"] = (0, 1)
        graph.nodes[1].pop(reactor_structural._EXACT_NODE_SIG)
        graph.edges[1, 2]["order"] = (1, 2)
        graph.edges[1, 2].pop(reactor_structural._EXACT_EDGE_SIG)

    incremental.graph["_structural_signatures_seeded"] = True
    incremental.graph[reactor_structural._EXACT_DIRTY_NODES] = {1}
    incremental.graph[reactor_structural._EXACT_DIRTY_EDGES] = {(1, 2)}
    reactor_structural._attach_exact_structural_signatures(incremental)

    for _, attrs in complete.nodes(data=True):
        attrs.pop(reactor_structural._EXACT_NODE_SIG, None)
    for _, _, attrs in complete.edges(data=True):
        attrs.pop(reactor_structural._EXACT_EDGE_SIG, None)
    reactor_structural._attach_exact_structural_signatures(complete)

    for node in base:
        assert (
            incremental.nodes[node][reactor_structural._EXACT_NODE_SIG]
            == complete.nodes[node][reactor_structural._EXACT_NODE_SIG]
        )
    for left, right in base.edges():
        assert (
            incremental.edges[left, right][reactor_structural._EXACT_EDGE_SIG]
            == complete.edges[left, right][reactor_structural._EXACT_EDGE_SIG]
        )


def test_active_neighbourhood_is_invariant_under_node_relabelling():
    graph = nx.cycle_graph(8)
    for _, attrs in graph.nodes(data=True):
        attrs.update(
            element=("C", "C"),
            aromatic=(False, False),
            hcount=(2, 2),
            charge=(0, 0),
            radical=(0, 0),
            lone_pairs=(0, 0),
            valence_electrons=(4, 4),
            present=(True, True),
        )
    for _, _, attrs in graph.edges(data=True):
        attrs.update(
            order=(1, 1),
            kekule_order=(1, 1),
            sigma_order=(1, 1),
            pi_order=(0, 0),
        )
    graph.edges[0, 1]["order"] = (1, 2)
    relabelled = nx.relabel_nodes(
        graph,
        {node: f"atom-{(3 * node + 1) % 8}" for node in graph},
    )

    node_palette = {}
    edge_palette = {}
    for candidate in (graph, relabelled):
        candidate.graph[reactor_structural._EXACT_NODE_PALETTE] = node_palette
        candidate.graph[reactor_structural._EXACT_EDGE_PALETTE] = edge_palette
        candidate.graph[reactor_structural._EXACT_IDENTITY_CACHE] = {}
        reactor_structural._attach_exact_structural_signatures(candidate)

    assert reactor_structural._active_neighbourhood_invariant(
        graph
    ) == reactor_structural._active_neighbourhood_invariant(relabelled)


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


def _install_serialization_results(monkeypatch, reactor, failed_indices):
    graphs = reactor.its_list
    by_graph = {
        id(graph): None if index in failed_indices else f"raw-{index}"
        for index, graph in enumerate(graphs)
    }
    monkeypatch.setattr(
        SynReactor,
        "_to_smarts",
        staticmethod(lambda graph: by_graph[id(graph)]),
    )
    return [
        f"raw-{index}" for index in range(len(graphs)) if index not in failed_indices
    ]


def test_raw_serialization_skip_preserves_valid_order_and_reports_all_indices(
    monkeypatch,
):
    reactor = SynReactor(
        "CC",
        ETHANE_DEHYDROGENATION,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
        serialization_errors="skip",
    )
    failed = (1, 4, 17)
    expected = _install_serialization_results(monkeypatch, reactor, failed)

    with pytest.warns(RawITSApplicationSerializationWarning) as caught:
        assert reactor.smarts_list == expected

    assert len(caught) == 1
    assert caught[0].message.indices == failed
    assert reactor.serialization_failure_indices == failed
    assert len(reactor.its_list) == 18

    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter("always")
        assert reactor.smarts_list is reactor._smarts
    assert repeated == []


def test_raw_serialization_skip_returns_empty_for_all_invalid(monkeypatch):
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        automorphism=False,
        dedup_its=False,
        serialization_errors="skip",
    )
    failed = tuple(range(len(reactor.its_list)))
    _install_serialization_results(monkeypatch, reactor, failed)

    with pytest.warns(RawITSApplicationSerializationWarning) as caught:
        assert reactor.smarts_list == []

    assert len(caught) == 1
    assert caught[0].message.indices == failed
    assert reactor.serialization_failure_indices == failed


def test_raw_serialization_skip_emits_no_warning_without_failures(monkeypatch):
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        automorphism=False,
        dedup_its=False,
        serialization_errors="skip",
    )
    expected = _install_serialization_results(monkeypatch, reactor, ())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert reactor.smarts_list == expected

    assert caught == []
    assert reactor.serialization_failure_indices == ()


def test_raw_serialization_raise_default_is_retryable_and_diagnostic(monkeypatch):
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        automorphism=False,
        dedup_its=False,
    )
    _install_serialization_results(monkeypatch, reactor, (0,))

    for _ in range(2):
        with pytest.raises(
            ValueError,
            match=r"Could not serialize raw ITS application\(s\): 0",
        ):
            _ = reactor.smarts_list
        assert reactor._smarts is None
        assert reactor.serialization_failure_indices == (0,)


def test_raw_serialization_skip_does_not_catch_programming_errors(monkeypatch):
    reactor = SynReactor(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
        serialization_errors="skip",
    )

    def fail_serialization(_graph):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(SynReactor, "_to_smarts", staticmethod(fail_serialization))
    with pytest.raises(RuntimeError, match="unrelated failure"):
        _ = reactor.smarts_list


def test_consolidated_serialization_is_unchanged_by_raw_skip_policy():
    kwargs = {
        "substrate": "C=CC=C.C=CC=O",
        "template": DIELS_ALDER,
        "template_format": "tuple",
        "explicit_h": False,
    }
    baseline = SynReactor(**kwargs)
    opted_in = SynReactor(**kwargs, serialization_errors="skip")

    assert opted_in.smarts_list == baseline.smarts_list
    assert opted_in.serialization_failure_indices == ()


def test_from_smiles_forwards_policy_and_rejects_non_boolean_values():
    reactor = SynReactor.from_smiles(
        "C=CC=C.C=CC=O",
        DIELS_ALDER,
        template_format="tuple",
        explicit_h=False,
        dedup_its=False,
        serialization_errors="skip",
    )
    assert reactor.dedup_its is False
    assert reactor.serialization_errors == "skip"
    assert len(reactor.its_list) == 2

    with pytest.raises(TypeError, match="dedup_its must be a bool"):
        SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            explicit_h=False,
            dedup_its="false",
        )

    with pytest.raises(ValueError, match="serialization_errors"):
        SynReactor(
            "CC",
            ETHANE_DEHYDROGENATION,
            explicit_h=False,
            serialization_errors="ignore",
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


def test_batch_rule_cache_observes_graph_mutation(monkeypatch):
    calls = []

    def fake_apply(substrate, rule, invert, **kwargs):
        calls.append((len(substrate), len(rule), invert))
        return [f"{len(substrate)}:{len(rule)}:{invert}"]

    monkeypatch.setattr(batch_reactor_module, "_apply_rule_raw", fake_apply)
    applier = _RuleApplier(
        strategy="all",
        explicit_h=False,
        implicit_temp=False,
        cache_enabled=True,
        cache_maxsize=4,
    )
    substrate = nx.Graph()
    substrate.add_node(0, element="C")
    rule = nx.Graph()
    rule.add_node(0, element="C")

    assert applier(substrate, rule, False) == ["1:1:False"]
    assert applier(substrate, rule, False) == ["1:1:False"]
    substrate.add_node(1, element="O")
    assert applier(substrate, rule, False) == ["2:1:False"]
    assert len(calls) == 2


def test_zero_sized_batch_rule_cache_is_safely_disabled(monkeypatch):
    calls = []

    def fake_apply(substrate, rule, invert, **kwargs):
        calls.append(None)
        return ["result"]

    monkeypatch.setattr(batch_reactor_module, "_apply_rule_raw", fake_apply)
    applier = _RuleApplier(
        strategy="all",
        explicit_h=False,
        implicit_temp=False,
        cache_enabled=True,
        cache_maxsize=0,
    )
    graph = nx.Graph()

    assert applier(graph, graph, False) == ["result"]
    assert applier(graph, graph, False) == ["result"]
    assert len(calls) == 2
