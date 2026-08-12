"""Invariant regressions from the FLOWER full-reaction replay."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import (  # noqa: E402
    canonical_unmapped_reaction,
    canonical_unmapped_side,
)
from Experiment.Lewis.rule_replay.benchmark import extract_rule  # noqa: E402
from Experiment.Lewis.rule_replay import benchmark as replay_benchmark  # noqa: E402
from synkit.Synthesis.Reactor.matching_policy import (  # noqa: E402
    contextual_electron_pattern_graph,
    deduplicate_joint_rule_mappings,
)
from synkit.Graph.Matcher.subgraph_matcher import (  # noqa: E402
    electron_aware_node_match,
)
from synkit.Synthesis.Reactor.product_state import (  # noqa: E402
    ProductStatePerceptionError,
    _electron_product_charge,
    _reperceive_product_kekule_phase,
)
from synkit.Synthesis.Reactor import product_state as product_state_module  # noqa: E402
from synkit.Synthesis.Reactor.deduplication import (  # noqa: E402
    _finalize_product_electron_fields,
    _prepare_its_for_structural_cluster,
)
from synkit.IO.graph_to_mol import GraphToMol  # noqa: E402
from synkit.Synthesis.Reactor.syn_reactor import SynReactor  # noqa: E402

SPECTATOR_METAL = (
    "[NH3:1].[CH3:2][Cl:3].[O:4]=[Ag:5]>>" "[NH3+:1][CH3:2].[Cl-:3].[O:4]=[Ag:5]"
)

ISOLATED_HYDROGEN = (
    "[NH2:1][H:4].[CH3:2][Cl:3].[H-:5].[H-:6]>>"
    "[NH2:1][CH3:2].[Cl-:3].[H:4][H:5].[H-:6]"
)

HYDROGEN_ONLY_COMPONENT = (
    "[NH2:1][H:4].[CH3:2][Cl:3].[H-:5].[H:6][H:7]>>"
    "[NH2:1][CH3:2].[Cl-:3].[H:4][H:5].[H:6][H:7]"
)

COPPER_HYDRIDE = (
    "[C:1](=[N:2][H:6])([H:4])[H:5].[Cu:3]([H:7])[H:8]>>"
    "[C:1](=[N+:2]([Cu-:3]([H:7])[H:8])[H:6])([H:4])[H:5]"
)

AROMATIC_AZOLE_ADDITION = (
    "[F:1][C:2]1=[C:3]([H:25])[C:4]([F:5])=[C:6]([H:26])"
    "[C:7]([C:8](=[C:9]([C+:10]([C:11]2=[C:12]([H:30])"
    "[C:13]([F:14])=[C:15]([H:31])[C:16]([F:17])=[C:18]2"
    "[H:32])[H:29])[H:28])[H:27])=[C:19]1[H:33]."
    "[N:20]1([H:34])[C:21]([H:35])=[C:22]([H:36])[N:23]="
    "[C:24]1[H:37]>>[F:1][C:2]1=[C:3]([H:25])[C:4]([F:5])="
    "[C:6]([H:26])[C:7]([C:8](=[C:9]([C:10]([C:11]2=[C:12]"
    "([H:30])[C:13]([F:14])=[C:15]([H:31])[C:16]([F:17])="
    "[C:18]2[H:32])([N+:20]2=[C:24]([H:37])[N:23]([H:34])"
    "[C:22]([H:36])=[C:21]2[H:35])[H:29])[H:28])[H:27])="
    "[C:19]1[H:33]"
)


def _reactor(
    reaction: str,
    *,
    invert: bool = False,
    dedup_its: bool = True,
) -> SynReactor:
    reactants, products = reaction.split(">>", 1)
    host = products if invert else reactants
    return SynReactor(
        canonical_unmapped_side(host),
        extract_rule(reaction, "tuple"),
        invert=invert,
        explicit_h=False,
        implicit_temp=False,
        automorphism=True,
        embed_pre_filter=True,
        template_format="tuple",
        radical_policy="strict",
        stereo_mode="ignore",
        dedup_its=dedup_its,
        serialization_errors="skip",
    )


def _canonical_outputs(reactor: SynReactor) -> set[str]:
    return {canonical_unmapped_reaction(candidate) for candidate in reactor.smarts_list}


def _deferred_metal_its(product_charge: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        1,
        element=("Ag", "Ag"),
        aromatic=(False, False),
        hcount=(0, 0),
        charge=(0, 0),
        radical=(0, 0),
        lone_pairs=(0, 0),
        valence_electrons=(1, 1),
        present=(True, True),
        atom_map=(1, 1),
        template_charge=(0, product_charge),
    )
    graph.graph.update(
        electron_aware_rewrite=True,
        _product_electron_fields_current=False,
        _product_kekule_phase_dirty=True,
    )
    return graph


def _correlated_component_its(*, split_changes: bool) -> nx.Graph:
    """Build a WL-colliding pair with distinct endpoint correlations."""
    graph = nx.Graph()
    component_size = 14
    for component in range(2):
        nodes = [(component, position) for position in range(component_size)]
        graph.add_nodes_from(
            (
                node,
                {
                    "element": ("C", "C"),
                    "aromatic": (False, False),
                    "hcount": (0, 0),
                    "charge": (0, 0),
                    "radical": (0, 0),
                    "lone_pairs": (0, 0),
                    "valence_electrons": (4, 4),
                    "present": (True, True),
                },
            )
            for node in nodes
        )
        graph.add_edges_from(
            (
                left,
                right,
                {
                    "order": (1.0, 1.0),
                    "kekule_order": (1.0, 1.0),
                    "sigma_order": (1.0, 1.0),
                    "pi_order": (0.0, 0.0),
                },
            )
            for left, right in zip(nodes, nodes[1:])
        )

    graph.nodes[0, 0]["charge"] = (0, 1)
    hydrogen_component = 1 if split_changes else 0
    graph.nodes[hydrogen_component, component_size - 1]["hcount"] = (0, 1)
    graph.graph["_product_electron_fields_current"] = True
    return graph


def _long_chain_its(marker_position: int) -> nx.Graph:
    """Build a positional isomer hidden from a three-round WL hash."""
    graph = nx.path_graph(48)
    for node in graph:
        graph.nodes[node].update(
            element=("C", "C"),
            aromatic=(False, False),
            hcount=(0, int(node == marker_position)),
            charge=(0, 0),
            radical=(0, 0),
            lone_pairs=(0, 0),
            valence_electrons=(4, 4),
            present=(True, True),
        )
    for _, _, attrs in graph.edges(data=True):
        attrs.update(
            order=(1.0, 1.0),
            kekule_order=(1.0, 1.0),
            sigma_order=(1.0, 1.0),
            pi_order=(0.0, 0.0),
        )
    graph.graph["_product_electron_fields_current"] = True
    return graph


def test_unchanged_spectator_charge_is_locality_invariant() -> None:
    expected = canonical_unmapped_reaction(SPECTATOR_METAL)

    for invert in (False, True):
        reactor = _reactor(SPECTATOR_METAL, invert=invert)
        assert expected in _canonical_outputs(reactor)


@pytest.mark.parametrize(
    "reaction",
    [ISOLATED_HYDROGEN, HYDROGEN_ONLY_COMPONENT],
)
def test_hydrogen_only_component_multiplicity_is_conserved(
    reaction: str,
) -> None:
    expected = canonical_unmapped_reaction(reaction)

    for invert in (False, True):
        reactor = _reactor(reaction, invert=invert)
        assert expected in _canonical_outputs(reactor)


def test_hidden_metal_hydrogen_context_does_not_block_matching() -> None:
    expected = canonical_unmapped_reaction(COPPER_HYDRIDE)

    for invert in (False, True):
        reactor = _reactor(COPPER_HYDRIDE, invert=invert)
        assert expected in _canonical_outputs(reactor)


def test_dirty_aromatic_rewrite_discards_the_stale_kekule_phase() -> None:
    expected = canonical_unmapped_reaction(AROMATIC_AZOLE_ADDITION)

    for invert in (False, True):
        reactor = _reactor(AROMATIC_AZOLE_ADDITION, invert=invert)
        assert expected in _canonical_outputs(reactor)


def test_endpoint_equivalence_is_component_permutation_invariant() -> None:
    first = "C.O>>CC.N"
    permuted = "O.C>>N.CC"

    assert canonical_unmapped_reaction(first) == canonical_unmapped_reaction(permuted)


def test_endpoint_equivalence_rejects_invalid_components() -> None:
    with pytest.raises(ValueError):
        canonical_unmapped_reaction("C.notasmiles>>C")


def test_query_uncertainty_is_typed_and_contextual() -> None:
    pattern = nx.Graph()
    pattern.add_node(
        1,
        element="Cu",
        neighbors=["H", "H"],
        radical=1,
    )
    unchanged_rc = nx.Graph()
    unchanged_rc.add_node(1, radical=(1, 1))
    changed_rc = nx.Graph()
    changed_rc.add_node(1, radical=(1, 0))

    contextual = contextual_electron_pattern_graph(
        pattern,
        reaction_center=unchanged_rc,
        template_format="tuple",
    )
    exact = contextual_electron_pattern_graph(
        pattern,
        reaction_center=changed_rc,
        template_format="tuple",
    )

    assert contextual.nodes[1]["_query_attribute_policies"] == {"radical": "unknown"}
    assert electron_aware_node_match(
        {"element": "Cu", "radical": 0},
        contextual.nodes[1],
        ("element", "radical"),
    )
    assert "_query_attribute_policies" not in exact.nodes[1]
    assert not electron_aware_node_match(
        {"element": "Cu", "radical": 0},
        exact.nodes[1],
        ("element", "radical"),
    )


def test_dirty_aromatic_reperception_fails_closed(monkeypatch) -> None:
    product = nx.Graph()
    product.add_nodes_from([(1, {"element": "C"}), (2, {"element": "C"})])
    product.add_edge(1, 2, order=1.5)
    its = product.copy()
    its.graph["_product_kekule_phase_dirty"] = True

    call = {}

    def reject(*args, **kwargs):
        call.update(kwargs)
        raise ValueError("invalid aromatic product")

    monkeypatch.setattr(GraphToMol, "graph_to_mol", reject)
    with pytest.raises(ProductStatePerceptionError):
        _reperceive_product_kekule_phase(product, its)
    assert call["prefer_kekule_order"] is False


def test_dirty_aromatic_reperception_does_not_swallow_replay_timeout(
    monkeypatch,
) -> None:
    product = nx.Graph()
    product.add_nodes_from([(1, {"element": "C"}), (2, {"element": "C"})])
    product.add_edge(1, 2, order=1.5)
    its = product.copy()
    its.graph["_product_kekule_phase_dirty"] = True

    def timeout(*_args, **_kwargs):
        raise replay_benchmark.CaseTimeout("alarm")

    monkeypatch.setattr(GraphToMol, "graph_to_mol", timeout)
    with pytest.raises(replay_benchmark.CaseTimeout):
        _reperceive_product_kekule_phase(product, its)


def test_unperceivable_candidate_is_rejected_without_losing_valid_ones(
    monkeypatch,
) -> None:
    rejected = nx.Graph()
    rejected.graph.update(
        electron_aware_rewrite=True,
        _product_electron_fields_current=False,
        reject=True,
    )
    valid = nx.Graph()
    valid.graph.update(
        electron_aware_rewrite=True,
        _product_electron_fields_current=False,
    )

    def refresh(candidate):
        if candidate.graph.get("reject"):
            raise ProductStatePerceptionError("invalid candidate")
        candidate.graph["_product_electron_fields_current"] = True

    monkeypatch.setattr(
        product_state_module,
        "_refresh_product_electron_fields",
        refresh,
    )

    assert _finalize_product_electron_fields([rejected, valid]) == [valid]


def test_deduplication_is_congruent_with_final_endpoint_state() -> None:
    neutral = _deferred_metal_its(0)
    cationic = _deferred_metal_its(1)

    unique = SynReactor._deduplicate_structural_its([neutral, cationic])

    assert len(unique) == 2
    assert [graph.nodes[1]["charge"][1] for graph in unique] == [0, 1]


def test_charge_authority_depends_on_model_consistency_not_element() -> None:
    def charge_its(element: str, reactant_charge: int) -> nx.Graph:
        graph = nx.Graph()
        graph.add_node(
            1,
            element=(element, element),
            aromatic=(False, False),
            hcount=(0, 0),
            charge=(reactant_charge, reactant_charge),
            radical=(0, 0),
            lone_pairs=(0, 0),
            valence_electrons=((1 if element == "Cu" else 4),) * 2,
            present=(True, True),
            template_charge=(reactant_charge, 9),
        )
        return graph

    consistent_metal = charge_its("Cu", 1)
    inconsistent_main_group = charge_its("C", 0)
    product_state = {
        "element": "ignored",
        "aromatic": False,
        "recomputed_charge": 2,
    }

    assert _electron_product_charge(consistent_metal, 1, product_state) == 2
    assert _electron_product_charge(inconsistent_main_group, 1, product_state) == 9

    inconsistent_destination = charge_its("Cu", 1)
    inconsistent_destination.nodes[1]["charge_model_consistent"] = (True, False)
    assert _electron_product_charge(inconsistent_destination, 1, product_state) == 9


def test_deduplication_preserves_long_range_component_correlations() -> None:
    coupled = _correlated_component_its(split_changes=False)
    split = _correlated_component_its(split_changes=True)

    unique = SynReactor._deduplicate_structural_its([coupled, split])

    assert len(unique) == 2


def test_long_chain_collision_is_refined_before_exact_vf2(monkeypatch) -> None:
    first = _long_chain_its(18)
    second = _long_chain_its(19)
    prepared = [
        _prepare_its_for_structural_cluster(graph, refresh_electrons=False)
        for graph in (first, second)
    ]
    initial_hashes = [
        nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="_its_node_sig",
            edge_attr="_its_edge_sig",
            iterations=3,
            digest_size=16,
        )
        for graph in prepared
    ]
    assert initial_hashes[0] == initial_hashes[1]

    def unexpected_vf2(*_args, **_kwargs):
        raise AssertionError("strong component invariant should split this collision")

    monkeypatch.setattr(nx, "is_isomorphic", unexpected_vf2)
    assert len(SynReactor._deduplicate_structural_its([first, second])) == 2


def test_disconnected_rule_does_not_use_nodewise_host_orbits() -> None:
    size = 18
    reactant = "".join(
        f"[CH2:{atom_map}]{'1' if atom_map == 1 else ''}"
        for atom_map in range(1, size + 1)
    )
    product = "".join(
        (
            f"[CH+:{atom_map}]1"
            if atom_map == 1
            else f"[CH-:{atom_map}]" if atom_map == 10 else f"[CH2:{atom_map}]"
        )
        for atom_map in range(1, size + 1)
    )
    reactor = _reactor(f"{reactant}1>>{product}1")

    assert reactor.mapping_count == size * (size - 1)


def test_joint_rule_symmetry_leaves_partial_mappings_unquotiented() -> None:
    pattern = nx.Graph([(1, 2)])
    mappings = [{1: "left"}, {1: "right"}]

    result = deduplicate_joint_rule_mappings(
        mappings,
        pattern,
        nx.Graph(),
        node_attrs=[],
        edge_attrs=[],
    )

    assert result is mappings


def test_joint_rule_symmetry_uses_the_simultaneous_group_action() -> None:
    pattern = nx.cycle_graph(4)
    nx.set_node_attributes(pattern, "C", "element")
    nx.set_node_attributes(pattern, 0, "charge")
    nx.set_edge_attributes(pattern, 1.0, "order")
    reaction_center = pattern.copy()
    nx.set_edge_attributes(reaction_center, (1.0, 0.0), "order")
    mappings = [
        dict(zip(pattern.nodes, permutation))
        for permutation in itertools.permutations((10, 11, 12, 13))
    ]

    result = deduplicate_joint_rule_mappings(
        mappings,
        pattern,
        reaction_center,
        node_attrs=["element", "charge"],
        edge_attrs=["order"],
        fixed_nodes=frozenset(),
    )

    # C4 has eight automorphisms, so its action has three orbits on the 24
    # bijections. A vertex-orbit bag would incorrectly collapse all 24.
    assert len(result) == 3


def test_joint_rule_symmetry_preserves_node_transition_roles() -> None:
    pattern = nx.path_graph(3)
    nx.set_node_attributes(pattern, "C", "element")
    nx.set_node_attributes(pattern, 0, "charge")
    nx.set_edge_attributes(pattern, 1.0, "order")
    reaction_center = pattern.copy()
    reaction_center.nodes[0]["charge"] = (0, 1)
    reaction_center.nodes[1]["charge"] = (0, 0)
    reaction_center.nodes[2]["charge"] = (0, 0)
    mappings = [
        {0: 10, 1: 11, 2: 12},
        {0: 12, 1: 11, 2: 10},
    ]

    result = deduplicate_joint_rule_mappings(
        mappings,
        pattern,
        reaction_center,
        node_attrs=["element", "charge"],
        edge_attrs=["order"],
        fixed_nodes=frozenset(),
    )

    assert result == mappings


def test_replay_timeout_excludes_evidence_postprocessing(monkeypatch) -> None:
    timer_calls = []

    class FakeReactor:
        mappings = [{1: 1}]
        its_list = [nx.Graph()]

        @property
        def smarts_list(self):
            assert timer_calls[-1] == 0.0
            return ["C>>C"]

    monkeypatch.setattr(replay_benchmark, "make_reactor", lambda *_args: FakeReactor())
    monkeypatch.setattr(
        replay_benchmark,
        "_set_timeout",
        timer_calls.append,
    )
    monkeypatch.setattr(
        replay_benchmark,
        "canonical_unmapped_reaction",
        lambda reaction: reaction,
    )

    result = replay_benchmark.replay_direction(
        host="C",
        expected="C>>C",
        rule=object(),
        representation="tuple",
        direction="forward",
        embedding_threshold=None,
        case_timeout=30.0,
    )

    assert result["status"] == "PASS"
    assert result["expansion_seconds"] <= result["seconds"]
    assert set(result["stage_seconds"]) == {
        "reactor_construction",
        "matching",
        "rewriting",
        "serialization",
        "canonicalization",
    }


def test_replay_timeout_is_reported_at_the_replay_boundary(monkeypatch) -> None:
    class FakeReactor:
        mappings = [{1: 1}]

        @property
        def its_list(self):
            raise replay_benchmark.CaseTimeout("alarm")

    monkeypatch.setattr(replay_benchmark, "make_reactor", lambda *_args: FakeReactor())
    monkeypatch.setattr(replay_benchmark, "_set_timeout", lambda _seconds: None)

    result = replay_benchmark.replay_direction(
        host="C",
        expected="C>>C",
        rule=object(),
        representation="tuple",
        direction="forward",
        embedding_threshold=None,
        case_timeout=30.0,
    )

    assert result["status"] == "ERROR"
    assert result["stage"] == "rewriting"
    assert result["error_type"] == "CaseTimeout"
    assert result["mapping_count"] == 1


def test_replay_runs_when_interval_timers_are_unavailable(monkeypatch) -> None:
    class FakeReactor:
        mappings = [{1: 1}]
        its_list = [nx.Graph()]
        smarts_list = ["C>>C"]

    monkeypatch.setattr(replay_benchmark, "make_reactor", lambda *_args: FakeReactor())
    monkeypatch.setattr(
        replay_benchmark,
        "canonical_unmapped_reaction",
        lambda reaction: reaction,
    )
    monkeypatch.setattr(
        replay_benchmark,
        "_supports_interval_timer",
        lambda: False,
    )

    result = replay_benchmark.replay_direction(
        host="C",
        expected="C>>C",
        rule=object(),
        representation="tuple",
        direction="forward",
        embedding_threshold=None,
        case_timeout=30.0,
    )

    assert result["status"] == "PASS"
