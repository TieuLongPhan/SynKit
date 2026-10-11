"""Two-sided pruning must cover every exact reactant/product orbit."""

from collections import Counter
from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def automorphisms(matrix):
    return [
        p
        for p in permutations(range(len(matrix)))
        if np.array_equal(matrix, matrix[np.ix_(p, p)])
    ]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("native", [False, True])
def test_two_sided_candidates_cover_brute_force_orbits(seed, native, request):
    rng = np.random.default_rng(seed)
    a = nx.to_numpy_array(nx.star_graph(4))
    b = rng.choice([0.0, 1.0, 1.5], size=(5, 5))
    b = np.triu(b, 1)
    b += b.T
    if seed == 0:
        b = a.copy()
    elif seed == 1:
        b = nx.to_numpy_array(nx.cycle_graph(5))
    rgroup, pgroup = automorphisms(a), automorphisms(b)
    costs = {
        p: float(np.abs(a - b[np.ix_(p, p)]).sum() / 2) for p in permutations(range(5))
    }

    def orbit_key(mapping):
        return min(
            tuple(p[mapping[r[i]]] for i in range(5)) for r in rgroup for p in pgroup
        )

    for target in sorted(set(costs.values())):
        result = enumerate_distance_mappings(
            (graph(a), graph(b)),
            CD=target,
            binary=False,
            max_bijections=None,
            compute_minimum_cost=False,
            symmetry_pruning=True,
        )
        if native:
            from synkit.Chem.Mapper.exact.native_candidates import (
                enumerate_native_candidates,
            )

            result = enumerate_native_candidates(
                (graph(a), graph(b)),
                target,
                library_path=request.getfixturevalue("native_library"),
                node_properties=(),
                time_limit_seconds=10,
            )
            assert result["complete"]
            assert {orbit_key(m) for m in result["mappings"]} == {
                orbit_key(m) for m, cost in costs.items() if cost == target
            }
            assert all(costs[m] == target for m in result["mappings"])
            continue
        assert result.complete
        assert {orbit_key(m) for m in result.mappings} == {
            orbit_key(m) for m, cost in costs.items() if cost == target
        }
        assert result.selected_labeled_mapping_count is not None


def test_native_canonical_codes_and_groups_equal_exhaustive_oracles(native_library):
    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

    for candidate in nx.graph_atlas_g():
        if len(candidate) > 5:
            break
        if not len(candidate):
            continue
        nx.set_node_attributes(candidate, ("atom", 6), "color")
        nx.set_edge_attributes(candidate, ("bond", 1.5), "color")
        expected = ExactColoredGraphCanonicalizer(candidate).canonicalize()
        code, order = native_canonical_code(candidate, library_path=native_library)
        assert code == expected.canonical_code
        assert order == len(expected.automorphisms)
        renamed = nx.relabel_nodes(candidate, {i: f"node:{10 - i}" for i in candidate})
        other, other_order = native_canonical_code(renamed, library_path=native_library)
        assert (other, other_order) == (code, order)


def test_native_colored_canonicalization_preserves_loops_and_budgets(native_library):
    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

    rng = np.random.default_rng(419)
    for _ in range(20):
        graph = nx.Graph()
        graph.add_nodes_from(
            (i, {"color": ("atom", int(rng.integers(2)))}) for i in range(5)
        )
        for i in range(5):
            for j in range(i, 5):
                if rng.integers(2):
                    graph.add_edge(i, j, color=("bond", float(rng.choice([0.5, 1, 2]))))
        expected = ExactColoredGraphCanonicalizer(graph).canonicalize()
        actual, order = native_canonical_code(graph, library_path=native_library)
        assert actual == expected.canonical_code
        assert order == len(expected.automorphisms)
    with pytest.raises(RuntimeError, match="incomplete"):
        native_canonical_code(graph, library_path=native_library, timeout_seconds=0)


@pytest.mark.parametrize("seed", range(6))
def test_weighted_double_orbits_equal_labeled_enumeration(native_library, seed):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates
    from synkit.Chem.Mapper.exact.native_canonical import compact_canonical_code
    from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator
    from synkit.Chem.Mapper.exact.symmetry import permutation_group_order
    from synkit.Chem.Mapper.graph.automorphism import bounded_automorphism_permutations
    from synkit.Chem.Mapper.spectrum import exact_its_and_template_codes

    rng = np.random.default_rng(seed)
    a = nx.to_numpy_array(nx.star_graph(4))
    b = nx.to_numpy_array(nx.cycle_graph(5)) if seed < 3 else a.copy()
    if seed % 3 == 1:
        a *= 0.5
    if seed % 3 == 2:
        b *= 1.5
    pair = (graph(a), graph(b))
    rg, _ = bounded_automorphism_permutations(pair[0], binary=False)
    pg, _ = bounded_automorphism_permutations(pair[1], binary=False)
    ro, po = permutation_group_order(rg[1:]), permutation_group_order(pg[1:])
    costs = {
        m: float(np.abs(a - b[np.ix_(m, m)]).sum() / 2) for m in permutations(range(5))
    }
    for target in sorted(set(costs.values())):
        observer = _BlindShellObserver(
            a,
            b,
            [6] * 5,
            {},
            GlobalShellConfig(
                symmetry_node_properties=(),
                reaction_center_properties=(),
                structure_timeout_seconds=1,
            ),
        )
        accumulator = OrbitAccumulator(
            observer, rg[1:], ro, po, library_path=native_library
        )
        result = enumerate_native_candidates(
            pair,
            target,
            library_path=native_library,
            node_properties=(),
            callback=accumulator.observe,
            time_limit_seconds=10,
        )
        assert result["complete"]
        accumulator.finish()
        accumulator.finish()
        selected = [m for m, cost in costs.items() if cost == target]
        assert observer.count * po == len(selected)
        expected_bonds = Counter()
        expected_its = Counter()
        expected_templates = Counter()
        for mapping in selected:
            transformed = b[np.ix_(mapping, mapping)]
            expected_bonds.update(
                (i, j)
                for i in range(5)
                for j in range(i + 1, 5)
                if a[i, j] != transformed[i, j]
            )
            its, template, reason = exact_its_and_template_codes(
                a, b, [6] * 5, {}, mapping, timeout_seconds=1
            )
            assert reason is None
            expected_its[its] += 1
            expected_templates[template] += 1
        assert {k: v * po for k, v in observer.bond_counts.items() if v} == dict(
            expected_bonds
        )
        assert {
            compact_canonical_code(k): v * po
            for k, v in observer.structure.its_counts.items()
        } == dict(expected_its)
        assert {
            compact_canonical_code(k): v * po
            for k, v in observer.structure.template_counts.items()
        } == dict(expected_templates)
        for mapping in rng.permutation(list(costs))[:5]:
            assert accumulator.contains_reference(tuple(mapping)) == (
                costs[tuple(mapping)] == target
            )


def test_native_shards_partition_the_materialized_candidates(native_library):
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    a = nx.to_numpy_array(nx.path_graph(5))
    b = nx.to_numpy_array(nx.cycle_graph(5))
    pair = (graph(a), graph(b))
    whole = enumerate_native_candidates(
        pair, 5, library_path=native_library, node_properties=()
    )
    pieces = [
        enumerate_native_candidates(
            pair,
            5,
            library_path=native_library,
            node_properties=(),
            shards=4,
            shard_index=i,
        )
        for i in range(4)
    ]
    assert whole["complete"] and all(part["complete"] for part in pieces)
    combined = [mapping for part in pieces for mapping in part["mappings"]]
    assert len(combined) == len(set(combined))
    assert set(combined) == set(whole["mappings"])


def test_atom_property_frequencies_use_exact_orbit_multiplicities(native_library):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates
    from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator
    from synkit.Chem.Mapper.exact.symmetry import permutation_group_order
    from synkit.Chem.Mapper.graph.automorphism import bounded_automorphism_permutations

    a = np.zeros((4, 4))
    pair = (graph(a), graph(a))
    left, right = [0, 0, 1, 1], [0, 1, 0, 1]
    pair[0].props["charges"] = left
    pair[1].props["charges"] = right
    rg, _ = bounded_automorphism_permutations(
        pair[0], binary=False, node_properties=("charges",)
    )
    pg, _ = bounded_automorphism_permutations(
        pair[1], binary=False, node_properties=("charges",)
    )
    ro, po = permutation_group_order(rg[1:]), permutation_group_order(pg[1:])
    config = GlobalShellConfig(
        symmetry_node_properties=("charges",), reaction_center_properties=("charges",)
    )
    observer = _BlindShellObserver(a, a, [6] * 4, {"charges": (left, right)}, config)
    accumulator = OrbitAccumulator(
        observer, rg[1:], ro, po, library_path=native_library
    )
    result = enumerate_native_candidates(
        pair,
        0,
        library_path=native_library,
        node_properties=("charges",),
        callback=accumulator.observe,
    )
    assert result["complete"]
    accumulator.finish()
    assert observer.count * po == 24
    assert dict(observer.atom_counts) == {i: 12 // po for i in range(4)}
    assert not observer.bond_counts
    assert sorted(observer.structure.its_counts.values()) == [1, 1, 4]


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mode", ["reference_cd", "minimal"])
def test_native_public_analysis_matches_python(native_library, workers, mode):
    from synkit.Chem.Mapper.analysis import (
        GlobalShellConfig,
        analyze_reference_blinded_global_shell,
    )
    from synkit.Chem.Mapper.native_analysis import (
        analyze_reference_blinded_native_shell,
    )

    a = nx.to_numpy_array(nx.star_graph(3))
    b = nx.to_numpy_array(nx.path_graph(4))
    lgp = graph(a), graph(b)
    config = GlobalShellConfig(
        time_limit_seconds=10,
        use_slap_seed=False,
        symmetry_node_properties=(),
        reaction_center_properties=(),
    )
    reference = (0, 1, 2, 3)
    expected = analyze_reference_blinded_global_shell(
        lgp,
        reference,
        target_mode=mode,
        config=config,
    )
    actual = analyze_reference_blinded_native_shell(
        lgp,
        reference,
        config=config,
        library_path=native_library,
        workers=workers,
        target_mode=mode,
    )
    assert expected.complete and actual.complete
    assert actual.reference_class_observed == expected.reference_class_observed
    assert actual.minimum_cost == expected.minimum_cost
    for name in (
        "target",
        "representative_solution_count",
        "labeled_solution_count",
        "symmetry_group_order",
        "mapping_hartley_entropy_nats",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.structure == expected.structure
    for name in (
        "bond_change_counts",
        "atom_change_counts",
        "bond_union",
        "bond_intersection",
        "frequency_denominator",
    ):
        assert getattr(actual.reaction_center, name) == getattr(
            expected.reaction_center, name
        )


def test_native_callback_budget_preserves_counters(native_library):
    from synkit.Chem.Mapper.exact.native_candidates import (
        NativeEnumerationStop,
        enumerate_native_candidates,
    )

    a = nx.to_numpy_array(nx.path_graph(4))

    def stop(mapping, cost):
        raise NativeEnumerationStop("mapping_limit")

    result = enumerate_native_candidates(
        (graph(a), graph(a)),
        0,
        node_properties=(),
        library_path=native_library,
        callback=stop,
    )
    assert not result["complete"]
    assert result["reason"] == "mapping_limit"
    assert result["candidate_count"] == 1
    assert result["visited_nodes"] > 0


def test_native_parallel_mapping_budget_is_global(native_library):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_parallel import parallel_orbit_search

    a = nx.to_numpy_array(nx.path_graph(5))
    b = nx.to_numpy_array(nx.star_graph(4))
    config = GlobalShellConfig(
        time_limit_seconds=10,
        max_mappings=1,
        symmetry_node_properties=(),
        reaction_center_properties=(),
    )
    observer = _BlindShellObserver(a, b, [6] * 5, {}, config)
    result, accumulator = parallel_orbit_search(
        (graph(a), graph(b)),
        4.0,
        config,
        None,
        observer,
        library_path=native_library,
        workers=2,
        shard_count=4,
    )
    assert not result["complete"]
    assert "mapping_limit" in result["reasons"]
    assert result["retained_worker_records"] <= 1
    assert len(accumulator.seen) <= 1


@pytest.mark.parametrize("budget", [1, 7, 40])
def test_resumable_frontier_partitions_candidates(native_library, budget):
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    a = nx.to_numpy_array(nx.path_graph(5))
    b = nx.to_numpy_array(nx.star_graph(4))
    lgp = graph(a), graph(b)
    args = {"library_path": native_library, "node_properties": (), "max_mappings": None}
    whole = enumerate_native_candidates(lgp, 4.0, **args)
    assert whole["complete"]
    pending, visited, found = [()], set(), []
    while pending:
        prefix = pending.pop()
        assert prefix not in visited
        visited.add(prefix)
        result = enumerate_native_candidates(
            lgp, 4.0, prefix=prefix, slice_nodes=budget, **args
        )
        assert result["reason"] in (None, "work_slice")
        assert all(len(child) > len(prefix) for child in result["frontier"])
        pending.extend(result["frontier"])
        found.extend(result["mappings"])
    assert len(found) == len(set(found))
    assert set(found) == set(whole["mappings"])


def test_frontier_merge_matches_static_search(native_library):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_frontier import frontier_orbit_search
    from synkit.Chem.Mapper.exact.native_parallel import parallel_orbit_search

    a = nx.to_numpy_array(nx.path_graph(5))
    b = nx.to_numpy_array(nx.star_graph(4))
    config = GlobalShellConfig(
        time_limit_seconds=10,
        symmetry_node_properties=(),
        reaction_center_properties=(),
    )
    observers = [_BlindShellObserver(a, b, [6] * 5, {}, config) for _ in range(2)]
    expected, _ = parallel_orbit_search(
        (graph(a), graph(b)),
        4.0,
        config,
        None,
        observers[0],
        library_path=native_library,
        workers=2,
    )
    actual, _ = frontier_orbit_search(
        (graph(a), graph(b)),
        4.0,
        config,
        None,
        observers[1],
        library_path=native_library,
        workers=2,
        slice_nodes=2,
    )
    assert actual["complete"] and expected["complete"]
    assert actual["candidate_count"] == expected["candidate_count"]
    assert (
        actual["weighted_product_representatives"]
        == expected["weighted_product_representatives"]
    )
    assert observers[0].bond_counts == observers[1].bond_counts
    from synkit.Chem.Mapper.exact.native_canonical import transport_canonical_key

    for field in ("its_counts", "template_counts"):
        expected_counts = getattr(observers[0].structure, field)
        actual_counts = getattr(observers[1].structure, field)
        assert {
            transport_canonical_key(key): value
            for key, value in expected_counts.items()
        } == actual_counts
    assert observers[0].structure.finalize(
        shell_complete=True, reference_mapping=tuple(range(5)), class_count_scope="test"
    ) == observers[1].structure.finalize(
        shell_complete=True, reference_mapping=tuple(range(5)), class_count_scope="test"
    )
    assert actual["completed_batches"] > 2


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_direct_its_encoding_matches_graph_encoding(native_library, radius):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code
    from synkit.Chem.Mapper.exact.native_its import NativeITSCanonicalizer
    from synkit.Chem.Mapper.spectrum import (
        _attributed_its_graph,
        _changed_atoms_and_bonds,
        _template_context,
    )

    a = nx.to_numpy_array(nx.path_graph(5)) * 1.5
    b = nx.to_numpy_array(nx.star_graph(4))
    properties = {"charges": ((0, 0.0, 1, 1, -1), (0, 1, 0.0, -1, 1))}
    observer = _BlindShellObserver(
        a, b, [6] * 5, properties, GlobalShellConfig(template_radius=radius)
    )
    direct = NativeITSCanonicalizer(observer, (), native_library)
    for mapping in permutations(range(5)):
        transported = b[np.ix_(mapping, mapping)]
        changed, _ = _changed_atoms_and_bonds(a, transported, properties, mapping, 1e-9)
        context = _template_context(a, transported, changed, radius)
        for selected in (None, context):
            graph = _attributed_its_graph(
                a,
                transported,
                [6] * 5,
                properties,
                mapping,
                context=selected,
                retain_boundary=selected is not None,
            )
            expected = native_canonical_code(
                graph, library_path=native_library, compact=True
            )
            assert direct.canonical(mapping, transported, selected) == expected
            paired = direct.paired_matrix(mapping)
            certificate, unused_order = direct.canonical(
                mapping, transported, selected, paired=paired, require_group=False
            )
            assert certificate == expected[0]
            assert unused_order == (1 if not certificate[0] else 0)
            assert np.array_equal(paired, direct.paired_matrix(mapping))
            from synkit.Chem.Mapper.exact.native_canonical import (
                PackedCanonicalKey,
                compact_canonical_code,
                compact_code_identifier,
            )
            from synkit.Chem.Mapper.spectrum import _code_identifier

            packed_key, packed_order = direct.canonical(
                mapping, transported, selected, paired=paired, packed=True
            )
            if not isinstance(packed_key, PackedCanonicalKey):
                packed_key = PackedCanonicalKey(packed_key)
            assert tuple(packed_key) == expected[0]
            assert packed_order == expected[1]
            assert compact_code_identifier(packed_key) == _code_identifier(
                compact_canonical_code(expected[0])
            )


def test_streamed_class_identifiers_preserve_legacy_escaping(native_library):
    from synkit.Chem.Mapper.exact.native_canonical import (
        compact_canonical_code,
        compact_code_identifier,
        native_canonical_code,
    )
    from synkit.Chem.Mapper.spectrum import _code_identifier

    for n in range(8):
        graph = nx.path_graph(n)
        for node in graph:
            graph.nodes[node]["color"] = (
                "single'quote",
                'double"quote',
                chr(92),
                "日本語",
                chr(10),
                node % 3,
            )
        for left, right in graph.edges:
            graph.edges[left, right]["color"] = ("edge", 0.5, True, None)
        if n:
            graph.add_edge(0, 0, color="loop")
        key, _ = native_canonical_code(graph, library_path=native_library, compact=True)
        assert compact_code_identifier(key) == _code_identifier(
            compact_canonical_code(key)
        )


def test_frontier_global_cap_and_zero_time_fail_closed(native_library):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_frontier import frontier_orbit_search

    a = nx.to_numpy_array(nx.path_graph(5))
    b = nx.to_numpy_array(nx.star_graph(4))
    for seconds, cap, expected in ((0, 100, "time_limit"), (10, 1, "mapping_limit")):
        config = GlobalShellConfig(
            time_limit_seconds=seconds,
            max_mappings=cap,
            symmetry_node_properties=(),
            reaction_center_properties=(),
        )
        observer = _BlindShellObserver(a, b, [6] * 5, {}, config)
        result, _ = frontier_orbit_search(
            (graph(a), graph(b)),
            4.0,
            config,
            None,
            observer,
            library_path=native_library,
            workers=2,
            slice_nodes=2,
        )
        assert not result["complete"]
        assert expected in result["reasons"]
        assert result["retained_worker_records"] <= cap


def test_cached_exact_keys_survive_transfer_and_hash_collisions():
    import pickle

    from synkit.Chem.Mapper.exact.native_canonical import CachedCanonicalKey

    a = CachedCanonicalKey(((("color", "a"),), ()))
    b = CachedCanonicalKey(((("color", "b"),), ()))
    from synkit.Chem.Mapper.exact.native_canonical import compact_code_identifier

    expected_id = compact_code_identifier(a)
    object.__setattr__(a, "identifier", expected_id)
    restored = pickle.loads(pickle.dumps(a))
    assert restored.identifier == expected_id
    assert compact_code_identifier(restored) == expected_id
    assert restored == a and hash(restored) == hash(a.parts)
    assert {a: 7}[a.parts] == 7
    object.__setattr__(b, "_hash", hash(a))  # simulate a hash collision
    assert a != b
    assert len({a: 7, b: 9}) == 2


@pytest.mark.parametrize("budget", [1, 3, 17])
@pytest.mark.parametrize("batch_size", [1, 3, 16, 64])
def test_prefix_trie_covers_exact_candidate_set(native_library, budget, batch_size):
    from collections import deque

    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    lgp = (
        graph(nx.to_numpy_array(nx.path_graph(5))),
        graph(nx.to_numpy_array(nx.star_graph(4))),
    )
    args = {"library_path": native_library, "node_properties": (), "max_mappings": None}
    whole = enumerate_native_candidates(lgp, 4.0, **args)
    pending, observed = deque([()]), []
    while pending:
        batch = tuple(pending.popleft() for _ in range(min(batch_size, len(pending))))
        result = enumerate_native_candidates(
            lgp, 4.0, prefixes=batch, slice_nodes=budget, **args
        )
        assert result["reason"] in (None, "work_slice")
        assert (
            result["visited_nodes"]
            == result["prefix_replay_nodes"] + result["new_search_nodes"]
        )
        observed.extend(result["mappings"])
        pending.extend(result["frontier"])
    assert Counter(observed) == Counter(whole["mappings"])


def test_prefix_trie_rejects_overlapping_subtrees(native_library):
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    lgp = (graph(nx.to_numpy_array(nx.path_graph(4))),) * 2
    for prefixes in (((), (0,)), ((0,), (0,)), ((0,), (0, 1))):
        with pytest.raises(ValueError, match="disjoint"):
            enumerate_native_candidates(
                lgp,
                0,
                library_path=native_library,
                prefixes=prefixes,
                slice_nodes=2,
                node_properties=(),
            )


def test_packed_key_encoding_is_independent_of_aliasing_and_roundtrips():
    import pickle

    from synkit.Chem.Mapper.exact.native_canonical import (
        PackedCanonicalKey,
        compact_canonical_code,
        compact_code_identifier,
    )
    from synkit.Chem.Mapper.spectrum import _code_identifier
    from synkit.Graph.Canon.exact import _encode_color

    color = _encode_color(("日本語", "quote'\\", 1, 1.0, True, None))
    edge = _encode_color((0.0, 1.5))
    code = ((color, color), ((0, 0, edge), (0, 1, edge)))
    separate = tuple(
        _encode_color(("日本語", "quote'\\", 1, 1.0, True, None)) for _ in range(2)
    )
    first, second = PackedCanonicalKey(code), PackedCanonicalKey((separate, code[1]))
    assert first.parts == second.parts and hash(first) == hash(second)
    restored = pickle.loads(pickle.dumps(first))
    assert restored == first and tuple(restored) == code
    assert compact_code_identifier(restored) == _code_identifier(
        compact_canonical_code(code)
    )
    other = PackedCanonicalKey((separate, ((0, 1, edge),)))
    object.__setattr__(other, "_hash", hash(first))
    assert first != other


@pytest.mark.parametrize("centers", [2, 3, 4])
def test_twin_shortcut_preserves_legacy_code_and_complete_group(
    native_library, centers
):
    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

    g = nx.path_graph(centers)
    for center in range(centers):
        g.add_edges_from((center, centers + 2 * center + leaf) for leaf in range(2))
    nx.set_node_attributes(g, ("typed", 6), "color")
    nx.set_edge_attributes(g, (1.0, 1.5), "color")
    expected = ExactColoredGraphCanonicalizer(g).canonicalize()
    actual, order = native_canonical_code(g, library_path=native_library)
    assert actual == expected.canonical_code
    assert order == len(expected.automorphisms)


def test_twin_product_group_order_and_overflow(native_library):
    import math

    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code

    g = nx.star_graph(11)
    _, order = native_canonical_code(g, library_path=native_library)
    assert order == math.factorial(11)
    with pytest.raises(RuntimeError, match="failed"):
        native_canonical_code(nx.empty_graph(21), library_path=native_library)


def test_pattern_cache_hits_preserve_all_exact_class_weights(
    native_library, monkeypatch
):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator

    a, b = nx.to_numpy_array(nx.star_graph(4)), nx.to_numpy_array(nx.path_graph(5))
    rg, pg = automorphisms(a), automorphisms(b)
    selected = [
        m
        for m in permutations(range(5))
        if float(np.abs(a - b[np.ix_(m, m)]).sum() / 2) == 4
    ]
    results = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("SYNKIT_NATIVE_PATTERN_CACHE", enabled)
        config = GlobalShellConfig(
            symmetry_node_properties=(), reaction_center_properties=()
        )
        observer = _BlindShellObserver(a, b, [6] * 5, {}, config)
        acc = OrbitAccumulator(
            observer, rg[1:], len(rg), len(pg), library_path=native_library
        )
        for mapping in selected:
            acc.observe(mapping, 4)
        acc.finish()
        assert observer.count * len(pg) == len(selected)
        if enabled == "1":
            assert acc.pattern_hits > 0
        results.append(
            (
                observer.count,
                dict(observer.structure.its_counts),
                dict(observer.structure.template_counts),
                dict(observer.bond_counts),
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("radius", [0, 1, 3, 20])
def test_native_changes_and_boundaries_match_public_semantics(native_library, radius):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_its import NativeITSCanonicalizer
    from synkit.Chem.Mapper.spectrum import (
        _changed_atoms_and_bonds,
        _template_context,
        _typed,
    )

    a = np.array(
        [[0, 1, 0, 0], [1, 0, 0.5, 0], [0, 0.5, 0, 2], [0, 0, 2, 0]], dtype=float
    )
    b = np.array(
        [[0, 1.0000001, 1.5, 0], [1.0000001, 0, 0, 0], [1.5, 0, 0, 2], [0, 0, 2, 0]]
    )
    # Numeric equality is the reporting rule, even when typed colors differ.
    properties = {"charges": ([0, 0.0, 1, -1], [0.0, 0, -1, 1])}
    elements = [6, "C", 8, "O"]
    observer = _BlindShellObserver(
        a, b, elements, properties, GlobalShellConfig(template_radius=radius)
    )
    observer.tolerance = 1e-6
    canon = NativeITSCanonicalizer(observer, [], native_library)
    for mapping in permutations(range(4)):
        transported = b[np.ix_(mapping, mapping)]
        paired = canon.paired_matrix(mapping)
        atoms, bonds, context = canon.changes(mapping, paired)
        expected_atoms, expected_bonds = _changed_atoms_and_bonds(
            a, transported, properties, mapping, observer.tolerance
        )
        assert bonds == sorted(expected_bonds)
        assert atoms == [
            i
            for i in range(4)
            if any(left[i] != right[mapping[i]] for left, right in properties.values())
        ]
        assert context == sorted(
            _template_context(a, transported, expected_atoms, radius)
        )
        for selected in ((), (0,), (0, 2), (0, 1, 2, 3), tuple(context)):
            expected = [
                tuple(
                    sorted(
                        (_typed(elements[j]), float(a[i, j]), float(transported[i, j]))
                        for j in range(4)
                        if j not in selected
                        and (a[i, j] != 0 or transported[i, j] != 0)
                    )
                )
                for i in selected
            ]
            assert canon.boundaries(selected, paired) == expected


def test_native_assignment_duals_and_forced_edges_against_permutations(native_library):
    import ctypes

    inf = 100000000
    function = ctypes.CDLL(str(native_library)).synkit_assignment_certificate
    pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ctypes.c_int, pointer, pointer, pointer, pointer, pointer]
    function.restype = ctypes.c_int
    rng = np.random.default_rng(20260908)
    matrices = [
        np.array([[0, 0], [0, 1]], dtype=np.int32),
        np.array([[0, inf, inf], [0, inf, inf], [inf, 0, 0]], dtype=np.int32),
        np.zeros((7, 7), dtype=np.int32),
    ]
    for n in range(2, 8):
        for _ in range(8):
            matrix = rng.integers(0, 12, size=(n, n), dtype=np.int32)
            matrix[rng.random((n, n)) < 0.3] = inf
            matrices.append(matrix)
    for matrix in matrices:
        n = len(matrix)
        expected = np.full((n, n), inf, dtype=np.int32)
        for mapping in permutations(range(n)):
            entries = matrix[np.arange(n), mapping]
            if np.any(entries == inf):
                continue
            cost = int(entries.sum())
            for row, col in enumerate(mapping):
                expected[row, col] = min(expected[row, col], cost)
        optimum = int(expected.min())
        match, u, v = [np.empty(n, dtype=np.int32) for _ in range(3)]
        forced = np.empty((n, n), dtype=np.int32)
        observed = function(
            n, *(x.ctypes.data_as(pointer) for x in (matrix, match, u, v, forced))
        )
        assert observed == optimum
        if optimum == inf:
            continue
        assert sorted(match.tolist()) == list(range(n))
        assert int(matrix[np.arange(n), match].sum()) == optimum
        assert int(u.sum() + v.sum()) == optimum
        assert np.all(u[:, None] + v[None, :] <= matrix)
        assert np.array_equal(forced, expected)


@pytest.mark.parametrize("same_components", [False, True])
def test_native_group_support_factorization_handles_component_exchange(
    native_library, same_components
):
    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

    graph = nx.disjoint_union(
        nx.cycle_graph(5), nx.cycle_graph(5 if same_components else 6)
    )
    code, order = native_canonical_code(
        graph, library_path=native_library, timeout_seconds=5
    )
    expected = ExactColoredGraphCanonicalizer(graph).canonicalize()
    assert expected.complete
    assert code == expected.canonical_code
    # D5 x D6, or (D5 x D5) semidirect the exchange of equal components.
    assert order == (200 if same_components else 120)


def test_wall_sliced_frontier_preserves_slow_callback_coverage(native_library):
    import time
    from collections import deque

    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    pair = (
        graph(nx.to_numpy_array(nx.path_graph(7))),
        graph(nx.to_numpy_array(nx.cycle_graph(7))),
    )
    options = {
        "library_path": native_library,
        "node_properties": (),
        "max_mappings": None,
        "time_limit_seconds": 10,
    }
    whole = enumerate_native_candidates(pair, 7.0, **options)
    expected = set(whole["mappings"])
    assert whole["complete"] and len(expected) > 10
    pending, observed, slices = deque([()]), [], 0

    def slow(mapping, _cost):
        observed.append(mapping)
        time.sleep(0.006)

    while pending:
        prefixes = [pending.popleft() for _ in range(min(16, len(pending)))]
        part = enumerate_native_candidates(
            pair, 7.0, prefixes=prefixes, slice_nodes=1000000, callback=slow, **options
        )
        assert part["reason"] in (None, "work_slice")
        slices += 1
        pending.extend(part["frontier"])
    assert slices > 1
    assert len(observed) == len(set(observed))
    assert set(observed) == expected


def test_palette_transport_preserves_identity_ids_and_digest_collisions():
    import gc
    import pickle

    from synkit.Chem.Mapper.exact.native_canonical import (
        PackedCanonicalKey,
        compact_code_identifier,
        transport_canonical_key,
    )

    left = PackedCanonicalKey(((("node", "C"),), ()))
    right = PackedCanonicalKey(((("node", "N"),), ()))
    transported = transport_canonical_key(left)
    assert isinstance(transported, tuple)
    gc.collect()
    assert not gc.is_tracked(transported)
    assert compact_code_identifier(transported) == compact_code_identifier(left)
    assert pickle.loads(pickle.dumps(transported)) == transported
    # Even a collision in the public digest retains distinct full certificates.
    for key in (left, right):
        object.__setattr__(key, "identifier", "0" * 64)
    assert compact_code_identifier(transport_canonical_key(left)) == "0" * 64
    assert len({transport_canonical_key(left), transport_canonical_key(right)}) == 2


def test_unique_terminal_completion_resumes_full_prefix_exactly(native_library):
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    n = 8
    a = LabeledGraph({i: {} for i in range(n)}, list(range(1, n + 1)))
    options = {
        "library_path": native_library,
        "node_properties": (),
        "max_mappings": None,
    }
    identity = tuple(range(n))
    whole = enumerate_native_candidates((a, a), 0.0, **options)
    assert whole["complete"] and whole["mappings"] == [identity]
    assert whole["visited_nodes"] == 2  # root plus verified unique leaf
    first = enumerate_native_candidates(
        (a, a), 0.0, prefixes=[()], slice_nodes=1, **options
    )
    assert first["reason"] == "work_slice" and first["frontier"] == [identity]
    resumed = enumerate_native_candidates(
        (a, a), 0.0, prefixes=first["frontier"], slice_nodes=1, **options
    )
    assert resumed["complete"] and resumed["mappings"] == [identity]
    partial = enumerate_native_candidates(
        (a, a), 0.0, prefix=identity[:3], slice_nodes=10, **options
    )
    assert partial["complete"] and partial["mappings"] == [identity]
    wrong = enumerate_native_candidates(
        (a, a), 0.0, prefix=(1,), slice_nodes=10, **options
    )
    assert wrong["complete"] and not wrong["mappings"]
    wrong_cost = enumerate_native_candidates((a, a), 1.0, **options)
    assert wrong_cost["complete"] and not wrong_cost["mappings"]
