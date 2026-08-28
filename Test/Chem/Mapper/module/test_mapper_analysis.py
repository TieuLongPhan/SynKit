import itertools
from collections import Counter

from scripts import run_synister_global_shells as campaign
from synkit.Chem.Mapper import (
    GlobalShellConfig,
    analyze_reference_blinded_global_shell,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def _graph(size, edges, labels=None):
    adjacency = {atom: {} for atom in range(size)}
    for left, right in edges:
        adjacency[left][right] = 1
        adjacency[right][left] = 1
    return LabeledGraph(adjacency, labels or [6] * size)


def _cycle_graph(size, permutation=None):
    permutation = list(range(size)) if permutation is None else permutation
    edges = [
        (permutation[atom], permutation[(atom + 1) % size]) for atom in range(size)
    ]
    return _graph(size, edges)


def _config(**kwargs):
    return GlobalShellConfig(
        binary=True,
        max_bijections=None,
        max_mappings=None,
        time_limit_seconds=5,
        **kwargs,
    )


def test_reference_cd_search_is_invariant_to_equal_cost_reference_choice():
    graph = _graph(3, ((0, 1), (1, 2)))
    lgp = [graph, graph.copy()]
    identity = analyze_reference_blinded_global_shell(
        lgp,
        [0, 1, 2],
        target_mode="reference_cd",
        config=_config(),
    )
    reversal = analyze_reference_blinded_global_shell(
        lgp,
        [2, 1, 0],
        target_mode="reference_cd",
        config=_config(),
    )

    assert identity.reference_cd == reversal.reference_cd == 0
    assert identity.complete is reversal.complete is True
    assert identity.reference_class_observed is True
    assert reversal.reference_class_observed is True
    assert identity.reaction_center == reversal.reaction_center
    assert identity.representative_solution_count == 1
    assert identity.labeled_solution_count == 2


def test_minimal_search_reveals_suboptimal_reference_after_completion():
    permutation = [0, 2, 4, 1, 3]
    lgp = [_cycle_graph(5), _cycle_graph(5, permutation)]

    result = analyze_reference_blinded_global_shell(
        lgp,
        list(range(5)),
        target_mode="minimal",
        config=_config(),
    )

    assert result.complete is True
    assert result.minimum_cost == 0
    assert result.reference_cd > result.minimum_cost
    assert result.reference_gap_from_minimum == result.reference_cd
    assert result.reference_class_observed is False
    assert result.reference_is_global_minimum_proven is False
    assert result.labeled_solution_count == 10


def test_reaction_center_counts_match_direct_labeled_shell():
    graph = _graph(3, ((0, 1), (1, 2)))
    lgp = [graph, graph.copy()]
    reference = [0, 2, 1]
    target = chemical_distance(lgp, reference, binary=True)
    expected = [
        permutation
        for permutation in itertools.permutations(range(3))
        if chemical_distance(lgp, permutation, binary=True) == target
    ]
    bond_counts = Counter()
    for mapping in expected:
        for left, right in itertools.combinations(range(3), 2):
            before = int(right in graph.graph[left])
            after = int(mapping[right] in graph.graph[mapping[left]])
            if before != after:
                bond_counts[(left, right)] += 1

    result = analyze_reference_blinded_global_shell(
        lgp,
        reference,
        target_mode="reference_cd",
        config=_config(symmetry_pruning=False),
    )

    observed = {
        (left, right): count
        for left, right, count in result.reaction_center.bond_change_counts
    }
    assert result.complete is True
    assert result.backend == "binary_edit_support"
    assert result.representative_solution_count == len(expected)
    assert observed == bond_counts
    assert set(result.reaction_center.bond_intersection) == {
        pair for pair, count in bond_counts.items() if count == len(expected)
    }


def test_zero_time_limit_is_explicitly_incomplete():
    graph = _graph(4, ((0, 1), (1, 2), (2, 3)))
    result = analyze_reference_blinded_global_shell(
        [graph, graph.copy()],
        list(range(4)),
        target_mode="minimal",
        config=GlobalShellConfig(
            binary=True,
            max_bijections=None,
            time_limit_seconds=0,
        ),
    )

    assert result.complete is False
    assert result.status == "timeout"
    assert result.representative_solution_count == 0
    assert result.reference_is_global_minimum_proven is False


def test_campaign_case_contains_blind_results_without_raw_reference_mapping():
    options = _campaign_options()
    record = campaign._run_case(_campaign_row(1), options)

    assert "reference_mapping" not in record
    assert set(record["shells"]) == {"reference_cd", "minimal"}
    assert all(shell["complete"] for shell in record["shells"].values())
    assert all(shell["structure"]["complete"] for shell in record["shells"].values())
    assert all(
        shell["backend_statistics"]["seed"]["available"]
        for shell in record["shells"].values()
    )


def _campaign_options():
    return {
        "heavy_only": True,
        "blind_seed": "campaign-test",
        "binary": True,
        "time_limit_per_shell": 5,
        "max_bijections": None,
        "max_mappings": 100,
        "tolerance": 1e-9,
        "symmetry_pruning": True,
        "backend": "auto",
        "max_edit_support_pairs": 50_000,
        "use_slap_seed": True,
        "max_symmetry_automorphisms": 256,
        "symmetry_timeout_seconds": 0.25,
        "symmetry_max_search_nodes": 10_000,
        "structure_analysis": True,
        "template_radius": 1,
        "structure_timeout_seconds": 0.25,
        "structure_max_search_nodes": 100_000,
        "mode": "both",
        "campaign_manifest_sha256": "test-manifest",
    }


def _campaign_row(source_line):
    reaction_id = f"example:{source_line}"
    reaction = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    return {
        "source_line": str(source_line),
        "reaction_id": reaction_id,
        "mapped_reaction": f"{reaction}|{reaction_id}",
    }


def test_campaign_parallel_workers_return_complete_unique_records():
    records = list(
        campaign._parallel_case_records(
            [_campaign_row(1), _campaign_row(2)],
            _campaign_options(),
            workers=2,
            memory_limit_gib=4,
        )
    )

    assert {record["source_line"] for record in records} == {1, 2}
    assert all(record["record_sha256"] for record in records)
    assert all(
        shell["complete"]
        for record in records
        for shell in record["shells"].values()
    )
