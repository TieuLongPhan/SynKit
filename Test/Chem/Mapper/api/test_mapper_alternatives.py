import hashlib
import itertools

import numpy as np

from synkit.Chem.Mapper import (
    GlobalShellConfig,
    enumerate_exact_its_alternatives,
    enumerate_mapped_reaction_its_alternatives,
    exact_its_and_template_codes,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def _graph(size, edges, labels=None):
    adjacency = {atom: {} for atom in range(size)}
    for left, right in edges:
        adjacency[left][right] = 1
        adjacency[right][left] = 1
    return LabeledGraph(adjacency, labels or [6] * size)


def _config(**kwargs):
    options = {
        "binary": True,
        "max_bijections": None,
        "max_mappings": None,
        "time_limit_seconds": 5,
        "symmetry_pruning": False,
    }
    options.update(kwargs)
    return GlobalShellConfig(**options)


def _matrix(graph):
    size = len(graph.labels)
    matrix = np.zeros((size, size), dtype=float)
    for source, neighbours in graph.graph.items():
        for target, weight in neighbours.items():
            matrix[source, target] = weight
    return matrix


def _code_identifier(code):
    return hashlib.sha256(repr(code).encode("utf-8", "surrogatepass")).hexdigest()


def test_reference_cd_returns_one_representative_per_alternative_its():
    graph = _graph(4, ((0, 1),))
    result = enumerate_exact_its_alternatives(
        [graph, graph.copy()],
        [0, 2, 1, 3],
        CD="reference",
        config=_config(),
    )

    assert result.complete is True
    assert result.target == result.reference_cd == 2
    assert result.shell_representative_mapping_count == 20
    assert result.shell_its_class_count == 2
    assert result.reference_its_class_observed is True
    assert result.alternative_its_class_count == 1
    assert len(result.classes) == 2
    assert len(result.alternatives) == 1
    assert {item.its_class_id for item in result.alternatives} < {
        item.its_class_id for item in result.classes
    }
    assert result.alternatives[0].representative_mapping_count == 4
    assert result.alternatives[0].labeled_mapping_count == 4


def test_arbitrary_and_minimal_cd_can_exclude_reference_class():
    graph = _graph(4, ((0, 1),))
    expected_ids = None
    for target in (0, "minimal"):
        result = enumerate_exact_its_alternatives(
            [graph, graph.copy()],
            [0, 2, 1, 3],
            CD=target,
            config=_config(),
        )
        assert result.complete is True
        assert result.reference_cd == 2
        assert result.reference_its_class_observed is False
        assert result.shell_its_class_count == 1
        assert result.alternative_its_class_count == 1
        assert tuple(item.its_class_id for item in result.classes) == tuple(
            item.its_class_id for item in result.alternatives
        )
        observed_ids = tuple(item.its_class_id for item in result.alternatives)
        expected_ids = observed_ids if expected_ids is None else expected_ids
        assert observed_ids == expected_ids


def test_all_three_vertex_numeric_shell_class_sets_match_brute_force():
    possible_edges = ((0, 1), (0, 2), (1, 2))
    graphs = [
        _graph(
            3,
            tuple(
                edge for index, edge in enumerate(possible_edges) if mask & (1 << index)
            ),
        )
        for mask in range(1 << len(possible_edges))
    ]
    permutations = tuple(itertools.permutations(range(3)))
    reference = (0, 1, 2)
    config = _config(backend="assignment")

    for reactant in graphs:
        for product in graphs:
            lgp = [reactant, product]
            reactant_matrix = _matrix(reactant)
            product_matrix = _matrix(product)
            classes_by_distance = {}
            for mapping in permutations:
                distance = chemical_distance(lgp, mapping, binary=True)
                code, _, reason = exact_its_and_template_codes(
                    reactant_matrix,
                    product_matrix,
                    reactant.labels,
                    {},
                    mapping,
                )
                assert reason is None
                classes_by_distance.setdefault(distance, set()).add(
                    _code_identifier(code)
                )
            reference_code, _, reason = exact_its_and_template_codes(
                reactant_matrix,
                product_matrix,
                reactant.labels,
                {},
                reference,
            )
            assert reason is None
            reference_id = _code_identifier(reference_code)

            targets = tuple(classes_by_distance) + (max(classes_by_distance) + 1,)
            for target in targets:
                expected = classes_by_distance.get(target, set()) - {reference_id}
                result = enumerate_exact_its_alternatives(
                    lgp,
                    reference,
                    CD=target,
                    seed_mode="none",
                    config=config,
                )
                assert result.complete is True
                assert {item.its_class_id for item in result.alternatives} == expected


def test_reference_seed_changes_only_search_strategy_not_exact_classes():
    graph = _graph(4, ((0, 1),))
    results = [
        enumerate_exact_its_alternatives(
            [graph, graph.copy()],
            [0, 2, 1, 3],
            CD="reference",
            seed_mode=mode,
            config=_config(backend="assignment"),
        )
        for mode in ("reference", "none")
    ]

    assert all(result.complete for result in results)
    assert results[0].seed_applied_to_backend is True
    assert results[1].seed_applied_to_backend is False
    assert results[0].shell_representative_mapping_count == (
        results[1].shell_representative_mapping_count
    )
    assert {item.its_class_id for item in results[0].alternatives} == {
        item.its_class_id for item in results[1].alternatives
    }


def test_interruption_never_claims_complete_alternative_set():
    graph = _graph(4, ((0, 1), (1, 2), (2, 3)))
    result = enumerate_exact_its_alternatives(
        [graph, graph.copy()],
        list(range(4)),
        CD="minimal",
        config=_config(time_limit_seconds=0),
    )

    assert result.shell_complete is False
    assert result.complete is False
    assert result.status == "timeout"
    assert result.incomplete_reason == "time_limit"


def test_symmetry_respects_every_unary_property_used_by_its_classes():
    graph = _graph(4, ())
    graph.set_prop("hcounts", [0, 0, 1, 1])
    results = []
    for symmetry_pruning in (False, True):
        results.append(
            enumerate_exact_its_alternatives(
                [graph, graph.copy()],
                list(range(4)),
                CD="reference",
                config=_config(
                    symmetry_pruning=symmetry_pruning,
                    reaction_center_properties=("hcounts",),
                    symmetry_node_properties=(),
                ),
            )
        )

    labeled, quotient = results
    assert labeled.shell_representative_mapping_count == 24
    assert quotient.shell_representative_mapping_count == 6
    assert quotient.shell_labeled_mapping_count == 24
    assert labeled.shell_its_class_count == quotient.shell_its_class_count == 3
    assert {item.its_class_id for item in labeled.alternatives} == {
        item.its_class_id for item in quotient.alternatives
    }
    assert sorted(item.labeled_mapping_count for item in quotient.alternatives) == [
        4,
        16,
    ]


def test_mapped_reaction_output_exposes_original_atom_map_correspondence():
    reaction = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    result = enumerate_mapped_reaction_its_alternatives(
        reaction,
        CD="minimal",
        seed_mode="reference",
        config=_config(),
        blind_seed="alternative-test",
    )

    assert result.shell.complete is True
    assert result.shell.alternative_its_class_count == 0
    assert len(result.shell.classes) == 1
    assert result.reaction_sha256
    assert sorted(result.reactant_atom_maps) == [1, 2, 3]
    assert sorted(result.product_atom_maps) == [1, 2, 3]
    payload = result.as_dict()
    assert payload["reactant_atom_maps"] == list(result.reactant_atom_maps)
    assert payload["product_atom_maps"] == list(result.product_atom_maps)
    assert payload["shell"]["alternatives"] == []
    assert len(payload["shell"]["classes"]) == 1
    correspondence = payload["shell"]["classes"][0]["atom_map_correspondence"]
    assert sorted(tuple(pair) for pair in correspondence) == [(1, 1), (2, 2), (3, 3)]
