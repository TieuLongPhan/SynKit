import math

import numpy as np

from synkit.Chem.Mapper import (
    GlobalShellConfig,
    analyze_reference_blinded_global_shell,
    exact_its_and_template_codes,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def _graph(size, edges, labels=None):
    adjacency = {atom: {} for atom in range(size)}
    for left, right in edges:
        adjacency[left][right] = 1
        adjacency[right][left] = 1
    return LabeledGraph(adjacency, labels or [6] * size)


def _matrix(graph):
    size = len(graph.labels)
    result = np.zeros((size, size), dtype=float)
    for left, neighbors in graph.graph.items():
        for right, order in neighbors.items():
            result[left, right] = order
    return result


def test_exact_codes_are_invariant_under_independent_endpoint_relabeling():
    reactant = _graph(4, ((0, 1), (1, 2)), [6, 6, 8, 7])
    product = _graph(4, ((0, 2), (2, 1)), [6, 7, 6, 8])
    mapping = (0, 2, 3, 1)
    reactant_matrix = _matrix(reactant)
    product_matrix = _matrix(product)
    original = exact_its_and_template_codes(
        reactant_matrix,
        product_matrix,
        reactant.labels,
        {},
        mapping,
    )

    reactant_relabel = (2, 0, 3, 1)
    product_relabel = (1, 3, 0, 2)
    relabeled_reactant = np.zeros_like(reactant_matrix)
    relabeled_product = np.zeros_like(product_matrix)
    relabeled_elements = [None] * 4
    relabeled_mapping = [None] * 4
    for old_left in range(4):
        new_left = reactant_relabel[old_left]
        relabeled_elements[new_left] = reactant.labels[old_left]
        relabeled_mapping[new_left] = product_relabel[mapping[old_left]]
        for old_right in range(4):
            relabeled_reactant[new_left, reactant_relabel[old_right]] = reactant_matrix[
                old_left, old_right
            ]
            relabeled_product[product_relabel[old_left], product_relabel[old_right]] = (
                product_matrix[old_left, old_right]
            )

    relabeled = exact_its_and_template_codes(
        relabeled_reactant,
        relabeled_product,
        relabeled_elements,
        {},
        relabeled_mapping,
    )
    assert original[2] is relabeled[2] is None
    assert original[:2] == relabeled[:2]


def test_structure_spectrum_finds_exact_its_and_template_ambiguity():
    graph = _graph(4, ((0, 1),))
    result = analyze_reference_blinded_global_shell(
        [graph, graph.copy()],
        [0, 2, 1, 3],
        target_mode="reference_cd",
        config=GlobalShellConfig(
            binary=True,
            max_bijections=None,
            max_mappings=None,
            time_limit_seconds=5,
            symmetry_pruning=False,
        ),
    )

    assert result.complete is True
    assert result.structure.complete is True
    assert result.structure.observed_its_class_count == 2
    assert result.structure.observed_template_class_count == 2
    assert math.isclose(result.mapping_hartley_entropy_nats, math.log(20))
    assert math.isclose(
        result.structure.its_hartley_entropy_nats,
        math.log(2),
    )
    assert sum(count for _, count in result.structure.its_class_counts) == 20
    assert result.structure.reference_its_class_observed is True


def test_structure_analysis_can_be_disabled_or_fail_closed_independently():
    graph = _graph(3, ((0, 1), (1, 2)))
    disabled = analyze_reference_blinded_global_shell(
        [graph, graph.copy()],
        [0, 1, 2],
        target_mode="reference_cd",
        config=GlobalShellConfig(
            binary=True,
            max_bijections=None,
            structure_analysis=False,
        ),
    )
    assert disabled.complete is True
    assert disabled.structure.complete is False
    assert disabled.structure.incomplete_reason == "disabled"

    bounded = analyze_reference_blinded_global_shell(
        [graph, graph.copy()],
        [0, 1, 2],
        target_mode="reference_cd",
        config=GlobalShellConfig(
            binary=True,
            max_bijections=None,
            structure_timeout_seconds=0,
        ),
    )
    assert bounded.complete is True
    assert bounded.structure.complete is False
    assert bounded.structure.incomplete_reason.startswith("full_its:")
