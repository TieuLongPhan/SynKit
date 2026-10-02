"""Completion must account for un-emitted members of the last symmetry orbit."""

import pytest
from itertools import permutations
from math import factorial

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.identifiability import Endpoint


@pytest.mark.parametrize('target', [0, 'minimal'])
@pytest.mark.parametrize('stream', [False, True])
def test_cap_inside_final_expanded_orbit_is_incomplete(target, stream):
    endpoint = Endpoint((6, 6), (0, 0), (0, 0), ())
    emitted = []
    result = enumerate_distance_mappings(
        [endpoint.graph(), endpoint.graph()], CD=target, binary=False,
        max_bijections=None, compute_minimum_cost=target == 'minimal',
        symmetry_pruning=True, expand_symmetry=True, max_mappings=1,
        collect_mappings=not stream,
        mapping_callback=(lambda mapping, cost: emitted.append(tuple(mapping))) if stream else None)
    assert not result.complete
    assert result.status == 'timeout'
    assert result.truncation_reason == 'mapping_limit'
    if target == 'minimal' and not stream:
        # Single-pass collected minimization has not established exhaustion;
        # its provisional result must be withdrawn, as elsewhere in the API.
        assert result.minimum_cost is None and result.mappings == []
        assert result.selected_mapping_count == 0
    else:
        assert result.selected_mapping_count == 1
        assert len(emitted if stream else result.mappings) == 1


@pytest.mark.parametrize('target', [0, 'minimal'])
def test_cap_at_exact_end_of_expanded_output_can_be_complete(target):
    endpoint = Endpoint((6, 6), (0, 0), (0, 0), ())
    result = enumerate_distance_mappings(
        [endpoint.graph(), endpoint.graph()], CD=target, binary=False,
        max_bijections=None, compute_minimum_cost=target == 'minimal',
        symmetry_pruning=True, expand_symmetry=True, max_mappings=2)
    assert result.complete and result.truncation_reason is None
    assert set(map(tuple, result.mappings)) == {(0, 1), (1, 0)}


@pytest.mark.parametrize('n', [1, 2, 3, 4])
@pytest.mark.parametrize('target', [0, 'minimal'])
@pytest.mark.parametrize('stream', [False, True])
def test_every_cap_around_small_complete_permutation_sets(n, target, stream):
    """148 boundaries: before, at and after exhaustion, collected and streamed."""
    endpoint = Endpoint((6,)*n, (0,)*n, (0,)*n, ())
    expected = set(permutations(range(n)))
    for cap in range(1, factorial(n)+2):
        emitted = []
        result = enumerate_distance_mappings(
            [endpoint.graph(), endpoint.graph()], CD=target, binary=False,
            max_bijections=None, compute_minimum_cost=target == 'minimal',
            symmetry_pruning=True, expand_symmetry=True, max_mappings=cap,
            collect_mappings=not stream,
            mapping_callback=(lambda mapping, cost: emitted.append(tuple(mapping))) if stream else None)
        assert result.complete == (cap >= len(expected))
        outputs = emitted if stream else list(map(tuple, result.mappings))
        assert len(outputs) == len(set(outputs)) and set(outputs) <= expected
        if result.complete:
            assert result.truncation_reason is None and set(outputs) == expected
        else:
            assert result.truncation_reason == 'mapping_limit'
            if target == 'minimal' and not stream:
                assert result.minimum_cost is None and not outputs
            else:
                assert len(outputs) == cap
