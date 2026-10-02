"""Literal-set and failure-status controls for the independent global MILP."""

from types import SimpleNamespace

import numpy as np
import pytest

from Experiment.Synister.all_distance_oracle import binary_endpoint, literal_sets, weighted_cases
from Experiment.Synister.global_milp import build_model, doubled_distance, enumerate_milp
from synkit.Chem.Mapper.identifiability import Endpoint


def test_sparse_linearization_equals_literal_cost_for_every_assignment():
    for _, r, p in weighted_cases(8):
        model = build_model(r, p)
        for distance, maps in literal_sets(r, p).items():
            for mapping in maps:
                x = np.array([int(mapping[i] == p) for i, p in model.pairs])
                y = np.array([x[a]*x[b] for a, b in model.products])
                values = np.concatenate((x, y))
                assert model.constant + model.objective @ values == distance
                lhs = model.matrix @ values
                assert (lhs >= model.lower).all() and (lhs <= model.upper).all()


def test_all_binary_map_sets_and_minimum():
    for a, b in ((0, 0), (0, 63), (3, 7), (13, 42)):
        r, p = binary_endpoint(a), binary_endpoint(b)
        oracle = literal_sets(r, p)
        for target in ["minimal", *range(0, 15, 2)]:
            result = enumerate_milp(r, p, target=target, seconds=10)
            assert result["complete"], result
            expected = oracle.get(min(oracle) if target == "minimal" else target, set())
            assert set(result["mappings"]) == expected
            if target == "minimal":
                assert result["minimum_proved"] and result["minimum_doubled_cd"] == min(oracle)


def test_weighted_half_integer_distances_and_element_order():
    r = Endpoint((6, 8, 6), (0, 0, 0), (0, 0, 0), ((0, 1, 3),))
    p = Endpoint((6, 6, 8), (0, 0, 0), (0, 0, 0), ((0, 2, 2),))
    for target in range(8):
        result = enumerate_milp(r, p, target=target, seconds=5)
        assert result["complete"]
        assert set(result["mappings"]) == literal_sets(r, p).get(target, set())


def test_caps_are_not_completion():
    r = binary_endpoint(0)
    assert not enumerate_milp(r, r, seconds=0)["complete"]
    result = enumerate_milp(r, r, seconds=5, max_maps=1)
    assert result["minimum_proved"] and result["termination"] == "output_limit"
    assert not result["complete"]


def test_solver_time_limit_or_bad_witness_never_proves_minimum():
    r = binary_endpoint(0)
    limited = lambda *args, **kwargs: SimpleNamespace(status=1, message="limit", x=None)
    result = enumerate_milp(r, r, _solve=limited)
    assert not result["minimum_proved"] and not result["complete"]
    bad = lambda *args, **kwargs: SimpleNamespace(status=0, message="bad", x=np.full(16, 0.5))
    result = enumerate_milp(r, r, _solve=bad)
    assert result["termination"] == "invalid_solver_witness" and not result["complete"]


def test_invalid_domain_or_target_is_explicit():
    r = binary_endpoint(0)
    with pytest.raises(ValueError):
        enumerate_milp(r, r, target=0.5)
    with pytest.raises(ValueError):
        doubled_distance(r, r, (0, 0, 1, 2))


def test_feasible_seed_cutoff_preserves_all_optima_without_fixing_the_map():
    for _, r, p in weighted_cases(6):
        oracle = literal_sets(r, p)
        for seed in (min(oracle[min(oracle)]), min(oracle[max(oracle)])):
            result = enumerate_milp(r, p, initial_mapping=seed, seconds=10)
            assert result['complete'] and set(result['mappings']) == oracle[min(oracle)]
            assert result['seed_doubled_cd'] == doubled_distance(r, p, seed)
            assert result['seed_interface'] == 'feasible_cost_cutoff_not_native_warm_start'
    r = binary_endpoint(0)
    with pytest.raises(ValueError, match='bijection'):
        enumerate_milp(r, r, initial_mapping=(0, 0, 1, 2))
    with pytest.raises(ValueError, match='only defined for minimum'):
        enumerate_milp(r, r, target=0, initial_mapping=tuple(range(4)))
