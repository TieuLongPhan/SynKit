from itertools import permutations, product

import pytest

from Experiment.Synister.hydrogen_oracle import literal_h_cost, oracle
from synkit.Chem.Mapper.identifiability import Endpoint


def test_conditional_formula_against_literal_labeled_h_assignments():
    for total in range(4):
        counts = [h for h in product(range(total+1), repeat=3) if sum(h) == total]
        for rh, ph in product(counts, repeat=2):
            for mapping in permutations(range(3)):
                expected = sum(abs(rh[i]-ph[mapping[i]]) for i in range(3))
                assert literal_h_cost(rh, ph, mapping) == expected


def test_combined_optimum_can_exclude_every_heavy_optimum():
    # Abstract graph control, not a claim of chemically valid valence.
    bonds = ((0, 1, 2), (1, 2, 2))
    r = Endpoint((6, 6, 6), (0, 0, 0), (3, 0, 0), bonds)
    p = Endpoint((6, 6, 6), (0, 0, 0), (0, 3, 0), bonds)
    result = oracle(r, p)
    assert result["compatible_maps_tested"] == 6
    assert result["minimum_doubled_cost"] == {"heavy": 0, "combined": 4}
    assert not (set(map(tuple, result["optimizers"]["heavy"])) &
                set(map(tuple, result["optimizers"]["combined"])))
    assert not result["bond_label_sets_equal"]


def test_oracle_refuses_unbalanced_h_and_large_inventory():
    r = Endpoint((6,), (0,), (4,), ())
    p = Endpoint((6,), (0,), (3,), ())
    with pytest.raises(ValueError, match="domain"):
        oracle(r, p)
    large = Endpoint((6,)*9, (0,)*9, (0,)*9, ())
    with pytest.raises(ValueError, match="domain"):
        oracle(large, large)
