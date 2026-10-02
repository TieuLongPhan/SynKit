from fractions import Fraction as F

import pytest

from synkit.Chem.Mapper.cohort_evaluation import paired_cohort_bounds


def test_wide_outer_interval_does_not_witness_reversal():
    result = paired_cohort_bounds([(0, 0), None])
    assert (result.outer_lower, result.outer_upper) == (F(-1, 2), F(1, 2))
    assert not result.guaranteed_substantive_reversal
    assert result.width_lower == 0 and result.width_upper == 1


def test_witnessed_reversal_despite_unresolved_case():
    result = paired_cohort_bounds([(-1, 1), None], weights=[F(9, 10), F(1, 10)])
    assert result.guaranteed_substantive_reversal
    assert result.lower_witness_assignment_upper == F(-4, 5)
    assert result.upper_witness_assignment_lower == F(4, 5)
    assert not result.exact


def test_zero_crossing_not_substantive_at_declared_margin():
    result = paired_cohort_bounds([("-1/100", "1/10")], margin="1/50")
    assert result.exact and not result.guaranteed_substantive_reversal


def test_robust_winner_and_unknown_only():
    assert paired_cohort_bounds([("1/2", "3/4")]).robust_a
    assert paired_cohort_bounds([("-3/4", "-1/2")]).robust_b
    unknown = paired_cohort_bounds([None])
    assert unknown.conditional_lower is None and unknown.width_upper == 2


@pytest.mark.parametrize("intervals,weights", [([], None), ([(0, 1)], ["1/2"]),
                                               ([(1, 0)], None), ([(0, 2)], None)])
def test_invalid_bounds_rejected(intervals, weights):
    with pytest.raises(ValueError):
        paired_cohort_bounds(intervals, weights=weights)
