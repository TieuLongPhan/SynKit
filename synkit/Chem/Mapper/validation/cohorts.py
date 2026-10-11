"""Separate intrinsic label sensitivity from unresolved computation.

Each case's paired F1 difference is in [-1,1]. Weights and scores are exact
rationals; these are identification bounds, not sampling confidence intervals.
"""

from dataclasses import dataclass
from fractions import Fraction


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise TypeError("Use an integer, Fraction or rational string, not a float")
    return Fraction(value)


@dataclass(frozen=True)
class CohortBounds:
    unresolved_weight: Fraction
    outer_lower: Fraction
    outer_upper: Fraction
    width_lower: Fraction
    width_upper: Fraction
    conditional_lower: Fraction | None
    conditional_upper: Fraction | None
    lower_witness_assignment_upper: Fraction
    upper_witness_assignment_lower: Fraction
    robust_a: bool
    robust_b: bool
    guaranteed_substantive_reversal: bool
    exact: bool


def paired_cohort_bounds(intervals, *, weights=None, margin=Fraction(1, 50)):
    """Bound a fixed common-valid cohort; ``None`` means unresolved.

    Resolved intervals must be attained shared-label extrema. For a global
    coupled annotation policy these sums need not be attainable; the function
    assumes casewise label choices are independent. Invalid predictions must
    be handled before calling, by an explicit cohort or operational policy.
    """
    intervals = list(intervals)
    if not intervals:
        raise ValueError("A nonempty comparison cohort is required")
    weights = ([Fraction(1, len(intervals))] * len(intervals) if weights is None
               else [rational(x) for x in weights])
    if len(weights) != len(intervals) or any(w < 0 for w in weights) or sum(weights) != 1:
        raise ValueError("Weights must be nonnegative, aligned and sum to one")
    margin = rational(margin)
    if not 0 <= margin <= 1:
        raise ValueError("Margin must lie in [0,1]")
    lower, upper, unknown = Fraction(), Fraction(), Fraction()
    for interval, weight in zip(intervals, weights):
        if interval is None:
            unknown += weight
        else:
            lo, hi = map(rational, interval)
            if not -1 <= lo <= hi <= 1:
                raise ValueError("Invalid paired score interval")
            lower += weight * lo
            upper += weight * hi
    resolved_weight = 1-unknown
    outer_lo, outer_hi = lower-unknown, upper+unknown
    return CohortBounds(
        unknown, outer_lo, outer_hi, upper-lower, upper-lower+2*unknown,
        lower/resolved_weight if resolved_weight else None,
        upper/resolved_weight if resolved_weight else None,
        lower+unknown, upper-unknown,
        outer_lo > margin, outer_hi < -margin,
        lower+unknown < -margin and upper-unknown > margin,
        unknown == 0,
    )
