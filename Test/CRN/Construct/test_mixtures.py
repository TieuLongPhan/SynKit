from __future__ import annotations

import unittest

from synkit.CRN.Construct.mixtures import (
    iter_mixtures_arity1,
    iter_mixtures_arity2,
    iter_mixtures_arityk,
)

from ._case import SEEDS, X, Y, Z


class TestMixtures(unittest.TestCase):
    def test_iter_mixtures_arity1_uses_frontier(self):
        out = list(
            iter_mixtures_arity1(
                pool_keys=SEEDS + [X],
                frontier_keys=[X],
                use_frontier=True,
                cap=10,
            )
        )
        self.assertEqual(out, [(X,)])

    def test_iter_mixtures_arity2_matches_user_case_pairs(self):
        out = list(
            iter_mixtures_arity2(
                pool_keys=SEEDS + [X],
                frontier_keys=[X],
                use_frontier=True,
                allow_self_mixtures=False,
                cap=10,
            )
        )
        self.assertEqual(
            out, [tuple(sorted((SEEDS[0], X))), tuple(sorted((SEEDS[1], X)))]
        )

    def test_iter_mixtures_arityk_generates_unique_sorted_tuples(self):
        out = list(
            iter_mixtures_arityk(
                pool_keys=SEEDS + [X, Y, Z],
                frontier_keys=[Y],
                use_frontier=True,
                allow_self_mixtures=False,
                arity=3,
                cap=10,
            )
        )
        self.assertTrue(all(tuple(sorted(t)) == t for t in out))
        self.assertTrue(all(Y in t for t in out))
        self.assertEqual(len(out), len(set(out)))
