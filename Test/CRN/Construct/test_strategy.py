from __future__ import annotations

import unittest

from synkit.CRN.Construct.strategy import FrontierStrategy

from ._case import SEEDS, X, Y, Z


class TestStrategy(unittest.TestCase):
    def test_frontier_strategy_dispatches_arity1(self):
        strat = FrontierStrategy()
        out = list(
            strat.iter_mixtures(
                pool_keys=SEEDS + [X],
                frontier_keys=[X],
                arity=1,
                use_frontier=True,
                allow_self_mixtures=False,
                cap=10,
                max_components=3,
            )
        )
        self.assertEqual(out, [(X,)])

    def test_frontier_strategy_dispatches_arity2(self):
        strat = FrontierStrategy()
        out = list(
            strat.iter_mixtures(
                pool_keys=SEEDS + [X],
                frontier_keys=[X],
                arity=2,
                use_frontier=True,
                allow_self_mixtures=False,
                cap=10,
                max_components=3,
            )
        )
        self.assertEqual(
            out, [tuple(sorted((SEEDS[0], X))), tuple(sorted((SEEDS[1], X)))]
        )

    def test_frontier_strategy_dispatches_arityk(self):
        strat = FrontierStrategy()
        out = list(
            strat.iter_mixtures(
                pool_keys=SEEDS + [X, Y, Z],
                frontier_keys=[Y],
                arity=3,
                use_frontier=True,
                allow_self_mixtures=False,
                cap=10,
                max_components=3,
            )
        )
        self.assertTrue(out)
        self.assertTrue(all(Y in t for t in out))
