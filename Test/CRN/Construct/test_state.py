from __future__ import annotations

import unittest

from synkit.CRN.Construct.state import DerivationState

from ._case import SEEDS, X, Y, Z


class TestState(unittest.TestCase):
    def test_set_initial_and_advance(self):
        state = DerivationState()
        state.set_initial(pool_keys=set(SEEDS), frontier_keys={SEEDS[0]})
        self.assertEqual(state.pool_keys, set(SEEDS))
        self.assertEqual(state.frontier_keys, {SEEDS[0]})
        self.assertEqual(state.step, 0)

        state.begin_step(1)
        self.assertEqual(state.step, 1)
        state.advance({X, Y, Z})
        self.assertEqual(state.frontier_keys, {X, Y, Z})
        self.assertEqual(state.step, 2)
