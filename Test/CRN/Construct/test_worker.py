from __future__ import annotations

import unittest
from unittest.mock import patch

from synkit.CRN.Construct.worker import apply_rule_worker
import synkit.CRN.Construct.worker as worker_mod

from ._case import A, RULES, FakeReactor, X, Y


class TestWorker(unittest.TestCase):
    def test_apply_rule_worker_uses_synreactor_and_deduplicates_outputs(self):
        args = (
            1,
            RULES[1],
            ".".join(sorted((A, X))),
            False,
            False,
            None,
            tuple(sorted((A, X))),
        )
        with patch.object(worker_mod, "SynReactor", FakeReactor):
            idx, reactants, products = apply_rule_worker(args)

        self.assertEqual(idx, 1)
        self.assertEqual(reactants, tuple(sorted((A, X))))
        self.assertEqual(products, [Y])
        self.assertEqual(FakeReactor.last_kwargs["template"], RULES[1])
        self.assertTrue(FakeReactor.last_kwargs["automorphism"])

    def test_apply_rule_worker_passes_optional_strategy(self):
        args = (
            1,
            RULES[1],
            ".".join(sorted((A, X))),
            False,
            False,
            "demo-strategy",
            tuple(sorted((A, X))),
        )
        with patch.object(worker_mod, "SynReactor", FakeReactor):
            apply_rule_worker(args)
        self.assertEqual(FakeReactor.last_kwargs["strategy"], "demo-strategy")
