from __future__ import annotations

import unittest

from synkit.CRN.Construct.arity import count_lhs_components, infer_rule_arity

from ._case import RULES


class RuleWrapper:
    def __init__(self, smirks: str):
        self.smirks = smirks


class BrokenRule:
    def __repr__(self):
        raise RuntimeError("boom")


class TestArity(unittest.TestCase):
    def test_count_lhs_components_matches_user_rules(self):
        self.assertEqual(count_lhs_components(RULES[0]), 1)
        self.assertEqual(count_lhs_components(RULES[1]), 2)
        self.assertEqual(count_lhs_components(RULES[2]), 1)
        self.assertEqual(count_lhs_components(RULES[3]), 1)

    def test_infer_rule_arity_from_string_and_object(self):
        self.assertEqual(infer_rule_arity(RULES[1]), 2)
        self.assertEqual(infer_rule_arity(RuleWrapper(RULES[1])), 2)
        self.assertEqual(infer_rule_arity(RuleWrapper(RULES[0])), 1)

    def test_infer_rule_arity_falls_back_to_default_for_bad_input(self):
        self.assertEqual(infer_rule_arity(BrokenRule()), 2)
        self.assertEqual(count_lhs_components(""), None)
