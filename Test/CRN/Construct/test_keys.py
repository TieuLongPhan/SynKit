from __future__ import annotations

import unittest

from synkit.CRN.Construct.keys import make_dedup_key


class TestKeys(unittest.TestCase):
    def test_make_dedup_key_keeps_rule_index_by_default(self):
        key = make_dedup_key(
            dedup_across_rules=False,
            rule_index=1,
            r_keep_keys=("C=O", "s_enol_1"),
            p_keep_keys=("s_adduct_1",),
        )
        self.assertEqual(key, (1, ("C=O", "s_enol_1"), ("s_adduct_1",)))

    def test_make_dedup_key_drops_rule_index_when_requested(self):
        key = make_dedup_key(
            dedup_across_rules=True,
            rule_index=1,
            r_keep_keys=("C=O", "s_enol_1"),
            p_keep_keys=("s_adduct_1",),
        )
        self.assertEqual(key, (None, ("C=O", "s_enol_1"), ("s_adduct_1",)))
