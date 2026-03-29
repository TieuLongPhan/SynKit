from __future__ import annotations

import unittest

from synkit.CRN.Construct.derivation import DerivationLog, DerivationRecord
from synkit.CRN.Construct.builder import CRNExpand
import synkit.CRN.Construct.builder as builder_mod

from ._case import RULES, SEEDS, patch_builder_chemistry


class TestDerivation(unittest.TestCase):
    def test_append_and_as_dicts(self):
        log = DerivationLog()
        log.append(
            event_id=7,
            label="r@1@2",
            step=2,
            rule_index=1,
            reactants=("C=O", "s_enol_1"),
            products=("s_adduct_1",),
        )
        self.assertEqual(
            log.records,
            [
                DerivationRecord(
                    event_id=7,
                    label="r@1@2",
                    step=2,
                    rule_index=1,
                    reactants=("C=O", "s_enol_1"),
                    products=("s_adduct_1",),
                )
            ],
        )
        self.assertEqual(log.as_dicts()[0]["products"], ["s_adduct_1"])

    def test_builder_derivation_records_are_serializable(self):
        dg = CRNExpand(
            rules=RULES,
            repeats=2,
            explicit_h=False,
            implicit_temp=False,
            keep_aam=False,
            use_frontier=True,
            dedup_delta=True,
        )
        with patch_builder_chemistry(builder_mod):
            dg.build(seeds=SEEDS, parallel=False)
        self.assertEqual(len(dg.derivation_records), 4)
        first = dg.derivation_records[0]
        self.assertEqual(first["step"], 1)
        self.assertIn("reactants", first)
        self.assertIn("products", first)
