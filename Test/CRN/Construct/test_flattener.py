from __future__ import annotations

import unittest

from synkit.CRN.Construct.builder import CRNExpand
from synkit.CRN.Construct.flattener import ReactionDeltaFlattener
import synkit.CRN.Construct.builder as builder_mod

from ._case import RULES, SEEDS, patch_builder_chemistry


class TestFlattener(unittest.TestCase):
    def _build_repeat_2(self):
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
        return dg.graph

    def test_reaction_delta_flattener_builds_expected_records(self):
        graph = self._build_repeat_2()
        flat = ReactionDeltaFlattener(graph).build().reactions
        self.assertEqual(len(flat), 4)
        self.assertEqual(flat[0]["step"], 1)
        self.assertEqual(flat[0]["reactants"], ["C=O"])
        self.assertEqual(flat[0]["products"], ["s_enol_1"])
        self.assertEqual(flat[1]["rule_smiles"], "OCC=O>>C=O")
        self.assertEqual(flat[-1]["rule_smiles"], "OCC=O.s_enol_1>>s_adduct_2")

    def test_reaction_delta_flattener_sorts_by_step_rule_and_app_index(self):
        graph = self._build_repeat_2()
        flat = ReactionDeltaFlattener(graph).build().reactions
        keys = [(r["step"], r["rule_index"], r["app_index"]) for r in flat]
        self.assertEqual(keys, sorted(keys))
