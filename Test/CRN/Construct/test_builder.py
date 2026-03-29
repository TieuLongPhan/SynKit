from __future__ import annotations

import unittest

from synkit.CRN.Construct.builder import CRNExpand, build_crn_from_smarts
import synkit.CRN.Construct.builder as builder_mod

from ._case import EXPECTED_COUNTS, RULES, SEEDS, patch_builder_chemistry


class TestBuilder(unittest.TestCase):
    def _build(self, repeats: int):
        dg = CRNExpand(
            rules=RULES,
            repeats=repeats,
            explicit_h=False,
            implicit_temp=False,
            keep_aam=False,
            use_frontier=True,
            dedup_delta=True,
        )
        with patch_builder_chemistry(builder_mod):
            crn = dg.build(seeds=SEEDS, parallel=False)
        return dg, crn

    def test_repeat_2_matches_expected_graph_size(self):
        dg, crn = self._build(2)
        self.assertEqual(crn.number_of_nodes(), EXPECTED_COUNTS[2]["nodes"])
        self.assertEqual(crn.number_of_edges(), EXPECTED_COUNTS[2]["edges"])
        self.assertEqual(len(dg.species_nodes), EXPECTED_COUNTS[2]["species"])
        self.assertEqual(len(dg.rxn_nodes), EXPECTED_COUNTS[2]["rules"])
        self.assertEqual(len(dg.derivation_records), EXPECTED_COUNTS[2]["rules"])

    def test_repeat_3_matches_expected_graph_size(self):
        dg, crn = self._build(3)
        self.assertEqual(crn.number_of_nodes(), EXPECTED_COUNTS[3]["nodes"])
        self.assertEqual(crn.number_of_edges(), EXPECTED_COUNTS[3]["edges"])
        self.assertEqual(len(dg.species_nodes), EXPECTED_COUNTS[3]["species"])
        self.assertEqual(len(dg.rxn_nodes), EXPECTED_COUNTS[3]["rules"])

    def test_repeat_4_matches_expected_graph_size(self):
        dg, crn = self._build(4)
        self.assertEqual(crn.number_of_nodes(), EXPECTED_COUNTS[4]["nodes"])
        self.assertEqual(crn.number_of_edges(), EXPECTED_COUNTS[4]["edges"])
        self.assertEqual(len(dg.species_nodes), EXPECTED_COUNTS[4]["species"])
        self.assertEqual(len(dg.rxn_nodes), EXPECTED_COUNTS[4]["rules"])

    def test_graph_node_and_edge_metadata_are_consistent(self):
        dg, crn = self._build(2)
        for nid, data in crn.nodes(data=True):
            self.assertIn(data.get("kind"), {"species", "rule"})
            if data.get("kind") == "species":
                self.assertEqual(data["label"], data["smiles"])
            else:
                self.assertIn("rule_index", data)
                self.assertIn("step", data)
        for _, _, edata in crn.edges(data=True):
            self.assertIn(edata.get("role"), {"reactant", "product"})
            self.assertGreaterEqual(int(edata.get("stoich", 0)), 1)

    def test_build_crn_from_smarts_wrapper(self):
        with patch_builder_chemistry(builder_mod):
            crn = build_crn_from_smarts(
                rules=RULES,
                seeds=SEEDS,
                repeats=3,
                explicit_h=False,
                implicit_temp=False,
                keep_aam=False,
                parallel=False,
                use_frontier=True,
                dedup_delta=True,
            )
        self.assertEqual(crn.number_of_nodes(), EXPECTED_COUNTS[3]["nodes"])
        self.assertEqual(crn.number_of_edges(), EXPECTED_COUNTS[3]["edges"])
