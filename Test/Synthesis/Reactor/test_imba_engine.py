import unittest
from synkit.IO import rsmi_to_its
from synkit.Graph.Wildcard.wildcard import WildCard
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Synthesis.Reactor.imba_engine import ImbaEngine


class TestImbaEngine(unittest.TestCase):
    def setUp(self):
        # A complex standardized RSMI from your example
        self.smart = (
            "[cH:1]1[cH:14][c:10]2[c:23]([cH:11][n:25]1)[cH:17][cH:12][cH:4][c:31]2[NH2:28]."
            "[cH:2]1[c:20]([C:22]([OH:7])=[O:21])[s:18][c:24]([S:6][c:29]2[c:15]"
            "([Cl:26])[cH:8][n:19][cH:9][c:16]2[Cl:27])[c:30]1[N+:5]([O-:3])=[O:13]>>"
            "[cH:1]1[cH:14][c:10]2[c:23]([cH:11][n:25]1)[cH:17][cH:12][cH:4]"
            "[c:31]2[NH:28][C:22]([c:20]1[cH:2][c:30]([N+:5]([O-:3])=[O:13])[c:24]([S:6]"
            "[c:29]2[c:15]([Cl:26])[cH:8][n:19][cH:9][c:16]2[Cl:27])[s:18]1)=[O:21]"
        )
        # Standardize removes AAM
        self.rsmi = Standardize().fit(self.smart, remove_aam=True)

    def test_pipeline_forward(self):
        """Test forward ImbaEngine pipeline end-to-end."""
        # Apply wildcard insertion
        wild_smart = WildCard().rsmi_with_wildcards(self.smart)
        # Build ITS graphs
        temp = rsmi_to_its(wild_smart, core=True)
        # substrate split from standardized RSMI
        substrate_r, _ = self.rsmi.split(">>")
        # Run engine forward with template without cleaning fragments
        engine = ImbaEngine(substrate_r, temp, add_wildcard=True, clean_fragments=False)
        out = engine.smarts_list
        self.assertEqual(len(out), 1)
        out_rsmi = Standardize().fit(out[0], remove_aam=True)

        self.assertIn("*", out_rsmi)
        self.assertNotEqual(out_rsmi, self.rsmi)

        # Run engine forward with template with cleaning fragments
        engine = ImbaEngine(substrate_r, temp, add_wildcard=True, clean_fragments=True)
        out = engine.smarts_list
        outs = [Standardize().fit(o, remove_aam=True) for o in out]
        self.assertIn(self.rsmi, outs)

    def test_pipeline_backward(self):
        """Test backward ImbaEngine pipeline end-to-end with and without fragment cleaning."""
        # Prepare wildcard and ITS template
        wild_rsmi = WildCard().rsmi_with_wildcards(self.smart)
        its = rsmi_to_its(wild_rsmi, core=True)

        _, substrate_p = self.rsmi.split(">>")

        # 1. Without fragment cleaning
        engine = ImbaEngine(
            substrate_p,
            its,
            add_wildcard=True,
            clean_fragments=False,
            invert=True,
            partial=True,
        )
        out = engine.smarts_list
        self.assertEqual(len(out), 2)
        out_rsmi = Standardize().fit(out[0], remove_aam=True)
        self.assertIn("*", out_rsmi)
        self.assertNotEqual(out_rsmi, self.rsmi)

        # 2. With fragment cleaning
        engine_clean = ImbaEngine(
            substrate_p,
            its,
            add_wildcard=True,
            clean_fragments=True,
            invert=True,
            partial=True,
        )

        out_clean = engine_clean.smarts_list
        self.assertEqual(len(out_clean), 2)
        outs = [Standardize().fit(o, remove_aam=True) for o in out_clean]
        self.assertIn(self.rsmi, outs)

    def test_invalid_rsmi(self):
        """Invalid RSMI pipeline should raise an exception at Standardize or ITS step."""
        # Standardize should fail for invalid RSMI
        with self.assertRaises(Exception):
            # Attempt full pipeline
            rsmi = Standardize().fit("not_a_rsmi", remove_aam=True)
            wild = WildCard().rsmi_with_wildcards(rsmi)
            _ = rsmi_to_its(wild, core=True)


if __name__ == "__main__":
    unittest.main()
