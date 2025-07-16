import io
import unittest
from contextlib import redirect_stdout

import networkx as nx

from synkit.IO.chem_converter import rsmi_to_its
from synkit.Rule.Apply.rule_matcher import RuleMatcher
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Chem.Reaction.balance_check import BalanceReactionCheck


class TestRuleMatcher(unittest.TestCase):

    def test_rule_match_balance(self):
        """Balanced reaction should match directly and produce correct SMARTS."""
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        rule = rsmi_to_its(input_rsmi, core=True)
        rsmi_std = Standardize().fit(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )

        matcher = RuleMatcher(rsmi_std, rule)
        smarts, returned_rule = matcher.get_result()

        # The returned SMARTS should regenerate the expected RSMI via AAMValidator
        self.assertTrue(AAMValidator.smiles_check(smarts, expected_rsmi, "ITS"))
        # The returned rule graph should be isomorphic to the input rule
        self.assertTrue(nx.is_isomorphic(returned_rule, rule))

    def test_rbl_missing_product(self):
        """Partial (RBL) match when product fragments are missing in rule."""
        rsmi = "CC(Br)C.CB(O)O>>CC(C)C"
        template = "[CH3:1][Br:2].[BH2:3][CH3:4]>>[CH3:1][CH3:4].[BH2:3][Br:2]"
        matcher = RuleMatcher(rsmi, template)
        smarts, _ = matcher.get_result()
        expect = "CB(O)O.CC(C)Br>>CC(C)C.OB(O)Br"
        self.assertEqual(Standardize().fit(smarts), expect)

    def test_rbl_missing_reactant(self):
        """Partial (RBL) match when reactant fragments are missing in rule."""
        rsmi = "CCC(=O)(O)>>CCC(=O)OC.O"
        template = (
            "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]"
            ">>[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        )
        matcher = RuleMatcher(rsmi, template)
        smarts, _ = matcher.get_result()
        expect = "CCC(=O)O.CO>>CCC(=O)OC.O"
        self.assertEqual(Standardize().fit(smarts), expect)

    def test_no_match_raises(self):
        """If no SMARTS reproduces the RSMI under the rule, a ValueError is raised."""
        rsmi = "CCO>>CC=O"
        # Use a completely unrelated template
        bad_template = "[CH3:1][OH:2]>>[CH2:1]=O"
        with self.assertRaises(ValueError):
            RuleMatcher(rsmi, bad_template)

    def test_str_and_repr(self):
        """__str__ and __repr__ reflect the RSMI, balance status, and rule size."""
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        template = rsmi_to_its(input_rsmi, core=True)
        rsmi = Standardize().fit(input_rsmi)
        matcher = RuleMatcher(rsmi, template)
        # str should mention balanced/unbalanced correctly
        if BalanceReactionCheck(n_jobs=1).rsmi_balance_check(matcher.rsmi):
            self.assertIn("(balanced)", str(matcher))
        else:
            self.assertIn("(unbalanced)", str(matcher))
        # repr should include node/edge counts of the rule
        rep = repr(matcher)
        self.assertIn("RuleMatcher(rsmi=", rep)
        self.assertIn("balanced=", rep)

    def test_help_output(self):
        """help() should print internal state and list candidate SMARTS patterns."""
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        template = rsmi_to_its(input_rsmi, core=True)
        rsmi = Standardize().fit(input_rsmi)
        matcher = RuleMatcher(rsmi, template)
        buf = io.StringIO()
        with redirect_stdout(buf):
            matcher.help()
        out = buf.getvalue()
        self.assertIn("RuleMatcher for RSMI", out)
        self.assertIn("Candidate SMARTS patterns:", out)


if __name__ == "__main__":
    unittest.main()
