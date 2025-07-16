import unittest
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Chem.Molecule.standardize import fix_radical_rsmi
from synkit.Rule.Apply.rule_matcher import RuleMatcher
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.aam_validator import AAMValidator


class TestRuleRBL(unittest.TestCase):

    # def setUp(self):
    #     """Setup for all tests."""
    #     input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
    #     rule = rsmi_to_its(input_rsmi, core=True)
    #     rsmi = Standardize().fit(input_rsmi)
    #     expected_rsmi = (
    #         "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
    #         + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
    #     )
    def test_rule_match_balance(self):

        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        rule = rsmi_to_its(input_rsmi, core=True)
        rsmi = Standardize().fit(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        matcher = RuleMatcher(rsmi, rule)
        result = matcher.get_result()
        self.assertTrue(AAMValidator.smiles_check(result[0], expected_rsmi, "ITS"))
        self.assertIsNotNone(result[1])

    def test_rbl_missing_product(self):
        """Test rbl method with missing product scenario."""
        rsmi = "CC(Br)C.CB(O)O>>CC(C)C"
        template = "[CH3:1][Br:2].[BH2:3][CH3:4]>>[CH3:1][CH3:4].[BH2:3][Br:2]"
        matcher = RuleMatcher(rsmi, template)
        result = matcher.get_result()
        expect = "CB(O)O.CC(C)Br>>CC(C)C.OB(O)Br"
        self.assertEqual(Standardize().fit(result[0]), expect)

    def test_rbl_missing_reactant(self):
        """Test rbl method with missing reactant scenario."""
        rsmi = "CCC(=O)(O)>>CCC(=O)OC.O"
        template = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]>>[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        matcher = RuleMatcher(rsmi, template)
        result = matcher.get_result()
        expect = "CCC(=O)O.CO>>CCC(=O)OC.O"
        self.assertEqual(Standardize().fit(result[0]), expect)


if __name__ == "__main__":
    unittest.main()
