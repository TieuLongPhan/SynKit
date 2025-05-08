import unittest
import importlib.util
from synkit.IO.chem_converter import smart_to_gml
from synkit.Chem.Molecule.standardize import fix_radical_rsmi
from synkit.Rule.Apply.rule_rbl import RuleRBL

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestRuleRBL(unittest.TestCase):

    def setUp(self):
        """Setup for all tests."""
        self.rule_rbl = RuleRBL()

    def test_rbl_missing_product(self):
        """Test rbl method with missing product scenario."""
        rsmi = "CC(Br)C.CB(O)O>>CC(C)C"
        template = "[CH3:1][Br:2].[BH2:3][CH3:4]>>[CH3:1][CH3:4].[BH2:3][Br:2]"
        gml = smart_to_gml(template, core=True, explicit_hydrogen=False)
        new_rsmi = RuleRBL().rbl(rsmi, gml)
        expect = "CB(O)O.CC(C)Br>>CC(C)C.OB(O)Br"
        self.assertEqual(new_rsmi, expect)

    def test_rbl_missing_reactant(self):
        """Test rbl method with missing reactant scenario."""
        rsmi = "CCC(=O)(O)>>CCC(=O)OC.O"
        template = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]>>[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        gml = smart_to_gml(template, core=True, explicit_hydrogen=False)
        new_rsmi = RuleRBL().rbl(rsmi, gml)
        expect = "CCC(=O)O.CO>>CCC(=O)OC.O"
        self.assertEqual(new_rsmi, expect)

    def test_rbl_missing_both_rule_full(self):
        """Test rbl method with missing both side scenario but rule capture missing molecules."""
        rsmi = "CCC(=O)(O)>>CCC(=O)OC"
        template = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]>>[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        gml = smart_to_gml(template, core=True, explicit_hydrogen=False)
        new_rsmi = RuleRBL().rbl(rsmi, gml)
        new_rsmi = fix_radical_rsmi(new_rsmi)  # Temporary fix
        expect = "CCC(=O)O.CO>>CCC(=O)OC.O"
        self.assertEqual(new_rsmi, expect)

    def test_rbl_missing_both_rule_partial(self):
        """Test rbl method with missing both side scenario but rule capture partial missing molecules."""
        rsmi = "CCC(=O)OC>>CCC(=O)OCC"
        template = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][O:6][H:7]>>[CH3:1][C:2](=[O:3])[O:6][CH3:5].[H:7][OH:4]"
        gml = smart_to_gml(template, core=True, explicit_hydrogen=False)
        new_rsmi = RuleRBL().rbl(rsmi, gml)
        expect = "CCC(=O)OC.CCO>>CO.CCOC(=O)CC"
        self.assertEqual(new_rsmi, expect)


if __name__ == "__main__":
    unittest.main()
