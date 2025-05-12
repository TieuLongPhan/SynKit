import unittest
import importlib.util
from synkit.IO.chem_converter import smart_to_gml
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Rule.Apply.reactor_rule import ReactorRule


MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestReactorRule(unittest.TestCase):

    def setUp(self):
        self.rsmi = "BrCc1ccc(Br)cc1.COCCO>>Br.COCCOCc1ccc(Br)cc1"
        self.gml = smart_to_gml("[Br:1][CH3:2].[OH:3][H:4]>>[Br:1][H:4].[CH3:2][OH:3]")
        self.expect_forward = (
            "[Br:1][CH2:2][C:3]1=[CH:4][CH:6]=[C:7]([Br:8])[CH:9]"
            + "=[CH:5]1.[CH3:10][O:11][CH2:12][CH2:13][O:14][H:15]>>"
            + "[Br:1][H:15].[CH2:2]([C:3]1=[CH:4][CH:6]=[C:7]([Br:8])"
            + "[CH:9]=[CH:5]1)[O:14][CH2:13][CH2:12][O:11][CH3:10]"
        )
        self.expect_backward = self.expect_forward

    def test_inference_smiles_forward(self):
        # Split the input SMILES into reactants and products
        reactants, _ = self.rsmi.split(">>")
        # Test forward reaction inference
        output_rsmis = ReactorRule()._process(reactants, self.gml)
        print(output_rsmis)
        self.assertTrue(
            any(
                AAMValidator.smiles_check(smiles, self.expect_forward, "ITS")
                for smiles in output_rsmis
            ),
            "Forward reaction test failed.",
        )

        reactants = reactants.split(".")
        for r in reactants:
            output_rsmis = ReactorRule()._process(r, self.gml)
            self.assertGreater(len(output_rsmis), 0)

    def test_inference_smiles_backward(self):
        # Split the input SMILES into reactants and products
        _, products = self.rsmi.split(">>")
        # Test forward reaction inference
        output_rsmis = ReactorRule()._process(products, self.gml, invert=True)

        self.assertTrue(
            any(
                AAMValidator.smiles_check(smiles, self.expect_forward, "ITS")
                for smiles in output_rsmis
            ),
            "Forward reaction test failed.",
        )

        products = products.split(".")
        results = []
        for r in products:
            output_rsmis = ReactorRule()._process(r, self.gml)
            results.extend(output_rsmis)
        self.assertGreater(len(results), 0)
