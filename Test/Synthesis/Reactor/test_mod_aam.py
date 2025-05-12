import unittest
import importlib.util
from synkit.IO.chem_converter import smart_to_gml
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Synthesis.Reactor.mod_aam import MODAAM


MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestMODAAM(unittest.TestCase):

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

    def test_inference_aam_expand_forward(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        gml = smart_to_gml(input_rsmi)
        rsmi = Standardize().fit(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        # Test forward
        output_rsmi = MODAAM(
            rsmi.split(">>")[0], gml, invert=False
        ).get_reaction_smiles()
        self.assertGreater(len(output_rsmi), 0)
        self.assertTrue(AAMValidator.smiles_check(output_rsmi[0], expected_rsmi, "ITS"))

    def test_inference_aam_expand_backward(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        gml = smart_to_gml(input_rsmi)
        rsmi = Standardize().fit(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        # Test forward
        output_rsmi = MODAAM(
            rsmi.split(">>")[1], gml, invert=True
        ).get_reaction_smiles()
        self.assertGreater(len(output_rsmi), 0)
        print(output_rsmi[0])
        self.assertTrue(AAMValidator.smiles_check(output_rsmi[0], expected_rsmi, "ITS"))

    def test_inference_aam_expand_with_reagent_forward(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4].O>>CC[CH2:3][NH2:2].[Cl:1][H:4].O"
        gml = smart_to_gml(input_rsmi)
        rsmi = Standardize().fit(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6].[OH2:7]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6].[OH2:7]"
        )
        # Test forward
        output_rsmi = MODAAM(
            rsmi.split(">>")[0], gml, invert=False
        ).get_reaction_smiles()
        self.assertGreater(len(output_rsmi), 0)
        self.assertTrue(AAMValidator.smiles_check(output_rsmi[0], expected_rsmi, "ITS"))

    def test_inference_aam_expand_with_reagent_backward(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4].O>>CC[CH2:3][NH2:2].[Cl:1][H:4].O"
        gml = smart_to_gml(input_rsmi)
        rsmi = Standardize().fit(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6].[OH2:7]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6].[OH2:7]"
        )
        # Test backward
        output_rsmi = MODAAM(
            rsmi.split(">>")[1], gml, invert=True
        ).get_reaction_smiles()
        self.assertGreater(len(output_rsmi), 0)
        self.assertTrue(AAMValidator.smiles_check(output_rsmi[0], expected_rsmi, "ITS"))

    def test_inference_smiles_forward(self):
        # Split the input SMILES into reactants and products
        reactants, _ = self.rsmi.split(">>")

        # Test forward reaction inference
        output_rsmis = MODAAM(reactants, self.gml, invert=False).get_reaction_smiles()
        # Validate each SMILES string in the output list against the expected results
        # Use 'in' operator to make sure that the output sequence contains expected result
        self.assertTrue(
            any(
                AAMValidator.smiles_check(smiles, self.expect_forward, "ITS")
                for smiles in output_rsmis
            ),
            "Forward reaction test failed.",
        )

    def test_inference_smiles_backward(self):
        # Split the input SMILES into reactants and products
        _, products = self.rsmi.split(">>")

        # Test backward reaction inference
        output_rsmis = MODAAM(products, self.gml, invert=True).get_reaction_smiles()
        # Validate each SMILES string in the output list against the expected results
        # Use 'in' operator to make sure that the output sequence contains expected result
        self.assertTrue(
            any(
                AAMValidator.smiles_check(smiles, self.expect_forward, "ITS")
                for smiles in output_rsmis
            ),
            "Backward reaction test failed.",
        )

    def test_inference_smiles_with_reagent_forward_with_aam(self):
        rsmi = "BrCc1ccc(Br)cc1.COCCO.O>>Br.COCCOCc1ccc(Br)cc1.O"
        # Split the input SMILES into reactants and products
        reactants, _ = rsmi.split(">>")
        expect = (
            "[Br:1][CH2:2][C:3]1=[CH:4][CH:6]=[C:7]([Br:8])[CH:9]"
            + "=[CH:5]1.[CH3:10][O:11][CH2:12][CH2:13][O:14][H:15].[OH2:16]>>"
            + "[Br:1][H:15].[CH2:2]([C:3]1=[CH:4][CH:6]=[C:7]([Br:8])"
            + "[CH:9]=[CH:5]1)[O:14][CH2:13][CH2:12][O:11][CH3:10].[OH2:16]"
        )

        # Test forward reaction inference
        output_rsmis = MODAAM(reactants, self.gml, invert=False).get_reaction_smiles()

        self.assertTrue(
            any(
                AAMValidator.smiles_check(smiles, expect, "ITS")
                for smiles in output_rsmis
            ),
            "Forward reaction test failed.",
        )

    def test_inference_smiles_with_reagent_backward_with_aam(self):
        rsmi = "BrCc1ccc(Br)cc1.COCCO.O>>Br.COCCOCc1ccc(Br)cc1.O"
        # Split the input SMILES into reactants and products
        _, products = rsmi.split(">>")
        # Test backward reaction inference
        output_rsmis = MODAAM(products, self.gml, invert=True).get_reaction_smiles()
        expected_reactants = (
            "[Br:1][CH2:2][C:3]1=[CH:4][CH:6]=[C:7]([Br:8])[CH:9]"
            + "=[CH:5]1.[CH3:10][O:11][CH2:12][CH2:13][O:14][H:15].[OH2:16]"
        )
        expected_products = (
            "[Br:1][H:15].[CH2:2]([C:3]1=[CH:4][CH:6]=[C:7]([Br:8])"
            + "[CH:9]=[CH:5]1)[O:14][CH2:13][CH2:12][O:11][CH3:10].[OH2:16]"
        )
        expected_bw_reversed = expected_reactants + ">>" + expected_products
        self.assertTrue(
            any(
                AAMValidator.smiles_check(smiles, expected_bw_reversed, "ITS")
                for smiles in output_rsmis
            ),
            "Backward reaction test failed.",
        )


if __name__ == "__main__":
    unittest.main()
