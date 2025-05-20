import unittest
from synkit.Chem.Reaction.aam_validator import AAMValidator
from synkit.Graph.ITS.its_expand import ITSExpand


class TestPartialExpand(unittest.TestCase):

    def test_expand_aam_with_its(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        output_rsmi = ITSExpand.expand_aam_with_its(input_rsmi, use_G=True)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        print(output_rsmi)
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))


if __name__ == "__main__":
    unittest.main()
