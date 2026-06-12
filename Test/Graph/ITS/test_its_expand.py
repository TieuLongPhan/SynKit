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
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))

    def test_expand_with_relabel(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        output_rsmi = ITSExpand.expand_aam_with_its(input_rsmi, relabel=True)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))

    def test_expand_preserve_sparse_atom_maps(self):
        input_rsmi = (
            "Br[C:64]1=[CH:63][CH:62]=[C:61]([S-:10])[CH:72]=[CH:73]1."
            "C[CH+:20][C:31]1=[C:32](C)[CH:33]=[C:34](C)[CH:35]=[C:36]1C"
            ">>"
            "C[CH:20]([S:10][C:61]1=[CH:62][CH:63]=[C:64](Br)[CH:73]=[CH:72]1)"
            "[C:31]1=[C:32](C)[CH:33]=[C:34](C)[CH:35]=[C:36]1C"
        )

        output_rsmi = ITSExpand.expand_aam_with_its(
            input_rsmi,
            relabel=False,
            preserve_older_map=True,
        )

        self.assertIn(":10]", output_rsmi)
        self.assertIn(":20]", output_rsmi)


if __name__ == "__main__":
    unittest.main()
