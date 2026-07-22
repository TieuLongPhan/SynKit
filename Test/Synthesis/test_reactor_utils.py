import unittest
from synkit.IO.chem_converter import smart_to_gml
from synkit.Synthesis.reactor_utils import (
    _get_connected_subgraphs,
    _get_reagent,
    _get_reagent_rsmi,
    _add_reagent,
    _remove_reagent,
    _get_unique_aam,
)


class TestReactorUtils(unittest.TestCase):
    def test_get_connected_subgraphs(self):
        smart = "[CH2:4]([CH:5]=[O:6])[H:8]>>[CH2:4]=[CH:5][O:6][H:8]"
        gml = smart_to_gml(smart)
        self.assertEqual(_get_connected_subgraphs(gml), 1)

    def test_get_reagents(self):
        original = ["CC=O", "O"]
        smart = "[CH2:4]([CH:5]=[O:6])[H:8]>>[CH2:4]=[CH:5][O:6][H:8]"
        reagent = _get_reagent(original, smart)
        self.assertTrue(reagent == ["O"])

    def test_get_reagents_rsmiles(self):
        smart = "[CH2:4]([CH:5]=[O:6])[H:8].O>>[CH2:4]=[CH:5][O:6][H:8].O"
        reagent = _get_reagent_rsmi(smart)
        print(reagent)
        self.assertTrue(reagent == ["O"])

    def test_add_reagent(self):
        rsmi = "CC=O.CC=O>>CC=CC=O.O"
        reagent = ["[H+]"]
        new_rsmi = _add_reagent(rsmi, reagent)
        expect = "CC=O.CC=O.[H+]>>CC=CC=O.O.[H+]"
        self.assertEqual(new_rsmi, expect)

    def test_remove_reagent(self):
        rsmi = "CC=O.CC=O.[H+]>>CC=CC=O.O.[H+]"
        new_rsmi = _remove_reagent(rsmi)
        expect = "CC=O.CC=O>>CC=CC=O.O"
        self.assertEqual(new_rsmi, expect)

    def test_get_unique_aam(self):
        aam_list = [
            "[CH2:1]=[CH2:2].[H:3][H:4]>>[CH2:1]([H:3])[CH2:2]([H:4])",
            "[CH2:2]=[CH2:3].[H:1][H:4]>>[CH2:2]([H:4])[CH2:3]([H:1])",
            "[CH2:1]=[CH2:2].[H:3][OH:4]>>[CH2:1]([H:3])[CH2:2]([OH:4])",
        ]
        new_aam = _get_unique_aam(aam_list)
        self.assertEqual(len(new_aam), 2)


if __name__ == "__main__":
    unittest.main()
