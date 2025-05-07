import unittest
import importlib.util
from synkit.IO.chem_converter import smart_to_gml
from synkit.Synthesis.reactor_utils import (
    _get_connected_subgraphs,
    _get_reagent,
    _get_reagent_rsmi,
    _add_reagent,
    _remove_reagent,
    _calculate_max_depth,
    _find_all_paths,
    _get_unique_aam,
)

from synkit.Synthesis.MSR.multi_steps import MultiSteps

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


class TestReactorUtils(unittest.TestCase):
    def setUp(self) -> None:
        smarts = [
            "[CH2:4]([CH:5]=[O:6])[H:7]>>[CH2:4]=[CH:5][O:6][H:7]",
            (
                "[CH2:2]=[O:3].[CH2:4]=[CH:5][O:6][H:7]>>[CH2:2]([O:3][H:7])[CH2:4]"
                + "[CH:5]=[O:6]"
            ),
            "[CH2:4]([CH:5]=[O:6])[H:8]>>[CH2:4]=[CH:5][O:6][H:8]",
            (
                "[CH2:2]([OH:3])[CH:4]=[CH:5][O:6][H:8]>>[CH2:2]=[CH:4][CH:5]=[O:6]"
                + ".[OH:3][H:8]"
            ),
        ]
        self.gml = [smart_to_gml(value) for value in smarts]
        self.order = [0, 1, 0, -1]
        self.rsmi = "CC=O.CC=O.CCC=O>>CC=O.CC=C(C)C=O.O"

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_calculate_max_depth(self):
        _, reaction_tree = MultiSteps._process(self.gml, self.order, self.rsmi)
        max_depth = _calculate_max_depth(reaction_tree)
        self.assertEqual(max_depth, 4)

    @unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
    def test_find_all_paths(self):
        results, reaction_tree = MultiSteps._process(self.gml, self.order, self.rsmi)
        target_products = sorted(self.rsmi.split(">>")[1].split("."))
        max_depth = len(results)
        all_paths = _find_all_paths(
            reaction_tree, target_products, self.rsmi, max_depth
        )
        self.assertEqual(len(all_paths), 1)
        real_path = all_paths[0][1:]  # remove the original reaction
        self.assertEqual(len(real_path), 4)

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
