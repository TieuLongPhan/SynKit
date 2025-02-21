import unittest
from synkit.IO.chem_converter import smart_to_gml
from synkit.Reactor.multi_step import (
    perform_multi_step_reaction,
    calculate_max_depth,
    find_all_paths,
)


class TestMultiStep(unittest.TestCase):
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

    def test_perform_multi_step_reaction(self):
        results, _ = perform_multi_step_reaction(self.gml, self.order, self.rsmi)
        self.assertEqual(len(results), 4)

    def test_calculate_max_depth(self):
        _, reaction_tree = perform_multi_step_reaction(self.gml, self.order, self.rsmi)
        max_depth = calculate_max_depth(reaction_tree)
        self.assertEqual(max_depth, 4)

    def test_find_all_paths(self):
        results, reaction_tree = perform_multi_step_reaction(
            self.gml, self.order, self.rsmi
        )
        target_products = sorted(self.rsmi.split(">>")[1].split("."))
        max_depth = len(results)
        all_paths = find_all_paths(reaction_tree, target_products, self.rsmi, max_depth)
        self.assertEqual(len(all_paths), 1)
        real_path = all_paths[0][1:]  # remove the original reaction
        self.assertEqual(len(real_path), 4)


if __name__ == "__main__":
    unittest.main()
