import unittest
from synkit.IO.data_io import load_database
from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO.chem_converter import smart_to_gml
from synkit.Synthesis.reactor_utils import _add_reagent, _find_all_paths
from synkit.Synthesis.MSR.multi_steps import MultiSteps
import importlib.util

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
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
        self.data = load_database("Data/Testcase/mech.json.gz")
        self.rsmi_2 = Standardize().fit(self.data[0]["reaction"])

    def test_perform_multi_step_reaction(self):
        results, _ = MultiSteps._process(self.gml, self.order, self.rsmi)
        self.assertEqual(len(results), 4)

    def test_get_aam(self):
        test = self.data[0]["mechanisms"][2]
        rule = [
            smart_to_gml(value["smart_string"], core=True, explicit_hydrogen=True)
            for value in test["steps"]
        ]
        order = list(range(len(rule)))
        rsmi = _add_reagent(self.rsmi, [test["cat"]])

        results, reaction_tree = MultiSteps._process(rule, order, rsmi)
        target_products = sorted(rsmi.split(">>")[1].split("."))
        max_depth = len(results)
        all_paths = _find_all_paths(reaction_tree, target_products, rsmi, max_depth)
        real_path = all_paths[0][1:]  # remove the original
        all_steps = MultiSteps._get_aam(real_path, rule, order)
        self.assertTrue(
            all(m is not None for m in all_steps),
            "All mechanism steps should have valid mappings",
        )

        self.assertEqual(len(all_steps), len(test["steps"]))

    def test_multi_steps(self):
        test = self.data[0]["mechanisms"][2]
        rule = [
            smart_to_gml(value["smart_string"], core=True, explicit_hydrogen=True)
            for value in test["steps"]
        ]
        order = list(range(len(rule)))
        all_steps = MultiSteps().multi_step(self.rsmi, rule, order, test["cat"])
        self.assertTrue(
            all(m is not None for m in all_steps),
            "All mechanism steps should have valid mappings",
        )

        self.assertEqual(len(all_steps), len(test["steps"]))


if __name__ == "__main__":
    unittest.main()
