import os
import glob
import unittest
import importlib.util
from synkit.IO.data_io import load_gml_as_text
from synkit.Rule.Compose.rule_compose import RuleCompose

if importlib.util.find_spec("mod"):
    from mod import ruleGMLString

    MOD_AVAILABLE = True
else:
    print("Optional 'mod' package not found")
    MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestRuleCompose(unittest.TestCase):

    def setUp(self):
        self.single_rule_path = "Data/Testcase/Compose/SingleRule/R0"

        self.compose_rule_path = "Data/Testcase/Compose/ComposeRule"
        self.rule_1 = ruleGMLString(load_gml_as_text(f"{self.single_rule_path}/1.gml"))
        self.rule_2 = ruleGMLString(load_gml_as_text(f"{self.single_rule_path}/2.gml"))
        self.rule_compose = RuleCompose()

    def test_compose(self):
        list_rule = self.rule_compose._compose(self.rule_1, self.rule_2)
        self.assertEqual(len(list_rule), 1)
        self.assertIn("ruleID", list_rule[0].getGMLString())

    def test_process_compose(self):
        list_rule = self.rule_compose._process_compose(
            1, 2, self.single_rule_path, None
        )
        self.assertEqual(len(list_rule), 1)
        self.assertIn("ruleID", list_rule[0].getGMLString())

    def test_auto_compose(self):
        files_before = set(glob.glob(os.path.join(self.compose_rule_path, "*.gml")))

        self.rule_compose._auto_compose(self.single_rule_path, self.compose_rule_path)

        # Check for new *.gml files in compose_rule_path
        files_after = set(glob.glob(os.path.join(self.compose_rule_path, "*.gml")))
        new_files = files_after - files_before

        # Check if there are new files created
        self.assertTrue(
            new_files, "No new .gml files were created in the" + " compose_rule_path."
        )
        for file_path in glob.glob(os.path.join(self.compose_rule_path, "*.gml")):
            os.remove(file_path)
            print(f"Removed {file_path}")


if __name__ == "__main__":
    unittest.main()
