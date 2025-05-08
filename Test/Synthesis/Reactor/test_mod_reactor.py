import os
import unittest
import tempfile
import importlib.util
from synkit.IO.chem_converter import smart_to_gml, gml_to_smart
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Synthesis.Reactor.mod_reactor import MODReactor


MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestMODReactor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

        # Path for the rule file
        self.rule_file_path = os.path.join(self.temp_dir.name, "test_rule.gml")

        # Define rule content
        self.rule_content = """
        rule [
           ruleID "1"
           left [
              edge [ source 1 target 2 label "=" ]
              edge [ source 3 target 4 label "-" ]
           ]
           context [
              node [ id 1 label "C" ]
              node [ id 2 label "C" ]
              node [ id 3 label "H" ]
              node [ id 4 label "H" ]
           ]
           right [
              edge [ source 1 target 2 label "-" ]
              edge [ source 1 target 3 label "-" ]
              edge [ source 2 target 4 label "-" ]
           ]
        ]
        """
        # self.smart = gml_to_smart(self.rule_content)

        # Write rule content to the temporary file
        with open(self.rule_file_path, "w") as rule_file:
            rule_file.write(self.rule_content)

        # Initialize SMILES strings for testing
        self.initial_smiles_fw = ["CC=CC", "[H][H]"]
        self.initial_smiles_bw = ["CCCC"]

        test_2 = (
            "[CH3:1][O:2][C:3](=[O:4])[CH:5]([CH2:6][CH2:7][CH2:8]"
            + "[CH2:9][NH:10][C:11](=[O:12])[O:13][CH2:14][c:15]1[cH:16]"
            + "[cH:17][cH:18][cH:19][cH:20]1)[NH:21][C:22](=[O:23])[NH:24]"
            + "[c:25]1[cH:26][c:27]([O:28][CH3:29])[cH:30][c:31]([C:32]([CH3:33])"
            + "([CH3:34])[CH3:35])[c:36]1[OH:37].[OH:38][H:39]>>[C:11](=[O:12])"
            + "([O:13][CH2:14][c:15]1[cH:16][cH:17][cH:18][cH:19][cH:20]1)[OH:38]"
            + ".[CH3:1][O:2][C:3](=[O:4])[CH:5]([CH2:6][CH2:7][CH2:8][CH2:9][NH:10]"
            + "[H:39])[NH:21][C:22](=[O:23])[NH:24][c:25]1[cH:26][c:27]([O:28]"
            + "[CH3:29])[cH:30][c:31]([C:32]([CH3:33])([CH3:34])[CH3:35])[c:36]1[OH:37]"
        )
        self.rsmi = Standardize().fit(test_2)
        self.gml_2 = smart_to_gml(test_2, sanitize=True, core=True)
        self.smart_2 = gml_to_smart(self.gml_2)

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_forward(self):
        reactor = MODReactor(
            self.initial_smiles_fw, self.rule_file_path, invert=False, strategy="comp"
        )
        reactor.run()
        result = reactor.get_reaction_smiles()
        self.assertIsInstance(
            result, list, "Expected a list of reaction SMILES strings."
        )
        self.assertTrue(
            len(result) > 0, "Result should contain reaction SMILES strings."
        )

        self.assertEqual(result[0], "CC=CC.[H][H]>>CCCC")

    def test_backward(self):
        reactor = MODReactor(
            self.initial_smiles_bw, self.rule_file_path, invert=True, strategy="comp"
        )
        reactor.run()
        result = reactor.get_reaction_smiles()
        result = [Standardize().fit(i) for i in result]
        self.assertIsInstance(
            result, list, "Expected a list of reaction SMILES strings."
        )
        self.assertTrue(
            len(result) > 0, "Result should contain reaction SMILES strings."
        )

        self.assertIn("CC=CC.[H][H]>>CCCC", result)

    def test_fw_all(self):
        reactor = MODReactor(
            self.rsmi.split(">>")[0], self.smart_2, invert=False, strategy="all"
        )
        reactor.run()
        result = reactor.get_reaction_smiles()
        print(result)
        self.assertIsInstance(
            result, list, "Expected a list of reaction SMILES strings."
        )
        self.assertEqual(len(result), 12)

    def test_fw_comp(self):
        reactor = MODReactor(
            self.rsmi.split(">>")[0], self.gml_2, invert=False, strategy="comp"
        )
        reactor.run()
        result = reactor.get_reaction_smiles()
        print(result)
        self.assertIsInstance(
            result, list, "Expected a list of reaction SMILES strings."
        )
        self.assertEqual(len(result), 6)

    def test_bw_all(self):
        reactor = MODReactor(
            self.rsmi.split(">>")[1], self.gml_2, invert=True, strategy="all"
        )
        reactor.run()
        result = reactor.get_reaction_smiles()
        print(result)
        self.assertIsInstance(
            result, list, "Expected a list of reaction SMILES strings."
        )
        self.assertEqual(len(result), 24)

    def test_bw_bt(self):
        reactor = MODReactor(
            self.rsmi.split(">>")[1], self.gml_2, invert=True, strategy="bt"
        )
        reactor.run()
        result = reactor.get_reaction_smiles()
        print(result)
        self.assertIsInstance(
            result, list, "Expected a list of reaction SMILES strings."
        )
        self.assertEqual(len(result), 9)


if __name__ == "__main__":
    unittest.main()
