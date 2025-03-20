import unittest
from synkit.IO.data_io import load_database
from synkit.Reactor.crn import CRN


class TestCRN(unittest.TestCase):
    def setUp(self):
        # Define sample data for the tests
        self.rules = load_database("Data/Testcase/para_rule.json.gz")
        self.smiles = [
            "c1ccccc1",
            "ClCl",
            "O[Na]",
            "O=[N+]([O-])O",
            "[H][H].[H][H].[H][H]",
            "CC(=O)Cl",
        ]
        self.crn_instance = CRN(
            rule_list=self.rules, smiles_list=self.smiles, n_repeats=2
        )

    def test_initialization(self):
        # Test the __init__ method

        self.assertEqual(self.crn_instance.rule_list, self.rules)
        self.assertEqual(self.crn_instance.smiles_list, self.smiles)
        self.assertEqual(self.crn_instance.n_repeats, 2)

    def test_update_smiles(self):
        # Test the static method update_smiles
        updated_smiles = CRN.update_smiles(
            list_smiles=["C", "CC"],
            solution=["C>>CC.C", "CC>>CCC"],
            prune=False,
            starting_compound="C",
            target_compound="CCC",
        )
        self.assertIn("CC", updated_smiles)
        self.assertIn("CCC", updated_smiles)
        self.assertEqual(len(updated_smiles), 3)

    def test_expand(self):
        expanded = self.crn_instance._expand(self.rules[0:1], self.smiles)
        print(expanded)
        self.assertIn("Clc1ccccc1", expanded[1])

    def test_build_crn(self):
        # This will use the _expand and update_smiles methods indirectly
        smiles, solutions = self.crn_instance._build_crn(
            "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"
        )
        self.assertTrue(isinstance(solutions, list))
        self.assertTrue(isinstance(smiles, list))
        self.assertEqual(len(solutions), self.crn_instance.n_repeats)


if __name__ == "__main__":
    unittest.main()
