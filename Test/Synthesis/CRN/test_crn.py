import unittest
from synkit.IO.data_io import load_database
from synkit.Synthesis.CRN.crn import CRN
import importlib.util

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestCRN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Heavy I/O only once
        cls.rules = load_database("Data/Testcase/para_rule.json.gz")

    def setUp(self):
        self.start_smiles = [
            "c1ccccc1",
            "ClCl",
            "O[Na]",
            "O=[N+]([O-])O",
            "[H][H].[H][H].[H][H]",
            "CC(=O)Cl",
        ]
        # Auto‑builds CRN on construction
        self.crn = CRN(self.rules, self.start_smiles, n_repeats=2)

    # ------------------------------------------------------------------ basic
    def test_initialisation(self):
        self.assertEqual(self.crn.initial_smiles, self.start_smiles)
        self.assertEqual(self.crn.rule_list, self.rules)
        self.assertEqual(self.crn.n_repeats, 2)
        # rounds are auto‑built
        self.assertEqual(len(self.crn.rounds), 2)

    # ------------------------------------------------------------------ rounds
    def test_product_sets_structure(self):
        ps = self.crn.product_sets  # Dict[str, List[str]]
        self.assertEqual(set(ps.keys()), {"Round 1", "Round 2"})
        # ensure each value is a list of reaction strings containing ">>"
        for rxn_list in ps.values():
            self.assertTrue(all(">>" in r for r in rxn_list))

    # ------------------------------------------------------------------ final pool
    def test_final_smiles_pool(self):
        final_pool = self.crn.final_smiles
        # starting molecules should be included
        self.assertTrue(set(self.start_smiles).issubset(final_pool))
        # should have grown
        self.assertGreater(len(final_pool), len(self.start_smiles))


if __name__ == "__main__":
    unittest.main()
