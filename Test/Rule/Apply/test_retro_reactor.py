import unittest
import importlib.util
from synkit.IO.data_io import load_list_from_file
from synkit.Rule.Apply.retro_reactor import RetroReactor

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestRetroReactor(unittest.TestCase):
    def setUp(self):
        self.retro_reactor = RetroReactor()
        self.rules = load_list_from_file("Data/Testcase/para_rule_retro.json.gz")

    def test_init(self):
        """Test the initialization of the RetroReactor class."""
        self.assertIsInstance(self.retro_reactor.backward_cache, dict)
        self.assertEqual(len(self.retro_reactor.backward_cache), 0)

    def test_heuristic(self):
        """Test the heuristic calculation based on the difference in carbon count."""
        # Example test cases
        self.assertEqual(
            RetroReactor()._heuristic("C1CCCCC1", "C1CC1"), 3
        )  # Direct carbon count comparison

    def test_backward_synthesis_search_simple_case(self):
        """Test backward synthesis search from benzene to aniline"""
        rf = RetroReactor()

        product_smiles = "Nc1ccccc1"
        known_precursor_smiles = "c1ccccc1"

        max_solutions = 1

        solutions = rf.backward_synthesis_search(
            product_smiles,
            known_precursor_smiles,
            self.rules,
            max_solutions=max_solutions,
            fast_process=True,
        )
        self.assertGreater(len(solutions), 0)


if __name__ == "__main__":
    unittest.main()
