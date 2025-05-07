import unittest

from synkit.IO.data_io import load_database
from synkit.Synthesis.CRN.crn import CRN
from synkit.Synthesis.MSR.path_finder import PathFinder
import importlib.util

MOD_AVAILABLE = importlib.util.find_spec("mod") is not None


@unittest.skipUnless(MOD_AVAILABLE, "requires `mod` package for rule backend")
class TestPathFinder(unittest.TestCase):
    def setUp(self):
        # Define a simple reaction round for testing
        self.reaction_rounds = [
            {"Round 1": ["A>>B.B", "B>>C", "C>>D"]},
            {"Round 2": ["B>>E", "C>>F", "D>>G"]},
        ]
        self.path_finder = PathFinder(self.reaction_rounds)
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
            rule_list=self.rules, smiles_list=self.smiles, n_repeats=3
        )

        solutions = self.crn_instance.rounds
        self.finder = PathFinder(solutions)

    def test_initialization(self):
        # Check that the initialization correctly sets the reaction rounds
        self.assertEqual(len(self.path_finder.reaction_rounds), 2)

    def test_adjacency_map(self):
        # Check that the adjacency map is built correctly
        expected_adjacency_round_1 = {
            "A": [("A>>B.B", ["B", "B"])],
            "B": [("B>>C", ["C"])],
            "C": [("C>>D", ["D"])],
        }
        self.assertEqual(self.path_finder._adjacency[0], expected_adjacency_round_1)
        expected_adjacency_round_2 = {
            "B": [("B>>E", ["E"])],
            "C": [("C>>F", ["F"])],
            "D": [("D>>G", ["G"])],
        }
        self.assertEqual(self.path_finder._adjacency[1], expected_adjacency_round_2)

    def test_search_paths_bfs(self):
        # Test BFS with limiting solutions and cheapest mode
        solutions = self.finder.search_paths(
            "c1ccccc1", "CC(=O)Nc1ccccc1", method="bfs", max_solutions=1, cheapest=True
        )
        expected_solutions = [
            [
                "O=[N+]([O-])O.c1ccccc1>>O.O=[N+]([O-])c1ccccc1",
                "O=[N+]([O-])c1ccccc1.[H][H].[H][H].[H][H]>>Nc1ccccc1.O.O",
                "CC(=O)Cl.Nc1ccccc1>>CC(=O)Nc1ccccc1.Cl",
            ]
        ]
        self.assertEqual(solutions, expected_solutions)

    def test_search_paths_astar(self):
        # Test A* with no solution limit and cheapest mode
        solutions = self.finder.search_paths(
            "c1ccccc1", "CC(=O)Nc1ccccc1", method="astar", cheapest=True
        )
        expected_solutions = [
            [
                "O=[N+]([O-])O.c1ccccc1>>O.O=[N+]([O-])c1ccccc1",
                "O=[N+]([O-])c1ccccc1.[H][H].[H][H].[H][H]>>Nc1ccccc1.O.O",
                "CC(=O)Cl.Nc1ccccc1>>CC(=O)Nc1ccccc1.Cl",
            ]
        ]
        self.assertEqual(solutions, expected_solutions)

    def test_invalid_method(self):
        # Test with an invalid method
        with self.assertRaises(ValueError):
            self.path_finder.search_paths("A", "B", method="invalid")

    def test_valid_intermediate(self):
        # Test the _valid_intermediate method
        self.assertTrue(self.path_finder._valid_intermediate("CC", "C", "CCC"))


if __name__ == "__main__":
    unittest.main()
