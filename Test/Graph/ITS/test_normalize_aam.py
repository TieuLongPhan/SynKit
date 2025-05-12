import unittest
import networkx as nx
from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from synkit.Chem.Reaction.aam_validator import AAMValidator


class TestNormalizeAAM(unittest.TestCase):
    def setUp(self):
        """Set up for testing."""
        self.normalizer = NormalizeAAM()

    # def test_fix_atom_mapping(self):
    #     """Test that atom mappings are incremented correctly."""
    #     input_smiles = "[C:0]([H:1])([H:2])[H:3]"
    #     expected_smiles = "[C:1]([H:2])([H:3])[H:4]"
    #     self.assertEqual(
    #         self.normalizer.fix_atom_mapping(input_smiles), expected_smiles
    #     )

    # def test_fix_rsmi(self):
    #     """Test that RSMI atom mappings are incremented correctly
    #     for both reactants and products."""
    #     input_rsmi = "[C:0]>>[C:1]"
    #     expected_rsmi = "[C:1]>>[C:2]"
    #     self.assertEqual(self.normalizer.fix_aam_rsmi(input_rsmi), expected_rsmi)

    def test_extract_subgraph(self):
        """Test extraction of a subgraph based on specified indices."""
        g = nx.complete_graph(5)
        indices = [0, 1, 2]
        subgraph = self.normalizer.extract_subgraph(g, indices)
        self.assertEqual(len(subgraph.nodes()), 3)
        self.assertTrue(all(node in subgraph for node in indices))

    def test_reset_indices_and_atom_map(self):
        """Test resetting of indices and atom map in a subgraph."""
        g = nx.path_graph(5)
        for i in range(5):
            g.nodes[i]["atom_map"] = i + 1
        reset_graph = self.normalizer.reset_indices_and_atom_map(g)
        expected_atom_maps = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        for node in reset_graph:
            self.assertEqual(
                reset_graph.nodes[node]["atom_map"], expected_atom_maps[node]
            )

    def test_reaction_smiles_processing(self):
        """Test that the reaction SMILES string is processed to meet expected output."""
        input_rsmi = (
            "[C:2]([C:3]([H:9])([H:10])[H:11])([H:8])=[C:1]([C:0]([H:6])([H:5])"
            + "[H:4])[H:7].[H:12][H:13]>>[C:3]([C:2]([C:1]([C:0]([H:6])([H:5])"
            + "[H:4])([H:12])[H:7])([H:8])[H:13])([H:9])([H:10])[H:11]"
        )
        expected_output = (
            "[CH3:1][CH:2]=[CH:3][CH3:4].[H:5][H:6]>>[CH3:1][CH:2]([CH:3]"
            + "([CH3:4])[H:6])[H:5]"
        )
        result = self.normalizer.fit(input_rsmi)
        print(result)
        self.assertTrue(AAMValidator().smiles_check(result, expected_output))


# Run the unittest
if __name__ == "__main__":
    unittest.main()
