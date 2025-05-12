import unittest
from rdkit import Chem
from synkit.Chem.Reaction.aam_utils import enumerate_tautomers, mapping_success_rate


class TestChemUtils(unittest.TestCase):
    def test_enumerate_tautomers_simple(self):
        # A simple keto-enol tautomerism: acetylacetone (CC(=O)CC=O) -> same product
        reaction = "CC(=O)CC=O>>O"
        tautomers = enumerate_tautomers(reaction)
        # Should return a list with at least the original reaction
        self.assertIsInstance(tautomers, list)
        self.assertIn(reaction, tautomers)
        # Each entry should be a valid reaction SMILES
        for rsmi in tautomers:
            self.assertIsInstance(rsmi, str)
            parts = rsmi.split(">>")
            self.assertEqual(len(parts), 2)
            # Reactant and product part parseable by RDKit
            self.assertIsNotNone(Chem.MolFromSmiles(parts[0]))
            self.assertIsNotNone(Chem.MolFromSmiles(parts[1]))

    def test_enumerate_tautomers_invalid(self):
        # Invalid SMILES input
        bad = "INVALID>>SMILES"
        result = enumerate_tautomers(bad)
        # Should return list with original
        self.assertEqual(result, [bad])

    def test_mapping_success_rate_normal(self):
        data = ["C:1CC", "CCC", "O:3=O", ":5", "N"]
        rate = mapping_success_rate(data)
        # Entries with mapping: 'C:1CC', 'O:3=O', ':5' => 3/5 = 60.0%
        self.assertEqual(rate, 60.0)

    def test_mapping_success_rate_empty(self):
        with self.assertRaises(ValueError):
            mapping_success_rate([])

    def test_mapping_success_rate_all(self):
        data = [":1C", ":2", "N:3"]
        rate = mapping_success_rate(data)
        self.assertEqual(rate, 100.0)

    def test_mapping_success_rate_none(self):
        data = ["C", "O", "N"]
        rate = mapping_success_rate(data)
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()
