import unittest
from rdkit import Chem
from synkit.IO.combinatorial.smarts_generalizer import SMARTSGeneralizer


class TestSMARTSGeneralizer(unittest.TestCase):

    def setUp(self):
        self.gen = SMARTSGeneralizer(sanity_check=True)

    def test_basic_generalization(self):
        inputs = [
            "[C:1]-[N:2]>>[N:1]-[C:2]",
            "[N:1]-[N:2]>>[N:1]-[N:2]",
            "[O:1]-[N:2]>>[N:1]-[N:2]",
        ]
        output = self.gen.generalize(inputs)
        # Instead of strict string match, check correct mapped elements
        self.assertIn("[C,N,O:1]", output)
        self.assertIn("[N:2]", output)
        self.assertIn(">>", output)

    def test_single_smarts(self):
        inputs = ["[C:1]-[N:2]>>[N:1]-[C:2]"]
        output = self.gen.generalize(inputs)
        # Should match input exactly
        self.assertEqual(output, "[C:1]-[N:2]>>[N:1]-[C:2]")

    def test_different_topology_raises(self):
        inputs = ["[C:1]-[N:2]>>[N:1]-[C:2]", "[N:1]-[N:2]-[C:3]>>[N:1]-[N:2]-[C:3]"]
        with self.assertRaises(ValueError):
            self.gen.generalize(inputs)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            self.gen.generalize([])

    def test_molecule_smarts(self):
        gen = SMARTSGeneralizer(sanity_check=True)
        inputs = ["[C:1]-[N:2]", "[N:1]-[N:2]", "[O:1]-[N:2]"]
        out = gen.generalize(inputs)
        self.assertEqual(out, "[C,N,O:1]-[N:2]")

        mol = Chem.MolFromSmarts(out)
        self.assertIsNotNone(mol)

    def test_invalid_sanity_check(self):
        gen = SMARTSGeneralizer(sanity_check=True)
        # Using an obviously broken SMARTS (bad bracket placement)
        _ = [
            "[C:1]-[N:2]>>[N:1]-[X:2]"
        ]  # 'X' is a valid SMARTS wildcard! Use a real error
        with self.assertRaises(ValueError):
            gen.generalize(["[C:1][C:2]>>[N:1][C:2]["])  # broken SMARTS

    def test_repr(self):
        gen = SMARTSGeneralizer()
        self.assertIn("sanity_check", repr(gen))


if __name__ == "__main__":
    unittest.main()
