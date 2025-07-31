import unittest
from synkit.Graph.Wildcard.radwc import RadWC


class TestRadWC(unittest.TestCase):
    def test_no_product_radicals(self):
        """If product has no radicals, output should be unchanged."""
        rxn = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
        self.assertEqual(RadWC.transform(rxn), rxn)

    def test_single_radical_in_product(self):
        """A single radical in product gets a wildcard."""
        rxn = "[CH3:1][OH:2]>>[CH2:1].[OH:2]"
        out = RadWC.transform(rxn)
        # Check [*:3] is attached to [CH2:1]
        self.assertIn("[CH2:1]([*:3])", out)  # Atom-maps: 1,2 exist, so 3 is next

    def test_multiple_radicals_in_product(self):
        """Multiple radicals in product get multiple wildcards."""
        rxn = "[CH3:1][OH:2]>>[CH2:1].[O:2]"
        out = RadWC.transform(rxn)
        # [CH2:1] has *:3 and *:4, [O:2] has *:5
        self.assertIn("[CH2:1]([*:3])", out)
        self.assertIn("[O:2]([*:5])", out)

    def test_radical_and_nonradical_mixture(self):
        """Mixed radical/non-radical product fragments, only radicals get wildcard."""
        rxn = "[CH3:1][OH:2]>>[CH2:1].[OH:2]"
        out = RadWC.transform(rxn)
        # [CH2:1] gets *:3, [OH:2] is unchanged
        self.assertIn("[CH2:1]([*:3])", out)
        self.assertIn("[OH:2]", out)

    def test_user_start_map(self):
        """User-supplied map index is used for wildcards."""
        rxn = "[CH3:7][OH:8]>>[CH2:7].[OH:8]"
        out = RadWC.transform(rxn, start_map=50)
        self.assertIn("[CH2:7]([*:50])", out)

    def test_empty_reaction(self):
        """Empty input should raise ValueError."""
        with self.assertRaises(ValueError):
            RadWC.transform("")

    def test_three_component(self):
        """Agent block is preserved."""
        rxn = "[CH3:1][OH:2]>[Na+]>[CH2:1].[OH:2]"
        out = RadWC.transform(rxn)
        self.assertTrue(out.startswith("[CH3:1][OH:2]>[Na+]>"))
        self.assertIn("[CH2:1]([*:3])", out)
        self.assertIn("[OH:2]", out)


if __name__ == "__main__":
    unittest.main()
