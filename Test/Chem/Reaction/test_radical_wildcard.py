import unittest
from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder


class TestRadicalWildcardAdder(unittest.TestCase):
    """
    Unit tests for RadicalWildcardAdder using unittest.
    """

    def test_transform_given_rxn(self):
        """
        Ensure that transform() applies the wildcard_map correctly and preserves explicit H.
        """
        rxn_in = (
            "[CH3:1][CH2:2][C:3](=[O:4])[OH:5]."
            "[O:6][H:7]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:6]."
            "[OH:5][H:7]"
        )
        adder = RadicalWildcardAdder()
        result = adder.transform(rxn_in)

        expected = (
            "[CH3:1][CH2:2][C:3](=[O:4])[OH:5]."
            "[O:6]([H:7])[*:8]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:6][*:8]."
            "[OH:5][H:7]"
        )
        self.assertEqual(result, expected)

    def test_auto_map_selection_and_repr_str(self):
        """
        Check that wildcard_map is auto-selected and repr/str methods report correctly.
        """
        rxn = (
            "[CH3:1][CH2:2][C:3](=[O:4])[OH:5]."
            "[O:6][H:7]>>"
            "[CH3:1][CH2:2][C:3](=[O:4])[O:6]."
            "[OH:5][H:7]"
        )
        adder = RadicalWildcardAdder()
        out = adder.transform(rxn)

        # The output should contain exactly one [*:8] in reactants and one in products
        reactants, products = out.split(">>")
        self.assertEqual(reactants.count("[*:8]"), 1)
        self.assertEqual(products.count("[*:8]"), 1)


if __name__ == "__main__":
    unittest.main()
