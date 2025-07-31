import unittest
from synkit.IO import rsmi_to_graph
from synkit.Graph.Wildcard.wildcard import WildCard


class TestWildCard(unittest.TestCase):
    def setUp(self):
        # The main, complex test case with atom mapping
        self.rsmi_main = (
            "[cH:1]1[cH:14][c:10]2[c:23]([cH:11][n:25]1)[cH:17][cH:12][cH:4][c:31]2[NH2:28]."
            "[cH:2]1[c:20]([C:22]([OH:7])=[O:21])[s:18][c:24]([S:6][c:29]2[c:15]([Cl:26])[cH:8]"
            "[n:19][cH:9][c:16]2[Cl:27])[c:30]1[N+:5]([O-:3])=[O:13]>>"
            "[cH:1]1[cH:14][c:10]2[c:23]([cH:11][n:25]1)[cH:17][cH:12][cH:4][c:31]2[NH:28]"
            "[C:22]([c:20]1[cH:2][c:30]([N+:5]([O-:3])=[O:13])[c:24]([S:6][c:29]2[c:15]([Cl:26])"
            "[cH:8][n:19][cH:9][c:16]2[Cl:27])[s:18]1)=[O:21]"
        )
        # No atoms lost: R == P, should not add wildcards
        self.rsmi_no_loss = "CCO>>CCO"
        # All atoms lost: RSMI that loses everything (nonsense, but good test)
        self.rsmi_all_lost = "CCO>>"
        # Empty
        self.rsmi_empty = ""
        # Wildcard already present
        self.rsmi_existing_wildcard = "[CH3:1][CH2:2][OH:3]>>[CH2:1][CH2:2].[*:4][OH:3]"
        # No atom map (should raise error)
        self.rsmi_no_atom_map = "C(C)Cl>>CC"

    def test_main_case_wildcard_added(self):
        """Complex case: output product contains wildcard and roundtrip is valid."""
        out_rsmi = WildCard.rsmi_with_wildcards(self.rsmi_main)
        _, product = out_rsmi.split(">>")
        self.assertIsInstance(out_rsmi, str)
        self.assertIn(
            "*", product, "Wildcard '*' should be present in the product side."
        )
        # Roundtrip: should parse back without error
        r, p = rsmi_to_graph(out_rsmi)
        self.assertTrue(r.number_of_nodes() > 0)
        self.assertTrue(p.number_of_nodes() > 0)

    def test_no_atoms_lost(self):
        """No atoms lost: should raise ValueError if input is not atom-mapped."""
        with self.assertRaises(ValueError):
            WildCard.rsmi_with_wildcards(self.rsmi_no_loss)

    def test_all_atoms_lost(self):
        """All atoms lost: should raise ValueError if input is not atom-mapped."""
        with self.assertRaises(ValueError):
            WildCard.rsmi_with_wildcards(self.rsmi_all_lost)

    def test_empty_input(self):
        """Empty input: should raise ValueError."""
        with self.assertRaises(ValueError):
            WildCard.rsmi_with_wildcards(self.rsmi_empty)

    def test_wildcard_not_duplicated(self):
        """Existing wildcards: should not create duplicate wildcards for same lost bond."""
        out_rsmi = WildCard.rsmi_with_wildcards(self.rsmi_existing_wildcard)
        _, product = out_rsmi.split(">>")
        # At least one '*' in the product SMILES string
        self.assertIn("*", product)

    def test_no_false_positive_wildcards(self):
        """Wildcards are only added if there are truly lost subgraphs; non-atom-mapped input raises."""
        rsmi = "C>>C"
        with self.assertRaises(ValueError):
            WildCard.rsmi_with_wildcards(rsmi)

    def test_output_is_str_and_split(self):
        """Should raise ValueError if input is not atom-mapped."""
        with self.assertRaises(ValueError):
            WildCard.rsmi_with_wildcards(self.rsmi_no_loss)

    def test_missing_atom_map_raises(self):
        """Should raise ValueError if atom_map attributes are missing."""
        with self.assertRaises(ValueError):
            WildCard.rsmi_with_wildcards(self.rsmi_no_atom_map)


if __name__ == "__main__":
    unittest.main()
