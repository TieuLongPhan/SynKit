# tests/test_rxn.py
import unittest

from synkit.CRN.Hypergraph.rxn import RXNSide


class TestRXNSide(unittest.TestCase):
    def test_from_any_mapping(self):
        rs = RXNSide.from_any({"A": 2, "B": 1})
        self.assertEqual(rs.to_dict(), {"A": 2, "B": 1})
        # negative or zero counts dropped
        rs2 = RXNSide.from_any({"A": -1, "C": 0, "D": 3})
        self.assertEqual(rs2.to_dict(), {"D": 3})

    def test_from_any_iterable_labels(self):
        rs = RXNSide.from_any(["A", "B", "A"])
        self.assertEqual(rs.to_dict(), {"A": 2, "B": 1})

    def test_from_any_iterable_pairs(self):
        rs = RXNSide.from_any([("A", 2), ("B", 3), ("A", 1)])
        self.assertEqual(rs.to_dict(), {"A": 3, "B": 3})

    def test_from_str_simple(self):
        r = RXNSide.from_str("A + B")
        self.assertEqual(r.to_dict(), {"A": 1, "B": 1})

    def test_from_str_coeff_and_star(self):
        r = RXNSide.from_str("2A + 3*B + C")
        # "3*B" treated like "3 B"
        self.assertEqual(r.to_dict(), {"A": 2, "B": 3, "C": 1})

    def test_from_str_attached_number_and_complex_names(self):
        r = RXNSide.from_str("10Fe+2Cl2")
        self.assertEqual(r.to_dict(), {"Fe": 10, "Cl2": 2})

    def test_from_str_space_separated(self):
        r = RXNSide.from_str("2 H2O + Na")
        self.assertEqual(r.to_dict(), {"H2O": 2, "Na": 1})

    def test_from_str_empty_and_empty_symbol(self):
        r = RXNSide.from_str("")
        self.assertEqual(r.to_dict(), {})
        r2 = RXNSide.from_str("∅")
        self.assertEqual(r2.to_dict(), {})

    def test_mapping_api_get_set_pop(self):
        r = RXNSide.from_any({"A": 2})
        self.assertEqual(r["A"], 2)
        r["A"] = 5
        self.assertEqual(r["A"], 5)
        # setting to 0 removes key
        r["A"] = 0
        self.assertNotIn("A", r.to_dict())

    def test_iter_len_contains_items_keys_values(self):
        r = RXNSide.from_any({"X": 1, "Y": 2})
        self.assertEqual(len(r), 2)
        self.assertIn("X", r)
        keys = set(r.keys())
        self.assertEqual(keys, {"X", "Y"})
        items = dict(r.items())
        self.assertEqual(items, {"X": 1, "Y": 2})

    def test_update_merging(self):
        r = RXNSide.from_any({"A": 1})
        r.update([("A", 2), ("B", 3)])
        self.assertEqual(r.to_dict(), {"A": 3, "B": 3})

    def test_copy_and_to_dict(self):
        r = RXNSide.from_any({"A": 2})
        r2 = r.copy()
        self.assertIsNot(r, r2)
        self.assertEqual(r2.to_dict(), {"A": 2})
        # modifying copy does not affect original
        r2["A"] = 1
        self.assertEqual(r.to_dict(), {"A": 2})

    def test_species_and_incr(self):
        r = RXNSide.from_any({"A": 2})
        self.assertEqual(r.species(), {"A"})
        r.incr("A", by=-1)
        self.assertEqual(r.to_dict(), {"A": 1})
        r.incr("A", by=-1)
        # now removed
        self.assertEqual(r.to_dict(), {})

    def test_arity_include_coeff_false_true(self):
        r = RXNSide.from_any({"A": 2, "B": 1})
        self.assertEqual(r.arity(include_coeff=False), 2)  # distinct species
        self.assertEqual(r.arity(include_coeff=True), 3)  # 2+1

    def test_expand(self):
        r = RXNSide.from_any({"A": 2, "B": 1})
        self.assertCountEqual(r.expand(), ["A", "A", "B"])

    def test_repr_sorted_and_empty(self):
        r = RXNSide.from_any({"B": 1, "A": 2})
        # __repr__ sorts keys, so expect "2A + B"
        self.assertEqual(repr(r), "2A + B")
        self.assertEqual(repr(RXNSide()), "∅")

    def test_normalize_mixed_inputs(self):
        # mixture of tuple-likes and labels
        r = RXNSide.from_any([("A", 2), "B", ("C", 0), ("D", -1), "B"])
        # C and D dropped due to nonpositive, B appears twice
        self.assertEqual(r.to_dict(), {"A": 2, "B": 2})

    def test_setitem_noninteger_coerced_and_removal(self):
        r = RXNSide()
        r["A"] = 3.0  # float coerced to int
        self.assertEqual(r.to_dict(), {"A": 3})
        r["A"] = -2  # negative => removal
        self.assertNotIn("A", r.to_dict())

    def test_from_str_edge_cases_spaces_and_tokens(self):
        # token like "3 FeCl3" (space separated) should be parsed correctly
        r = RXNSide.from_str("3 FeCl3 + Na")
        self.assertEqual(r.to_dict(), {"FeCl3": 3, "Na": 1})
        # token "02A" should parse as coefficient 2 and species "A" (leading zero)
        r2 = RXNSide.from_str("02A + B")
        self.assertEqual(r2.to_dict(), {"A": 2, "B": 1})

    def test_update_with_mapping_and_iterable(self):
        r = RXNSide.from_any({"A": 1})
        r.update({"A": 2, "B": 1})
        self.assertEqual(r.to_dict(), {"A": 3, "B": 1})
        r.update([("A", 1), ("C", 2)])
        self.assertEqual(r.to_dict(), {"A": 4, "B": 1, "C": 2})


if __name__ == "__main__":
    unittest.main()
