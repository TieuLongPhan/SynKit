import unittest

from synkit.CRN.Hypergraph.hyperedge import HyperEdge
from synkit.CRN.Hypergraph.rxn import RXNSide


class TestHyperEdge(unittest.TestCase):
    def test_init_with_dicts(self):
        e = HyperEdge("e1", {"A": 2, "B": 1}, {"C": 1}, rule="r1")
        self.assertEqual(e.id, "e1")
        self.assertIsInstance(e.reactants, RXNSide)
        self.assertIsInstance(e.products, RXNSide)
        self.assertEqual(e.reactants.to_dict(), {"A": 2, "B": 1})
        self.assertEqual(e.products.to_dict(), {"C": 1})
        self.assertEqual(e.rule, "r1")

    def test_init_with_iterables_and_pairs(self):
        # reactants as iterable of labels; products as (species,count) pairs
        e = HyperEdge("e2", ["A", "A", "B"], [("C", 2), ("D", 1)])
        self.assertEqual(e.reactants.to_dict(), {"A": 2, "B": 1})
        self.assertEqual(e.products.to_dict(), {"C": 2, "D": 1})

    def test_accept_RXNSide_instances(self):
        rside = RXNSide.from_any({"X": 3})
        pside = RXNSide.from_any(["Y", "Z"])
        e = HyperEdge("e3", rside, pside, rule="foo")
        # ensure reactants/products are preserved and not double-wrapped
        self.assertIs(e.reactants, rside)
        self.assertIs(e.products, pside)
        self.assertEqual(e.reactants.to_dict(), {"X": 3})
        self.assertEqual(e.products.to_dict(), {"Y": 1, "Z": 1})

    def test_species_union(self):
        e = HyperEdge("e4", {"A": 1}, {"B": 2, "A": 1})
        self.assertEqual(e.species(), {"A", "B"})

    def test_is_trivial_true_and_false(self):
        e1 = HyperEdge("e5", {"A": 2, "B": 1}, {"B": 1, "A": 2})
        self.assertTrue(e1.is_trivial())
        e2 = HyperEdge("e6", {"A": 1}, {"A": 2})
        self.assertFalse(e2.is_trivial())

    def test_arity_counts(self):
        e = HyperEdge("e7", {"A": 2, "B": 1}, {"C": 3})
        arity_simple = e.arity(include_coeff=False)
        arity_coeff = e.arity(include_coeff=True)
        # include_coeff=False counts distinct species: reactants 2, products 1
        self.assertEqual(arity_simple, (2, 1))
        # include_coeff=True sums coefficients: reactants 3, products 3
        self.assertEqual(arity_coeff, (3, 3))

    def test_copy_deep(self):
        e = HyperEdge("e8", {"A": 2}, {"B": 1}, rule="r8")
        e_copy = e.copy()
        # deep copy: not the same object
        self.assertIsNot(e, e_copy)
        self.assertIsNot(e.reactants, e_copy.reactants)
        self.assertIsNot(e.products, e_copy.products)
        # modifying the copy should not affect original
        e_copy.reactants["A"] = 1
        self.assertEqual(e.reactants.to_dict(), {"A": 2})
        self.assertEqual(e_copy.reactants.to_dict(), {"A": 1})

    def test_repr_contains_id_and_rule_and_arrow(self):
        e = HyperEdge("eid", {"X": 1}, {"Y": 2}, rule="Rk")
        s = repr(e)
        self.assertIn("eid", s)
        self.assertIn(">>", s)
        self.assertIn("rule=Rk", s)

    def test_productless_and_reactantless_allowed(self):
        # Edge with no reactants (creation)
        e1 = HyperEdge("create", {}, {"P": 1})
        self.assertEqual(e1.reactants.to_dict(), {})
        self.assertEqual(e1.products.to_dict(), {"P": 1})
        # Edge with no products (degradation)
        e2 = HyperEdge("destroy", {"Q": 1}, {})
        self.assertEqual(e2.reactants.to_dict(), {"Q": 1})
        self.assertEqual(e2.products.to_dict(), {})

    def test_species_returns_strings(self):
        e = HyperEdge("e9", {"Na": 1, "Cl": 2}, {"NaCl": 1})
        sp = e.species()
        self.assertTrue(all(isinstance(s, str) for s in sp))
        self.assertEqual(sp, {"Na", "Cl", "NaCl"})

    def test_mutation_of_initial_inputs_does_not_affect_edge(self):
        # If user passes a mutable dict and later modifies it, edge should keep its own copy
        rdict = {"A": 1}
        pdict = {"B": 2}
        e = HyperEdge("e10", rdict, pdict)
        # mutate originals
        rdict["A"] = 99
        pdict["B"] = 0
        # edge must be unaffected (because RXNSide._normalize_any used a copy)
        self.assertEqual(e.reactants.to_dict(), {"A": 1})
        self.assertEqual(e.products.to_dict(), {"B": 2})

    def test_edge_repr_sorted_species_order(self):
        # representation sorts species names
        e = HyperEdge("e11", {"B": 1, "A": 2}, {"C": 1})
        self.assertEqual(repr(e).split(":")[1].strip().split(">>")[0].strip(), "2A + B")


if __name__ == "__main__":
    unittest.main()
