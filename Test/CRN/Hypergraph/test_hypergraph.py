import unittest
import numpy as np

from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
from synkit.CRN.Hypergraph.hyperedge import HyperEdge


class TestCRNHyperGraph(unittest.TestCase):
    def test_add_rxn_and_basic_properties(self):
        H = CRNHyperGraph()
        e = H.add_rxn({"A": 1, "B": 2}, {"C": 1}, rule="r1", edge_id="e1")
        self.assertIsInstance(e, HyperEdge)
        self.assertIn("A", H.species)
        self.assertIn("C", H.species)
        self.assertIn("e1", H.edges)
        self.assertEqual(len(H), 1)
        self.assertTrue("A" in H)  # __contains__ supports species
        self.assertTrue("e1" in H)  # and edge ids

    def test_add_rxn_auto_id_and_duplicate_id_error(self):
        H = CRNHyperGraph()
        e1 = H.add_rxn({"X": 1}, {"Y": 1}, rule="rx")
        # adding with explicit same id should raise
        with self.assertRaises(KeyError):
            H.add_rxn({"X": 1}, {"Y": 1}, edge_id=e1.id)

    def test_add_rxn_both_sides_empty_raises(self):
        H = CRNHyperGraph()
        with self.assertRaises(ValueError):
            H.add_rxn({}, {})

    def test_add_rxn_from_str_and_suffix_rule_parsing(self):
        H = CRNHyperGraph()
        H.add_rxn_from_str("P + Q >> R | rule=MYRULE", parse_rule_from_suffix=True)
        # edge created and rule parsed
        self.assertEqual(len(H.edges), 1)
        e2 = next(iter(H.edge_list()))
        self.assertEqual(e2.rule, "MYRULE")

    def test_parse_rxns_with_rules_list_and_mapping(self):
        # list + rules parallel
        rxns = ["A >> B", "B >> C"]
        H = CRNHyperGraph()
        H.parse_rxns(rxns, rules=["R1", "R2"], parse_rule_from_suffix=False)
        self.assertEqual(len(H.edges), 2)
        # mapping form
        rxn_map = {"X >> Y": "m1", "Y >> Z": "m2"}
        H2 = CRNHyperGraph()
        H2.parse_rxns(rxn_map)
        self.assertEqual(len(H2.edges), 2)

    def test_parse_rxns_prefer_suffix_behavior(self):
        # explicit rule provided but suffix present; prefer_suffix True -> suffix overrides
        rxns = [("A >> B | rule=SUF", "EXPL")]
        H = CRNHyperGraph()
        H.parse_rxns(
            [r[0] for r in rxns],
            rules=[rxns[0][1]],
            prefer_suffix=True,
            parse_rule_from_suffix=True,
        )
        e = next(iter(H.edge_list()))
        self.assertEqual(e.rule, "SUF")

        # prefer_suffix False -> explicit wins
        H2 = CRNHyperGraph()
        H2.parse_rxns(
            [r[0] for r in rxns],
            rules=[rxns[0][1]],
            prefer_suffix=False,
            parse_rule_from_suffix=True,
        )
        e2 = next(iter(H2.edge_list()))
        self.assertEqual(e2.rule, "EXPL")

    def test_remove_rxn_and_species_pruning(self):
        H = CRNHyperGraph()
        H.add_rxn({"A": 1}, {"B": 1}, edge_id="e1")
        H.add_rxn({"B": 1}, {"C": 1}, edge_id="e2")
        self.assertIn("B", H.species)
        H.remove_rxn("e1")
        self.assertNotIn("e1", H.edges)
        # species A should be pruned (no incidence), B remains because e2 still references it
        self.assertNotIn("A", H.species)
        self.assertIn("B", H.species)

    def test_remove_species_and_edge_cleanup(self):
        H = CRNHyperGraph()
        H.add_rxn({"A": 1}, {"B": 1}, edge_id="e1")  # A >> B
        H.add_rxn({"B": 1}, {}, edge_id="e2")  # B >> ∅
        # remove species B should remove edges that become empty on both sides (e2),
        # while edges that keep some reactants or products (e1 -> A >> ∅) remain.
        H.remove_species("B", prune_orphans=True)

        # e2 had only B on reactant side and no products -> becomes empty and should be removed
        self.assertNotIn("e2", H.edges)

        # e1 had reactant A; after dropping B it becomes A >> ∅ and should still exist
        self.assertIn("e1", H.edges)
        e1 = H.get_edge("e1")
        # product side of e1 should be empty now
        self.assertEqual(e1.products.to_dict(), {})

        # B removed from species set
        self.assertNotIn("B", H.species)
        # A should still be present (still used by e1)
        self.assertIn("A", H.species)

    def test_neighbors_and_paths(self):
        H = CRNHyperGraph()
        H.parse_rxns(["A >> B", "B >> C", "C >> D", "A >> E"])
        nb = H.neighbors("A")
        self.assertTrue({"B", "E"}.issubset(nb))
        # path A->D should exist
        ps = H.paths("A", "D", max_hops=3)
        self.assertTrue(any(p[-1] == "D" for p in ps))
        # nonexistent species raises
        with self.assertRaises(KeyError):
            H.neighbors("Z")
        with self.assertRaises(KeyError):
            H.paths("A", "Z")

    def test_incidence_matrix_sparse_and_dense(self):
        H = CRNHyperGraph()
        H.parse_rxns(["A + B >> C", "C >> A"])
        species_order, edge_order, mapping = H.incidence_matrix(sparse=True)
        self.assertIn(("A", edge_order[0]), mapping)
        # dense form
        s2, e2, mat = H.incidence_matrix(sparse=False)
        self.assertEqual(species_order, s2)
        self.assertEqual(edge_order, e2)
        self.assertIsInstance(mat, np.ndarray)
        # mapping and dense should correspond
        s_idx = {s: i for i, s in enumerate(species_order)}
        for (s, eid), val in mapping.items():
            j = edge_order.index(eid)
            self.assertEqual(mat[s_idx[s], j], val)

    def test_stoichiometric_matrix_alias(self):
        H = CRNHyperGraph()
        H.add_rxn({"X": 2}, {"Y": 1}, edge_id="e1")
        s1, e1, m1 = H.stoichiometric_matrix(sparse=False)
        s2, e2, m2 = H.incidence_matrix(sparse=False)
        self.assertEqual(s1, s2)
        self.assertEqual(e1, e2)
        self.assertTrue(np.array_equal(m1, m2))

    def test_merge_and_copy(self):
        H1 = CRNHyperGraph()
        H1.add_rxn({"A": 1}, {"B": 1}, edge_id="e1")
        H2 = CRNHyperGraph()
        H2.add_rxn({"C": 1}, {"D": 1}, edge_id="e1")  # same id intentionally
        H1.merge(H2, prefix_edges=True)
        # merging with prefix should keep original edges and add new ones
        self.assertTrue(len(H1.edges) >= 2)
        # copy is deep
        H_copy = H1.copy()
        self.assertIsNot(H_copy, H1)
        # modifying copy should not affect original
        some_edge = next(iter(H_copy.edge_list()))
        some_edge.reactants["NEW_SPEC"] = 1
        # original should not contain NEW_SPEC unless it was there originally
        self.assertNotIn("NEW_SPEC", H1.species)

    def test_repr_contains_edges_and_species(self):
        H = CRNHyperGraph()
        H.add_rxn({"A": 1}, {"B": 1}, edge_id="eX")
        s = repr(H)
        self.assertIn("CRNHyperGraph:", s)
        self.assertIn("eX", s)
        self.assertIn("A", s)

    def test_parse_rxns_tuple_input_and_mapping_input(self):
        # tuple input with explicit per-line rule
        seq = [("A >> B", "rA"), ("B >> C", "rB")]
        H = CRNHyperGraph()
        H.parse_rxns(seq)
        self.assertEqual(len(H.edges), 2)
        # mapping input already tested above; verify mapping order-insensitivity
        mp = {"X >> Y": None, "Y >> Z": None}
        H2 = CRNHyperGraph()
        H2.parse_rxns(mp)
        self.assertEqual(len(H2.edges), 2)


if __name__ == "__main__":
    unittest.main()
